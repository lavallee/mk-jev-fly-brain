// Model Kombat audio: an original fight theme synthesized with Web Audio (nothing borrowed from
// Mortal Kombat), recorded impacts and reactions from effects/bank (Pixabay, see effects/CREDITS.md;
// the rest of the effects are synthesized), plus an announcer baked once with ElevenLabs
// (scripts/bake_announcer.py -> voice/<set>/*.mp3). Browsers block audio until the viewer interacts,
// so nothing plays until unlock() runs from a click or key press.

const mtof = m => 440 * 2 ** ((m - 69) / 12);
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const MUSIC_LEVEL = 0.26;
const SAMPLE_GAIN = 0.55;   // recorded one-shots are peak-normalised; this sets them against the music
const DUCKED = 0.35;   // music level under the announcer, relative

// Fetch the recorded one-shots (punches, kicks, reactions). Resolves to { kind: [ArrayBuffer] }.
async function fetchBank() {
    try {
        const r = await fetch('effects/bank/manifest.json');
        if (!r.ok) return {};
        const { categories } = await r.json();
        const out = {};
        await Promise.all(Object.entries(categories).map(async ([kind, clips]) => {
            const bufs = await Promise.all(clips.map(async c => {
                const res = await fetch(`effects/bank/${c.key}.mp3`);
                return res.ok ? { key: c.key, buf: await res.arrayBuffer() } : null;
            }));
            out[kind] = bufs.filter(Boolean);
        }));
        return out;
    } catch {
        return {};
    }
}

// Fetch a baked announcer set (manifest + mp3s). Resolves to { key: ArrayBuffer }, or {} if absent.
async function fetchVoice(set) {
    try {
        const r = await fetch(`voice/${set}/manifest.json`);
        if (!r.ok) return {};
        const { lines } = await r.json();
        const entries = await Promise.all(Object.keys(lines).map(async key => {
            const res = await fetch(`voice/${set}/${key}.mp3`);
            return res.ok ? [key, await res.arrayBuffer()] : null;
        }));
        return Object.fromEntries(entries.filter(Boolean));
    } catch {
        return {};
    }
}

const BPM = 140;
const STEP = 60 / BPM / 4;                            // one 16th note, in seconds
const ROOTS = [45, 41, 43, 40];                       // Am, F, G, Em: one bar each
const CHORDS = [[57, 60, 64], [53, 57, 60], [55, 59, 62], [52, 55, 59]];
const BASS = { 0: 0, 2: 0, 3: 12, 6: 0, 8: 0, 10: 12, 11: 0, 14: 7 };  // step -> semitones above the root
// lead stabs: [step, chord tone, octave shift, length in steps]
const LEAD = [[0, 2, 0, 2], [3, 1, 0, 1], [6, 0, 12, 2], [10, 2, 0, 1], [12, 1, 12, 3]];
const LEAD_TURN = [[0, 2, 0, 2], [3, 1, 0, 1], [6, 0, 12, 2], [12, 0, 0, 1], [13, 1, 0, 1], [14, 2, 0, 1], [15, 0, 12, 1]];

function distortionCurve(k) {
    const n = 1024, c = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const x = (i * 2) / n - 1;
        c[i] = ((1 + k) * x) / (1 + k * Math.abs(x));
    }
    return c;
}

export class ArcadeAudio {
    constructor({ voice = true, announcer = 'daniel' } = {}) {
        this.voice = voice;
        this.ctx = null;
        this.mode = 'off';          // music: off | menu | fight | tense
        this.timer = null;
        this.step = 0;
        this.nextTime = 0;
        this.muted = false;
        try { this.muted = localStorage.getItem('mk.muted') === '1'; } catch { /* storage unavailable */ }
        // start downloading the announcer now so it's ready by the first gesture
        this.voiceData = voice && typeof fetch !== 'undefined' ? fetchVoice(announcer) : Promise.resolve({});
        this.bankData = typeof fetch !== 'undefined' ? fetchBank() : Promise.resolve({});
        this.samples = null;
        this.lastGrunt = {};
        this.clips = null;
        this.voiceSources = [];
        this.voiceEnd = 0;
        this.sayToken = 0;
    }

    // ---------- lifecycle ----------

    unlock() {
        if (!this.ctx) {
            const Ctx = window.AudioContext || window.webkitAudioContext;
            if (!Ctx) return;
            this._build(new Ctx());
        }
        if (this.ctx.state === 'suspended') this.ctx.resume();
        if (this.mode !== 'off') this._startScheduler();
    }

    setMuted(muted) {
        this.muted = muted;
        try { localStorage.setItem('mk.muted', muted ? '1' : '0'); } catch { /* storage unavailable */ }
        if (this.ctx) this.master.gain.setTargetAtTime(muted ? 0 : 0.85, this.ctx.currentTime, 0.02);
        if (muted) this._stopVoice();
    }

    _build(ctx) {
        this.ctx = ctx;
        this.offline = typeof OfflineAudioContext !== 'undefined' && ctx instanceof OfflineAudioContext;
        const comp = ctx.createDynamicsCompressor();
        comp.threshold.value = -16;
        comp.knee.value = 12;
        comp.ratio.value = 4;
        comp.attack.value = 0.003;
        comp.release.value = 0.2;
        comp.connect(ctx.destination);

        this.master = ctx.createGain();
        this.master.gain.value = this.muted ? 0 : 0.85;
        this.master.connect(comp);

        this.musicFilter = ctx.createBiquadFilter();
        this.musicFilter.type = 'lowpass';
        this.musicFilter.frequency.value = 2400;
        this.musicBus = ctx.createGain();
        this.musicBus.gain.value = MUSIC_LEVEL;
        this.musicBus.connect(this.musicFilter);
        this.musicFilter.connect(this.master);

        this.sfxBus = ctx.createGain();
        this.sfxBus.gain.value = 2;
        this.sfxBus.connect(this.master);

        const verb = ctx.createConvolver();
        verb.buffer = this._impulse(2.2);
        this.verbSend = ctx.createGain();
        this.verbSend.gain.value = 0.35;
        this.verbSend.connect(verb);
        verb.connect(this.master);

        this.noiseBuf = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
        const d = this.noiseBuf.getChannelData(0);
        for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
        this.distCurve = distortionCurve(30);

        // announcer: its own bus with a little arena reverb
        this.voiceBus = ctx.createGain();
        this.voiceBus.gain.value = 1.1;
        const hp = this._filter('highpass', 90);
        this.voiceBus.connect(hp);
        hp.connect(this.master);
        const vsend = ctx.createGain();
        vsend.gain.value = 0.3;
        hp.connect(vsend);
        vsend.connect(this.verbSend);
        this.bankReady = this.bankData.then(async data => {
            const out = {};
            await Promise.all(Object.entries(data).map(async ([kind, clips]) => {
                const decoded = await Promise.all(clips.map(async c => {
                    try { return { key: c.key, buf: await ctx.decodeAudioData(c.buf.slice(0)) }; } catch { return null; }
                }));
                out[kind] = decoded.filter(Boolean);
            }));
            this.samples = out;
            return out;
        });
        this.clipsReady = this.voiceData.then(async data => {
            const out = {};
            await Promise.all(Object.entries(data).map(async ([key, buf]) => {
                try { out[key] = await ctx.decodeAudioData(buf.slice(0)); } catch { /* skip an undecodable clip */ }
            }));
            this.clips = out;
            return out;
        });
    }

    _impulse(seconds) {
        const len = Math.floor(this.ctx.sampleRate * seconds);
        const buf = this.ctx.createBuffer(2, len, this.ctx.sampleRate);
        for (let ch = 0; ch < 2; ch++) {
            const d = buf.getChannelData(ch);
            for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / len) ** 3;
        }
        return buf;
    }

    _live() {
        return this.ctx && !this.muted && (this.offline || this.ctx.state === 'running');
    }

    get now() { return this.ctx.currentTime + 0.005; }

    // ---------- building blocks ----------

    _env(param, t, peak, attack, decay) {
        param.setValueAtTime(0.0001, t);
        param.exponentialRampToValueAtTime(Math.max(peak, 0.0002), t + attack);
        param.exponentialRampToValueAtTime(0.0001, t + attack + decay);
    }

    _osc(type, freq, t, stop, detune = 0) {
        const o = this.ctx.createOscillator();
        o.type = type;
        o.frequency.setValueAtTime(freq, t);
        o.detune.value = detune;
        o.start(t);
        o.stop(stop);
        return o;
    }

    _noise(t, dur) {
        const s = this.ctx.createBufferSource();
        s.buffer = this.noiseBuf;
        s.start(t, Math.random() * 1.2, dur + 0.05);
        return s;
    }

    _filter(type, freq, q = 0.7) {
        const f = this.ctx.createBiquadFilter();
        f.type = type;
        f.frequency.value = freq;
        f.Q.value = q;
        return f;
    }

    // Sound-effect output: panned (-1 left .. 1 right) with an optional reverb send.
    _out(pan = 0, verb = 0.12) {
        const g = this.ctx.createGain();
        const p = this.ctx.createStereoPanner();
        p.pan.value = clamp(pan, -1, 1);
        g.connect(p);
        p.connect(this.sfxBus);
        if (verb > 0) {
            const s = this.ctx.createGain();
            s.gain.value = verb;
            p.connect(s);
            s.connect(this.verbSend);
        }
        return g;
    }

    // ---------- music instruments ----------

    _kick(t, vel, dest) {
        const g = this.ctx.createGain();
        const o = this._osc('sine', 165, t, t + 0.42);
        o.frequency.exponentialRampToValueAtTime(44, t + 0.11);
        this._env(g.gain, t, vel, 0.002, 0.36);
        o.connect(g);
        g.connect(dest);
    }

    _clap(t, vel, dest) {
        const f = this._filter('bandpass', 1400, 0.9);
        f.connect(dest);
        [0, 0.011, 0.022].forEach((dt, i) => {
            const g = this.ctx.createGain();
            const len = i === 2 ? 0.16 : 0.02;
            this._env(g.gain, t + dt, vel * 0.9, 0.001, len);
            this._noise(t + dt, len + 0.02).connect(g);
            g.connect(f);
        });
    }

    _hat(t, vel, open, dest) {
        const f = this._filter('highpass', 7500);
        const g = this.ctx.createGain();
        const len = open ? 0.2 : 0.035;
        this._env(g.gain, t, vel * 0.28, 0.001, len);
        this._noise(t, len + 0.02).connect(f);
        f.connect(g);
        g.connect(dest);
    }

    _bass(t, midi, len, dest, vel = 1) {
        const f = this._filter('lowpass', 1600, 7);
        f.frequency.setValueAtTime(1600, t);
        f.frequency.exponentialRampToValueAtTime(180, t + len);
        const g = this.ctx.createGain();
        this._env(g.gain, t, 0.32 * vel, 0.004, len);
        const sub = this.ctx.createGain();
        sub.gain.value = 0.5;
        this._osc('sawtooth', mtof(midi), t, t + len + 0.05).connect(f);
        this._osc('square', mtof(midi - 12), t, t + len + 0.05).connect(sub);
        sub.connect(f);
        f.connect(g);
        g.connect(dest);
    }

    _lead(t, midi, len, vel) {
        const f = this._filter('lowpass', 2800, 2);
        const g = this.ctx.createGain();
        this._env(g.gain, t, 0.09 * vel, 0.004, len * 1.1);
        for (const dt of [-6, 6]) this._osc('square', mtof(midi), t, t + len * 1.1 + 0.05, dt).connect(f);
        f.connect(g);
        g.connect(this.musicBus);
        const send = this.ctx.createGain();
        send.gain.value = 0.4;
        g.connect(send);
        send.connect(this.verbSend);
    }

    _pad(t, midis, len) {
        const f = this._filter('lowpass', 1100);
        const g = this.ctx.createGain();
        g.gain.setValueAtTime(0.0001, t);
        g.gain.linearRampToValueAtTime(0.035, t + 0.5);
        g.gain.linearRampToValueAtTime(0.0001, t + len + 0.4);
        for (const m of midis) for (const dt of [-9, 9]) this._osc('sawtooth', mtof(m), t, t + len + 0.5, dt).connect(f);
        f.connect(g);
        g.connect(this.musicBus);
        g.connect(this.verbSend);
    }

    // ---------- music sequencer ----------

    // off: silence; menu: pads and a slow arpeggio; fight: the full theme; tense: fight plus 16th hats
    music(mode) {
        if (mode === this.mode) return;
        this.mode = mode;
        if (!this.ctx) return;
        const t = this.ctx.currentTime;
        this.musicFilter.frequency.cancelScheduledValues(t);
        this.musicFilter.frequency.setTargetAtTime(mode === 'menu' ? 2400 : 14000, t, 0.15);
        if (mode === 'off') {
            clearInterval(this.timer);
            this.timer = null;
            return;
        }
        if (mode === 'fight') {  // start the theme on its downbeat
            this.step = 0;
            this.nextTime = t + 0.05;
        }
        this._startScheduler();
    }

    _startScheduler() {
        if (this.timer || !this.ctx) return;
        this.nextTime = Math.max(this.nextTime, this.ctx.currentTime + 0.05);
        this.timer = setInterval(() => this._tick(), 25);
    }

    _tick() {
        if (!this.ctx || this.ctx.state !== 'running') return;
        // a throttled background tab falls behind; skip ahead instead of firing a burst of notes
        if (this.nextTime < this.ctx.currentTime - 0.2) this.nextTime = this.ctx.currentTime + 0.02;
        while (this.nextTime < this.ctx.currentTime + 0.12) {
            if (!this.muted) this._scheduleStep(this.step, this.nextTime);
            this.nextTime += STEP;
            this.step = (this.step + 1) % 64;
        }
    }

    _scheduleStep(step, t) {
        const bar = Math.floor(step / 16) % 4, s = step % 16;
        const root = ROOTS[bar], chord = CHORDS[bar], M = this.musicBus;
        if (this.mode === 'menu') {
            if (s === 0) {
                this._pad(t, chord.map(n => n - 12), STEP * 16);
                this._kick(t, 0.45, M);
            }
            if (s % 4 === 2) this._lead(t, chord[(s >> 2) % 3] + 12, STEP * 3, 0.4);
            if (s === 8) this._bass(t, root, STEP * 6, M, 0.6);
            return;
        }
        const tense = this.mode === 'tense';
        if (s % 4 === 0) this._kick(t, 1, M);
        if (tense && bar === 3 && s >= 14) this._kick(t, 0.6, M);
        if (s === 4 || s === 12) this._clap(t, 0.8, M);
        if (bar === 3 && s >= 13 && s !== 12) this._clap(t, 0.35 + (s - 13) * 0.12, M);
        if (s % 4 === 2) this._hat(t, 0.9, s === 14, M);
        else if (tense && s % 2 === 1) this._hat(t, 0.45, false, M);
        if (step === 0) this._hat(t, 1.3, true, M);
        if (BASS[s] !== undefined) this._bass(t, root + BASS[s], STEP * (s === 14 ? 2 : 1.2), M);
        for (const [st, tone, oct, len] of bar === 3 ? LEAD_TURN : LEAD) {
            if (st === s) this._lead(t, chord[tone] + oct, STEP * len, tense ? 1 : 0.8);
        }
    }

    // ---------- sound effects ----------

    _whoosh(t, pan, dur, f0, f1, vel) {
        const out = this._out(pan, 0.08);
        const f = this._filter('bandpass', f0, 1.3);
        f.frequency.setValueAtTime(f0, t);
        f.frequency.exponentialRampToValueAtTime(f1, t + dur);
        const g = this.ctx.createGain();
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(vel * 0.5, t + dur * 0.35);
        g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
        this._noise(t, dur).connect(f);
        f.connect(g);
        g.connect(out);
    }

    _thump(t, pan, f0, f1, dur, vel, verb = 0.1) {
        const out = this._out(pan, verb);
        const g = this.ctx.createGain();
        const o = this._osc('sine', f0, t, t + dur + 0.05);
        o.frequency.exponentialRampToValueAtTime(f1, t + dur * 0.6);
        this._env(g.gain, t, vel, 0.002, dur);
        o.connect(g);
        g.connect(out);
        const f = this._filter('lowpass', 800);
        const ng = this.ctx.createGain();
        this._env(ng.gain, t, vel * 0.5, 0.001, dur * 0.4);
        this._noise(t, dur * 0.5).connect(f);
        f.connect(ng);
        ng.connect(out);
    }

    _crack(t, pan, vel) {
        const out = this._out(pan, 0.05);
        const f = this._filter('highpass', 2200);
        const g = this.ctx.createGain();
        this._env(g.gain, t, vel * 0.6, 0.001, 0.04);
        this._noise(t, 0.06).connect(f);
        f.connect(g);
        g.connect(out);
    }

    // Play a recorded one-shot with a little pitch and level variation, panned where it happened.
    _sample(kind, pan, { gain = 1, rate = 1, spread = 0.06, pick = null } = {}) {
        const clips = this.samples?.[kind];
        if (!clips || !clips.length) return false;
        const clip = pick ? clips.find(c => c.key === pick) || clips[0] : clips[Math.floor(Math.random() * clips.length)];
        const src = this.ctx.createBufferSource();
        src.buffer = clip.buf;
        src.playbackRate.value = rate * (1 + (Math.random() * 2 - 1) * spread);
        const g = this.ctx.createGain();
        g.gain.value = SAMPLE_GAIN * gain;
        src.connect(g);
        g.connect(this._out(pan, 0.14));
        src.start(this.now);
        return true;
    }

    // an attack being thrown (lands or not)
    swing(kind, pan) {
        if (!this._live()) return;
        const kick = kind === 'kick';
        this._whoosh(this.now, pan, kick ? 0.24 : 0.14, 350, kick ? 1600 : 2400, 0.9);
    }

    // a clean hit; power 1 = a full kick. Recorded impact if the bank loaded, else the synth one.
    hit(kind, pan, power = 1) {
        if (!this._live()) return;
        const t = this.now, p = clamp(power, 0.3, 1.3), kick = kind === 'kick';
        if (this._sample(kind, pan, { gain: (kick ? 1 : 1.4) * (0.55 + 0.5 * p), rate: kick ? 0.98 : 1.02 })) {
            this._thump(t, pan, kick ? 110 : 150, 40, 0.14, 0.35 * p, 0.05);  // a little low end under the recording
            return;
        }
        this._thump(t, pan, kick ? 120 : 160, 42, kick ? 0.32 : 0.22, 0.75 * p);
        this._crack(t, pan, 0.7 * p);
    }

    // the fighter who just took an unblocked hit: 'fly' grunts "ough", 'jev' "umph"
    grunt(who, pan, power = 1) {
        if (!this._live()) return;
        const now = performance.now();
        if (now - (this.lastGrunt[who] || 0) < 260) return;
        this.lastGrunt[who] = now;
        this._sample('grunt', pan, { gain: 0.5 + 0.5 * clamp(power, 0.3, 1.3), rate: who === 'fly' ? 1.04 : 0.94,
            spread: 0.05, pick: who === 'fly' ? 'grunt_ough' : 'grunt_umph' });
    }

    // a hit absorbed by a guard
    block(pan) {
        if (!this._live()) return;
        const t = this.now, out = this._out(pan, 0.35);
        for (const [freq, type, vel, dur] of [[620, 'triangle', 0.36, 0.4], [932, 'square', 0.12, 0.26], [1397, 'triangle', 0.2, 0.2], [2093, 'sine', 0.1, 0.12]]) {
            const g = this.ctx.createGain();
            this._env(g.gain, t, vel, 0.001, dur);
            this._osc(type, freq, t, t + dur + 0.05).connect(g);
            g.connect(out);
        }
        this._crack(t, pan, 0.7);
        this._thump(t, pan, 260, 140, 0.08, 0.35, 0);
    }

    jump(pan) {
        if (!this._live()) return;
        this._whoosh(this.now, pan, 0.3, 280, 2000, 0.8);
    }

    ko() {
        if (!this._live()) return;
        const t = this.now;
        this._thump(t, 0, 95, 26, 1.3, 0.85, 0.5);
        const f = this._filter('lowpass', 260);
        const g = this.ctx.createGain();
        this._env(g.gain, t, 0.7, 0.01, 1.6);
        this._noise(t, 1.7).connect(f);
        f.connect(g);
        g.connect(this._out(0, 0.6));
        this._crack(t, 0, 0.9);
        this._sample('grunt', 0, { gain: 1.2, rate: 0.82, spread: 0.02 });  // the loser goes down
        this.musicFilter.frequency.setTargetAtTime(350, t, 0.25);
    }

    // round bell: inharmonic partials with long decays
    bell() {
        if (!this._live()) return;
        const t = this.now, out = this._out(0, 0.6);
        for (const [ratio, vel, dur] of [[1, 0.22, 2.8], [2.01, 0.1, 1.9], [2.76, 0.08, 1.3], [5.4, 0.04, 0.7], [8.9, 0.02, 0.35]]) {
            const g = this.ctx.createGain();
            this._env(g.gain, t, vel, 0.003, dur);
            this._osc('sine', 147 * ratio, t, t + dur + 0.05).connect(g);
            g.connect(out);
        }
    }

    // "FIGHT!" stinger: a distorted power chord, a kick and a crash
    fight() {
        if (!this._live()) return;
        const t = this.now, out = this._out(0, 0.45);
        const shaper = this.ctx.createWaveShaper();
        shaper.curve = this.distCurve;
        const f = this._filter('lowpass', 2800);
        const g = this.ctx.createGain();
        this._env(g.gain, t, 0.16, 0.005, 0.8);
        for (const m of [45, 52, 57, 64]) this._osc('sawtooth', mtof(m), t, t + 0.9, Math.random() * 10 - 5).connect(shaper);
        shaper.connect(f);
        f.connect(g);
        g.connect(out);
        this._kick(t, 0.7, this.sfxBus);
        const cf = this._filter('highpass', 5000);
        const cg = this.ctx.createGain();
        this._env(cg.gain, t, 0.2, 0.002, 1.4);
        this._noise(t, 1.5).connect(cf);
        cf.connect(cg);
        cg.connect(this._out(0, 0.3));
    }

    // coin-in blip when a match starts
    coin() {
        if (!this._live()) return;
        const t = this.now, out = this._out(0, 0.1);
        const g1 = this.ctx.createGain();
        g1.gain.value = 0.16;
        this._osc('square', 988, t, t + 0.07).connect(g1);
        g1.connect(out);
        const g2 = this.ctx.createGain();
        this._env(g2.gain, t + 0.07, 0.16, 0.002, 0.28);
        this._osc('square', 1319, t + 0.07, t + 0.4).connect(g2);
        g2.connect(out);
    }

    // last-seconds countdown tick
    tick() {
        if (!this._live()) return;
        const t = this.now, g = this.ctx.createGain();
        this._env(g.gain, t, 0.28, 0.002, 0.06);
        this._osc('sine', 1100, t, t + 0.1).connect(g);
        g.connect(this._out(0, 0));
    }

    victory() {
        if (!this._live()) return;
        const t = this.now, out = this._out(0, 0.4);
        const f = this._filter('lowpass', 2600);
        f.connect(out);
        [57, 60, 64, 69].forEach((m, i) => {
            const start = t + i * 0.13, len = i === 3 ? 0.9 : 0.12;
            const g = this.ctx.createGain();
            this._env(g.gain, start, 0.1, 0.004, len);
            this._osc('sawtooth', mtof(m), start, start + len + 0.05).connect(g);
            this._osc('square', mtof(m), start, start + len + 0.05, 7).connect(g);
            g.connect(f);
        });
    }

    // round over: muffle the music
    roundEnd() {
        if (!this.ctx) return;
        this.musicFilter.frequency.setTargetAtTime(600, this.ctx.currentTime, 0.3);
    }

    // ---------- announcer ----------

    // Play baked lines back to back, e.g. say('time', 'jev_wins'). A new call interrupts the old one.
    // Missing keys are skipped (the bake covers rounds 1-9).
    say(...keys) {
        this.saying = this._say(keys);
        return this.saying;
    }

    async _say(keys) {
        if (!this.voice || !this.ctx) return;
        const token = ++this.sayToken;
        const clips = await this.clipsReady;
        if (token !== this.sayToken || !this._live()) return;
        this._stopVoice();
        const start = this.now;
        let t = start;
        for (const key of keys) {
            const buf = clips[key];
            if (!buf) continue;
            const src = this.ctx.createBufferSource();
            src.buffer = buf;
            src.connect(this.voiceBus);
            src.start(t);
            this.voiceSources.push(src);
            t += buf.duration + 0.06;
        }
        if (t === start) return;
        this.voiceEnd = t;
        const music = this.musicBus.gain;
        music.cancelScheduledValues(start);
        music.setTargetAtTime(MUSIC_LEVEL * DUCKED, start, 0.04);
        music.setTargetAtTime(MUSIC_LEVEL, t, 0.25);
    }

    // Resolves when the announcer has finished (capped), so the next cue doesn't talk over it.
    async voiceIdle(capMs = 7000) {
        await this.saying;  // a line still decoding hasn't set voiceEnd yet
        if (!this.ctx || this.muted) return;
        const ms = Math.min(capMs, Math.max(0, (this.voiceEnd - this.ctx.currentTime) * 1000));
        await new Promise(r => setTimeout(r, ms));
    }

    _stopVoice() {
        for (const src of this.voiceSources) {
            try { src.stop(); } catch { /* already stopped */ }
        }
        this.voiceSources = [];
        this.voiceEnd = 0;
        if (this.ctx) {
            const music = this.musicBus.gain;
            music.cancelScheduledValues(this.ctx.currentTime);
            music.setTargetAtTime(MUSIC_LEVEL, this.ctx.currentTime, 0.1);
        }
    }
}
