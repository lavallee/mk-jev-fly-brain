// Fly brain (maleCNS LIF) vs Jev (TypeSafe) in mk.js.
// Both fighters decide on the same 100 ms clock from the same game state and
// pick from the same seven moves. URL params: ?rounds=5&seconds=60&seed=1&auto=1

import { MOTORS } from './flybrain.js';
import { ArcadeAudio } from './audio.js';
import { features, LocalPolicy, ACTIONS as POLICY_ACTIONS } from './policy.js';

const mk = window.mk;
const T = mk.moves.types;
const params = new URLSearchParams(location.search);
const ROUNDS = Number(params.get('rounds') || 5);
const ROUND_S = Number(params.get('seconds') || 60);
const SEED = Number(params.get('seed') || 1);
const TICK_MS = 100;
const TRACE = Boolean(params.get('trace'));
// ?announcer=daniel (announcer mode, default) or callum; ?voice=0 silences the announcer
const audio = new ArcadeAudio({ voice: params.get('voice') !== '0', announcer: params.get('announcer') || 'daniel' });
const LEARN = params.get('learn') !== '0';  // dopamine-like learning on by default; ?learn=0 for the reflex-only fly
const FLY_EVERY = Math.max(1, Math.round(Number(params.get('flyevery')) || 1));  // decide on every Nth tick (handicap)
// Off by default: the fly acts the moment it decides. ?lag=auto makes it act on what it sensed one
// Jev-round-trip ago (its live median), ?lag=250 pins a delay in ms. Jev can't react to an attack in
// flight - a punch lands 80 ms after it starts, a kick 160 ms - and with lag on, neither can the fly.
const LAG = params.get('lag') || 'off';
const LAG_FALLBACK = 350;
// Jev's answer always arrives after the fight has moved on. ?lead=auto sends it the state as it will
// be when the answer lands (its own live median latency ahead); ?lead=250 pins the horizon; ?lead=dist
// projects only where the fighters will be and keeps what the opponent is doing now; off = as-is.
const LEAD = params.get('lead') || 'off';
const LEAD_DIST_ONLY = LEAD === 'dist';
// ?log=jev records every Jev decision as (feature vector, its probabilities) for training a local
// policy offline: scripts/train_local_policy.py. Off by default; a 7-round match adds ~0.5 MB.
const LOG_JEV = (params.get('log') || '').includes('jev');
const LOG_FLY = (params.get('log') || '').includes('fly');  // same, for the fly's own decisions
// ?jev=api (default) has every move come from the API. ?jev=hybrid lets the distilled local policy
// move on the fly's 100 ms clock while Jev keeps answering in the background: each answer corrects
// the move if it disagrees and teaches the policy one gradient step. ?jev=local never calls the API.
const JEV_MODE = params.get('jev') || 'api';  // api | hybrid | local | rules
const TEACH = params.get('teach') !== '0';        // in hybrid, learn from each Jev answer
const LOCAL_SURE = Number(params.get('sure')) || 0.15;  // margin at which the local policy is "confident"
// ?plastic=500 also makes the synapses onto the 500 neurons that drive the pools hardest learnable.
const PREMOTOR = Math.max(0, Math.round(Number(params.get('plastic')) || 0));
// ?circuit=rewired swaps in the control built by `build_malecns_fighter.py --rewire`: same neurons,
// degrees, signs and weights, but who connects to whom is shuffled. The test of whether the real
// wiring is doing any work.
const CIRCUIT = params.get('circuit') === 'rewired' ? 'rewired' : 'real';
// ?evolve=N plays N matches back to back, each starting from the best brain saved so far, so
// within-match learning stacks across generations. ?lineage=0 restarts from the wiring each time
// (the control: same number of matches, no inheritance).
const EVOLVE = Math.max(0, Math.round(Number(params.get('evolve')) || 0));
const LINEAGE = params.get('lineage') !== '0';
const Z_TH = 4;            // a motor pool must beat its resting baseline by 4 SD to act
// ?thresh=0.15 makes that bar learnable per pool: landing a hit with a pool lowers its bar (fire
// more readily), taking damage raises it (be choosier). Gains decide which reflex shouts loudest;
// this decides when a reflex is worth acting on at all — the one thing pool gains cannot express.
const THRESH_LR = Number(params.get('thresh')) || 0;
const THRESH_RANGE = [1.5, 20];
const JEV_MAX_INFLIGHT = 3;

const ACTIONS = ['stand', 'walk_forward', 'walk_backward', 'jump_away', 'punch', 'kick', 'block'];
const POOL_ACTION = { fwd: 'walk_forward', back: 'walk_backward', jump: 'jump_away', punch: 'punch', kick: 'kick', wing: 'block' };
const POOL_LABEL = { fwd: 'DNp09 walk fwd', back: 'MDN walk back', jump: 'TTMn jump', punch: 'Front legs', kick: 'Hind legs', wing: 'Wings' };
const SENSOR_LABEL = { loom: 'LPLC2+LC4 loom', object: 'LC9 motion', target: 'LC10a target', recede: 'LPLC4 recede', p1: 'P1 arousal', taste: 'Foreleg taste', body_touch: 'Body touch' };
const ATTACKS = new Set([T.HIGH_PUNCH, T.LOW_PUNCH, T.HIGH_KICK, T.LOW_KICK, T.UPPERCUT, T.SPIN_KICK,
    T.SQUAT_LOW_KICK, T.SQUAT_HIGH_KICK, T.SQUAT_LOW_PUNCH, T.FORWARD_JUMP_KICK, T.BACKWARD_JUMP_KICK,
    T.FORWARD_JUMP_PUNCH, T.BACKWARD_JUMP_PUNCH]);

const $ = id => document.getElementById(id);
const clamp01 = x => Math.max(0, Math.min(1, x));
const round1 = x => Math.round(x * 10) / 10;
// Float32Array <-> base64 (little-endian bytes), for saving learned weights as JSON
function toB64(f32) {
    const u8 = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
    let bin = '';
    for (let i = 0; i < u8.length; i += 0x8000) bin += String.fromCharCode(...u8.subarray(i, i + 0x8000));
    return btoa(bin);
}
function fromB64(b64) {
    const bin = atob(b64), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(u8.buffer);
}
const panOf = f => clamp01((f.getX() + f.getVisibleWidth() / 2) / 600) * 1.6 - 0.8;

// ---------- UI helpers ----------
function makeRows(el, keys, labels, { learned = false } = {}) {
    el.innerHTML = '';
    const rows = {};
    for (const k of keys) {
        const row = document.createElement('div');
        row.className = 'row';
        row.innerHTML = `<span class="name">${labels[k] || k}</span><div class="track"><div></div></div><span class="val">–</span>`
            + (learned ? '<span class="learned">–</span>' : '');
        el.appendChild(row);
        rows[k] = { row, bar: row.children[1].firstChild, val: row.children[2], learned: learned ? row.children[3] : null };
    }
    return rows;
}

// learned strength (1 = as wired) shown next to the pool it belongs to
function setLearned(r, ratio) {
    if (!r.learned) return;
    const pct = Math.round((ratio - 1) * 100);
    r.learned.textContent = `${pct > 0 ? '+' : ''}${pct}%`;
    r.learned.classList.toggle('up', pct > 0);
    r.learned.classList.toggle('down', pct < 0);
}
function setRow(r, frac, text, win = false) {
    r.bar.style.width = `${Math.round(clamp01(frac) * 100)}%`;
    r.val.textContent = text;
    r.row.classList.toggle('win', win);
}
const pretty = a => a.replace('_', ' ');

// ---------- geometry shared by both agents ----------
function view(me, opp, prev) {
    // Body centre from the fixed collision box. mk.js draws every sprite from its left edge, so
    // the image width (getWidth) grows during kicks, blocks and hit reactions, which slides a
    // width-based "centre" to the right. That made the opponent's strikes look like an approach
    // only when the fly stood on the right, firing its looming/escape-jump circuit.
    const body = f => f.getX() + f.getVisibleWidth() / 2;
    const dist = Math.abs(body(me) - body(opp));
    // mk.js's own hit test does use image width; keep it so in-range matches the game
    const sprite = f => f.getX() + f.getWidth() / 2;
    const move = opp.getMove() ? opp.getMove().type : T.STAND;
    return {
        dist,
        dDist: prev ? dist - prev.dist : 0,
        oppRight: opp.getX() > me.getX(),
        oppMove: move,
        oppAttacking: ATTACKS.has(move),
        // same rule mk.js uses to decide whether a standard attack connects
        inRange: Math.abs(sprite(me) - sprite(opp)) <= opp.getWidth(),
        oppWidth: opp.getWidth(),
        oppImage: opp.getWidth(),   // sprite width: grows as a limb extends, shrinks as it is pulled back
    };
}

// Returns 'started' if the move begins now, 'holding' if the fighter was already doing it, or false
// if it was ignored (a fighter mid-attack or mid-jump ignores new moves).
function applyAction(f, action, v) {
    const toward = v.oppRight ? T.WALK : T.WALK_BACKWARD;
    const away = v.oppRight ? T.WALK_BACKWARD : T.WALK;
    const move = {
        stand: T.STAND, walk_forward: toward, walk_backward: away,
        jump_away: v.oppRight ? T.BACKWARD_JUMP : T.FORWARD_JUMP,
        punch: T.HIGH_PUNCH, kick: T.HIGH_KICK, block: T.BLOCK,
    }[action];
    if (!move || f.getLife() <= 0) return false;
    const before = f.getMove();
    if (before && before.type === move) return 'holding';
    f.setMove(move);
    return f.getMove() !== before ? 'started' : false;
}

// ---------- fly agent ----------
// The brain runs in flyworker.js. Like Jev, it answers asynchronously: a tick is
// skipped if the previous 100 ms of spiking hasn't finished simulating yet.
const OPP_HEIGHT_PX = 120;     // apparent height of the opponent's body
// The giant-fiber escape responds to fast looms (size/speed of roughly 10-80 ms), not to something
// walking in slowly. Expansion below LOOM_THRESHOLD rad per 100 ms (a walk-in from beyond ~70 px)
// doesn't drive the looming detectors; LOOM_RANGE more saturates them (a jump-in or lunge).
const LOOM_THRESHOLD = 0.12;
const LOOM_RANGE = 0.25;
const MOVING = new Set([T.WALK, T.WALK_BACKWARD, T.JUMP, T.FORWARD_JUMP, T.BACKWARD_JUMP, ...ATTACKS]);

class FlyAgent {
    constructor() {
        this.worker = new Worker('flyworker.js', { type: 'module' });
        this.sRows = makeRows($('flySensors'), Object.keys(SENSOR_LABEL), SENSOR_LABEL);
        this.mRows = makeRows($('flyMotors'), MOTORS, POOL_LABEL, { learned: LEARN });
        this.rewards = { pos: 0, neg: 0 };
        this.hitUntil = 0;
        this.loomUntil = 0;
        this.prevOppAttacking = false;
        this.gain = null;
        this.thresh = Object.fromEntries(MOTORS.map(k => [k, Z_TH]));
        this.lastPool = null;
        this.decisions = 0;
        this.skipped = 0;
        this.ms = [];
        this.inflight = false;
        this.pending = null;
        this.seq = 0;
        this.applied = 0;
        this.stale = 0;
        this.reactions = [];
        this.lagOf = () => (LAG === 'off' ? 0 : LAG === 'auto' ? LAG_FALLBACK : Number(LAG) || 0);
        this.calls = new Map();
        this.callId = 0;
        this.meta = null;
        this.worker.onmessage = ({ data }) => this._onMessage(data);
        this.ready = new Promise(res => { this._ready = res; });
        this.worker.postMessage({ type: 'init', seed: SEED, zFloor: Z_TH, learn: LEARN, premotor: PREMOTOR, circuit: CIRCUIT, lr: Number(params.get('lr')) || undefined,
            baseline: Number(params.get('baseline')) || undefined });
    }

    _onMessage(m) {
        if (m.type === 'ready') {
            this.gain = m.gain;
            console.log('fly pool gain (z at 100 Hz full drive)', JSON.stringify(m.gain));
            this._initRaster(m.rasterMotor);
            for (const k of MOTORS) setLearned(this.mRows[k], 1);
            this.meta = m.meta;
            this._ready(m.meta);
        } else if (m.type === 'weights' || m.type === 'weightsSet') {
            const res = this.calls.get(m.id);
            this.calls.delete(m.id);
            res?.(m);
        } else if (m.type === 'learned') {
            this.strength = m.strength;
            for (const k of MOTORS) setLearned(this.mRows[k], m.strength[k]);
            $('flyDopa').textContent = `+${Math.round(this.rewards.pos)}/−${Math.round(this.rewards.neg)}`;
        } else if (m.type === 'rested') {
            this._rested?.();
        } else if (m.type === 'step') {
            this.inflight = false;
            const cb = this.pending;
            this.pending = null;
            this.ms.push(m.ms);
            if (this.ms.length > 50) this.ms.shift();
            this._drawRaster(m.spikes);
            cb?.(m.z);
        }
    }

    median(xs) { const sorted = [...xs].sort((a, b) => a - b); return sorted.length ? sorted[sorted.length >> 1] : 0; }

    // Dopamine-like reward: + for landing a hit, - for taking one, scaled by damage (a full kick = 1).
    reward(r) {
        if (!LEARN) return;
        if (THRESH_LR && this.lastPool) {  // the same signal moves the bar for acting on that pool
            const t = this.thresh[this.lastPool] - THRESH_LR * r * Z_TH;
            this.thresh[this.lastPool] = Math.min(THRESH_RANGE[1], Math.max(THRESH_RANGE[0], t));
        }
        this.rewards[r > 0 ? 'pos' : 'neg'] += Math.abs(r);
        this.worker.postMessage({ type: 'reward', r });
    }

    // Efference copy of the move that actually started, so learning credits that pool alone.
    executed(action) {
        if (!LEARN) return;
        const pool = Object.keys(POOL_ACTION).find(k => POOL_ACTION[k] === action);
        if (pool) {
            this.lastPool = pool;
            this.worker.postMessage({ type: 'executed', pool });
        }
    }

    resetLearning() {
        this.rewards = { pos: 0, neg: 0 };
        this.strength = null;
        this.thresh = Object.fromEntries(MOTORS.map(k => [k, Z_TH]));
        this.worker.postMessage({ type: 'resetLearning' });
    }

    _call(type, payload = {}, transfer = []) {
        const id = ++this.callId;
        return new Promise(res => {
            this.calls.set(id, res);
            this.worker.postMessage({ type, id, ...payload }, transfer);
        });
    }

    // { ratios: Float32Array (learned / wired, per plastic synapse), strength, fingerprint }
    getWeights() { return this._call('getWeights'); }

    async setWeights(ratios) {
        this.rewards = { pos: 0, neg: 0 };
        const copy = new Float32Array(ratios);
        return (await this._call('setWeights', { ratios: copy.buffer }, [copy.buffer])).ok;
    }

    rest(windows) {
        return new Promise(res => { this._rested = res; this.worker.postMessage({ type: 'rest', windows }); });
    }

    // Transient events are tracked every tick, even when the brain is busy.
    observe(v, now) {
        if (v.oppAttacking && !this.prevOppAttacking && v.inRange) this.loomUntil = now + 200;
        this.prevOppAttacking = v.oppAttacking;
    }

    // Game → sensory Poisson rates (Hz). The opponent is straight ahead, so both eyes get the same drive.
    //   loom   LPLC2+LC4: angular expansion of the opponent's image (dθ/dt, θ = 2·atan(H / 2d)), plus
    //          200 ms at the onset of a strike from within range. Slow walk-ins stay under threshold;
    //          jump-ins and strikes saturate, which is what drives the giant-fiber escape.
    //   object LC9: visible motion — the opponent walking, jumping or striking, or the gap changing
    //   target LC10a: apparent size of the opponent;  p1: male-specific arousal with proximity
    //   recede LPLC4: the opponent's image contracting, i.e. a limb being withdrawn. In this subgraph
    //          that reaches the wing pool (a guard), not the legs - no visual cue has a strong path to
    //          the legs, which is why the fly cannot learn to strike into a recovery.
    //   taste  front-leg taste bristles: forelegs in contact with the opponent
    //   body_touch thorax + mid-leg tactile bristles: being hit (250 ms)
    encode(v, now) {
        const close = clamp01(1 - v.dist / 600);
        const theta = d => 2 * Math.atan(OPP_HEIGHT_PX / (2 * Math.max(d, 1)));
        const expansion = Math.max(0, theta(v.dist) - theta(v.dist - v.dDist));  // radians per 100 ms
        const loom = 200 * clamp01((expansion - LOOM_THRESHOLD) / LOOM_RANGE + (now < this.loomUntil ? 1 : 0));
        const shrink = Math.max(0, (this.prevImage ?? v.oppImage) - v.oppImage);
        this.prevImage = v.oppImage;
        const recede = 200 * clamp01(shrink / 12);
        const object = 150 * clamp01(Math.abs(v.dDist) / 10 + (MOVING.has(v.oppMove) ? 0.8 : 0));
        const target = 120 * close;
        const p1 = 80 * close;
        const taste = v.dist < 0.75 * v.oppWidth ? 120 : 0;
        const body_touch = now < this.hitUntil ? 200 : 0;
        return {
            drive: { loom_L: loom, loom_R: loom, object_L: object, object_R: object, target_L: target, target_R: target, recede, p1, taste, body_touch },
            shown: { loom, object, target, recede, p1, taste, body_touch },
        };
    }

    tick(v, now, onApply) {
        this.observe(v, now);
        if (this.inflight || !this.gain) { this.skipped++; return; }
        const { drive, shown } = this.encode(v, now);
        const senseAt = performance.now();
        this.inflight = true;
        this.pending = z => {
            let best = null;
            const score = k => z[k] / this.gain[k];
            for (const k of MOTORS) if (z[k] > this.thresh[k] && (best === null || score(k) > score(best))) best = k;
            const action = best ? POOL_ACTION[best] : 'stand';
            this.decisions++;
            const ranked = MOTORS.filter(k => z[k] > this.thresh[k]).map(score).sort((a, b) => b - a);
            this.confidence = ranked.length > 1 ? 1 - ranked[1] / ranked[0] : ranked.length ? 1 : 0;
            const seq = ++this.seq;
            const commit = () => {
                if (seq < this.applied) { this.stale++; return; }  // a newer decision already acted
                this.applied = seq;
                this.reactions.push(performance.now() - senseAt);
                if (this.reactions.length > 60) this.reactions.shift();
                for (const k of Object.keys(SENSOR_LABEL)) setRow(this.sRows[k], shown[k] / 200, shown[k].toFixed(0));
                for (const k of MOTORS) setRow(this.mRows[k], z[k] / this.gain[k], z[k].toFixed(1), k === best);
                $('flyAction').textContent = pretty(action);
                const react = Math.round(this.median(this.reactions));
                const compute = (this.ms.reduce((a, b) => a + b, 0) / this.ms.length).toFixed(0);
                $('flyReact').textContent = `${react} ms`;
                $('flyReact').parentElement.title = `Sense to move: ${react} ms (${compute} ms of it simulating spikes)`;
                $('flyConf').textContent = this.confidence.toFixed(2);
                $('flyStale').textContent = this.stale;
                onApply(action, z, shown);
            };
            const wait = Math.max(0, this.lagOf() - (performance.now() - senseAt));
            if (wait > 0) setTimeout(commit, wait); else commit();
        };
        this.worker.postMessage({ type: 'step', drive });
    }

    _initRaster(motor) {
        this.rasterMotor = motor;
        const c = $('raster');
        c.height = motor.length;
        this.rctx = c.getContext('2d', { willReadFrequently: true });
    }

    _drawRaster(spikes) {
        const ctx = this.rctx, w = ctx.canvas.width, h = ctx.canvas.height;
        ctx.drawImage(ctx.canvas, -2, 0);
        ctx.fillStyle = '#000';
        ctx.fillRect(w - 2, 0, 2, h);
        const img = ctx.getImageData(w - 2, 0, 2, h);
        spikes.forEach((c, row) => {
            if (!c) return;
            const a = Math.min(255, 90 + c * 40);
            const [r, g, b] = this.rasterMotor[row] ? [69, 212, 131] : [242, 193, 78];
            for (let x = 0; x < 2; x++) img.data.set([r, g, b, a], (row * 2 + x) * 4);
        });
        ctx.putImageData(img, w - 2, 0);
    }
}

// ---------- Jev agent ----------
class JevAgent {
    constructor() {
        this.rows = makeRows($('jevProbs'), ACTIONS, Object.fromEntries(ACTIONS.map(a => [a, pretty(a)])));
        this.seq = 0;
        this.applied = 0;
        this.inflight = 0;
        this.lat = [];
        this.decisions = 0;
        this.stale = 0;
        this.errors = 0;
        this.calls = 0;
        this.tokens = 0;
        this.predErr = [];   // |distance we sent - distance when the answer landed|, px
        this.log = [];       // with ?log=jev: training data for a local policy
        this.policy = null;  // hybrid/local: the distilled policy (see policy.js)
        this.local = { moves: 0, sure: 0, corrections: 0, agree: [] };
        this.leadOf = () => (LEAD === 'off' ? 0 : LEAD === 'auto' || LEAD_DIST_ONLY ? (this.lat.length ? this.median() : LAG_FALLBACK) : Number(LEAD) || 0);
    }

    // The control: what a person would write by hand, from the same state on the same 100 ms clock.
    // If this matches a policy distilled from Jev, the game rewards simple positional rules.
    static rules(state) {
        const opp = state.opponent || {}, me = state.you || {};
        if (state.in_attack_range && opp.attacking) return 'block';
        if (state.in_attack_range) return (opp.action || '').includes('block') ? 'punch' : 'kick';
        if (state.distance_px > 80) return 'walk_forward';
        return me.busy ? 'stand' : 'walk_forward';
    }

    rulesTick(me, opp, v, timeLeft, onApply) {
        const state = this.state(me, opp, v, timeLeft);
        const action = JevAgent.rules(state);
        this.decisions++;
        const out = { action, probabilities: Object.fromEntries(ACTIONS.map(a => [a, a === action ? 1 : 0])), confidence: 1 };
        this._showLocal(out, state);
        onApply(action, out, 'rules');
    }

    // Local policy decides now; Jev corrects and teaches when its answer lands.
    hybridTick(me, opp, v, timeLeft, onApply) {
        const state = this.state(me, opp, v, timeLeft);
        const x = features(state);
        const out = this.policy.predict(x);
        this.local.moves++;
        if (out.confidence >= LOCAL_SURE) this.local.sure++;
        this._showLocal(out, state);
        onApply(out.action, out, 'local');
        if (JEV_MODE === 'local') return;
        this.ask(state, x, res => {
            // Jev's verdict: teach the policy, then correct the move if it still disagrees
            if (TEACH) this.policy.learn(x, res.probabilities);
            this.local.agree.push(out.action === res.action ? 1 : 0);
            if (this.local.agree.length > 80) this.local.agree.shift();
            const nowOut = this.policy.predict(features(this.state(me, opp, v, timeLeft)));
            if (nowOut.action !== res.action) {
                this.local.corrections++;
                onApply(res.action, res, 'jev');
            }
        });
    }

    agreement() { return this.local.agree.length ? this.local.agree.reduce((a, b) => a + b, 0) / this.local.agree.length : null; }

    _showLocal(out, state) {
        for (const a of ACTIONS) setRow(this.rows[a], out.probabilities[a] || 0, (out.probabilities[a] || 0).toFixed(2), a === out.action);
        $('jevAction').textContent = pretty(out.action);
        $('jevState').textContent = JSON.stringify(state, null, 2);
        if (JEV_MODE === 'rules') {
            $('jevHybrid').textContent = `Hand-written rules · ${this.decisions} moves`;
            return;
        }
        const agree = this.agreement();
        const sure = Math.round((this.local.sure / Math.max(1, this.local.moves)) * 100);
        $('jevHybrid').textContent = `${this.local.moves} local moves (${sure}% confident)`
            + ` · ${this.local.corrections} corrected by Jev`
            + (agree === null ? '' : ` · they agree ${Math.round(agree * 100)}%`)
            + (TEACH && JEV_MODE === 'hybrid' ? ` · ${this.policy.trained} taught` : '');
    }

    // How much longer a fighter is committed to its current move (mk.js ignores new moves until then).
    // Attacks animate out and back; finite moves (block, jump) just run out.
    commitMs(f) {
        const m = f.getMove();
        if (!m || !f._locked) return 0;
        const step = m._stepDuration || 40, total = m._totalSteps || 0, at = m._currentStep || 0;
        const left = m._moveBack ? at : m._dontReturn ? total - at : 2 * total - at;
        return Math.max(0, Math.round(left * step));
    }

    // The fight as Jev will find it when its answer lands, `lead` ms from now.
    state(me, opp, v, timeLeft) {
        const lead = this.leadOf();
        const closing = -v.dDist * 10;                       // px/s, positive = closing
        const dist = Math.max(0, v.dist + (v.dDist * lead) / TICK_MS);
        const myCommit = this.commitMs(me), oppCommit = this.commitMs(opp);
        const oppStillIn = oppCommit > lead;
        const keepAction = LEAD_DIST_ONLY;  // project the geometry only; report the opponent as seen
        return {
            ...(lead ? { lead_ms: Math.round(lead) } : {}),
            you: {
                life: round1(me.getLife()), action: me.getMove().type,
                busy: lead && !keepAction ? myCommit > lead : Boolean(me._locked),
                ...(lead && !keepAction ? { free_in_ms: Math.max(0, myCommit - Math.round(lead)) } : {}),
            },
            opponent: {
                life: round1(opp.getLife()),
                action: lead && !oppStillIn && !keepAction ? 'unknown' : v.oppMove,
                attacking: v.oppAttacking && (!lead || oppStillIn || keepAction),
                approaching: closing > 20, retreating: closing < -20,
                ...(lead && !keepAction ? { still_committed_when_you_act: oppStillIn } : {}),
            },
            distance_px: Math.round(lead ? dist : v.dist),
            closing_px_per_s: Math.round(closing),
            in_attack_range: (lead ? Math.abs(dist) : v.dist) <= v.oppWidth,
            time_left_s: Math.round(timeLeft - lead / 1000),
        };
    }

    // Fire-and-apply: requests leave on the shared clock; an answer is applied only
    // if nothing newer has been applied, and only against the current fight state.
    tick(me, opp, v, timeLeft, onApply) {
        if (JEV_MODE === 'rules') return this.rulesTick(me, opp, v, timeLeft, onApply);
        if (this.policy) return this.hybridTick(me, opp, v, timeLeft, onApply);
        if (this.inflight >= JEV_MAX_INFLIGHT) return;
        const state = this.state(me, opp, v, timeLeft);
        this.ask(state, null, res => onApply(res.action, res, 'jev'));
    }

    ask(state, x, onAnswer) {
        if (this.inflight >= JEV_MAX_INFLIGHT) return;
        const seq = ++this.seq;
        const t0 = performance.now();
        this.inflight++;
        this.calls++;
        fetch('/api/jev/move', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(state),
        }).then(r => r.json()).then(res => {
            if (res.error) throw new Error(res.error);
            this.lat.push(performance.now() - t0);
            if (this.lat.length > 60) this.lat.shift();
            this.tokens += (res.usage?.input_tokens || 0) + (res.usage?.output_tokens || 0);
            if (seq < this.applied) { this.stale++; return; }
            this.applied = seq;
            this.decisions++;
            if (LOG_JEV && this.log.length < 20000) {
                this.log.push({ x: Array.from(x || features(state), v => +v.toFixed(4)), p: res.probabilities,
                    a: res.action, c: res.confidence, lat: Math.round(performance.now() - t0) });
            }
            if (!this.policy) this._show(res, state);
            else { $('jevConf').textContent = (res.confidence ?? 0).toFixed(2); $('jevLat').textContent = `${Math.round(this.median())} ms`; }
            onAnswer(res);
        }).catch(err => {
            this.errors++;
            $('jevErr').textContent = this.errors;
            console.warn('jev', err);
        }).finally(() => { this.inflight--; });
    }

    median(xs = this.lat) { const sorted = [...xs].sort((a, b) => a - b); return sorted.length ? sorted[sorted.length >> 1] : 0; }

    _show(res, state) {
        $('jevStale').textContent = this.stale;
        $('jevErr').textContent = this.errors;
        $('jevState').textContent = JSON.stringify(state, null, 2);
        let best = res.action;
        for (const a of ACTIONS) setRow(this.rows[a], res.probabilities[a] || 0, (res.probabilities[a] || 0).toFixed(2), a === best);
        $('jevAction').textContent = pretty(res.action);
        $('jevConf').textContent = res.confidence != null ? res.confidence.toFixed(2) : '–';
        $('jevLat').textContent = `${Math.round(this.median())} ms`;
    }
}

// ---------- arcade screen ----------
function Headless(options) { mk.controllers.Base.call(this, options); }
Headless.prototype = new mk.controllers.Base();
Headless.prototype._initialize = function () {};

const results = [];
window.__mk = { results, done: false, audio };
const score = { fly: 0, jev: 0, draw: 0 };
const totals = { fly: { hits: 0, dmg: 0, blocked: 0 }, jev: { hits: 0, dmg: 0, blocked: 0 } };
const NAME = { fly: 'Fly brain', jev: 'Jev' };
const sleep = ms => new Promise(r => setTimeout(r, ms));

function announce(text, sub = '', { small = false, blink = false } = {}) {
    const b = $('banner'), s = $('bannerSub');
    b.textContent = text;
    s.textContent = sub;
    b.className = small ? 'small' : '';
    s.className = '';
    if (!text) return;
    void b.offsetWidth;  // restart the slam animation
    b.classList.add(blink ? 'blink' : 'slam');
    if (sub) { void s.offsetWidth; s.classList.add('slam'); }
}

function renderMedals(flyIdx) {
    const draw = (el, n) => { el.innerHTML = '<span class="medal"></span>'.repeat(n); };
    draw($(flyIdx === 0 ? 'leftWins' : 'rightWins'), score.fly);
    draw($(flyIdx === 0 ? 'rightWins' : 'leftWins'), score.jev);
}

// One chip per round: who won, how, and with how much life left. Unplayed rounds show dimmed.
function renderHistory() {
    const el = $('history');
    el.innerHTML = '';
    for (let i = 0; i < ROUNDS; i++) {
        const r = results[i];
        const li = document.createElement('li');
        if (r) {
            li.className = r.winner;
            const how = r.reason === 'KO' ? 'KO' : 'Time';
            const left = r.winner === 'draw' ? '' : ` ${Math.round(r.life[r.winner])}`;
            li.innerHTML = `<b>${r.winner === 'draw' ? 'Draw' : NAME[r.winner]}</b><span>R${r.round} ${how}${left}</span>`;
            li.title = `Round ${r.round}: ${r.winner === 'draw' ? 'draw' : NAME[r.winner] + ' won'} (${how}), fly ${Math.round(r.life.fly)} vs Jev ${Math.round(r.life.jev)} life`;
        } else {
            li.className = 'next';
            li.innerHTML = `<b>R${i + 1}</b><span>${i === results.length ? 'Next' : '&nbsp;'}</span>`;
        }
        el.appendChild(li);
    }
}

function renderTotals() {
    for (const who of ['fly', 'jev']) {
        const id = who === 'fly' ? 'fly' : 'jev';
        $(`${id}Hits`).textContent = totals[who].hits;
        $(`${id}Dmg`).textContent = Math.round(totals[who].dmg);
        $(`${id}Blocked`).textContent = totals[who].blocked;  // this fighter's hits the opponent guarded
    }
}

function runRound(round, fly, jev) {
    return new Promise(resolve => {
        const flyIdx = 0;  // fly brain is always Sub-Zero on the left, Jev always Kano on the right
        const names = ['Subzero', 'Kano'];
        const who = i => (i === flyIdx ? 'fly' : 'jev');
        $('leftName').textContent = NAME[who(0)];
        $('rightName').textContent = NAME[who(1)];
        $('leftChar').textContent = 'Sub-Zero';
        $('rightChar').textContent = 'Kano';
        const lifeBars = [$('leftLife'), $('rightLife')];
        lifeBars.forEach(b => { b.style.width = '100%'; });
        $('timer').textContent = ROUND_S;
        $('timer').classList.remove('low');
        renderMedals(flyIdx);

        let fighters, ended = false, timer = null, start = 0;
        fly.decisions = 0;
        fly.skipped = 0;
        fly.stale = 0;
        fly.reactions = [];
        Object.assign(jev, { decisions: 0, calls: 0, errors: 0, stale: 0, lat: [], predErr: [] });
        const hits = { fly: 0, jev: 0 }, dmg = { fly: 0, jev: 0 };
        const trace = [];
        // combat bookkeeping: attacks started (and whether the target was in range), hits and blocks
        const combat = { fly: { attacks: 0, inRange: 0, hits: 0, blocked: 0 }, jev: { attacks: 0, inRange: 0, hits: 0, blocked: 0 } };
        // what became of each decision: a new move, the same move held, or ignored (mid-attack / mid-jump)
        const moves = { fly: { started: 0, holding: 0, ignored: 0 }, jev: { started: 0, holding: 0, ignored: 0 } };
        const moveSource = {};  // hybrid: which moves came from the local policy vs from Jev
        const flyLog = [];      // with ?log=fly: the fly's own state -> move, for distilling it
        const NOMINAL = { [T.HIGH_PUNCH]: 8, [T.HIGH_KICK]: 10 };
        const noteAttack = (who_, action, started, v, f) => {
            moves[who_][started === 'started' ? 'started' : started === 'holding' ? 'holding' : 'ignored']++;
            if (started === 'started' && action === 'jump_away') audio.jump(panOf(f));
            if (started !== 'started' || (action !== 'punch' && action !== 'kick')) return;
            audio.swing(action, panOf(f));
            combat[who_].attacks++;
            if (v.inRange) combat[who_].inRange++;
        };
        const actionCounts = { fly: {}, jev: {} };

        const finish = (winner, reason) => {
            if (ended) return;
            ended = true;
            clearInterval(timer);
            const life = { fly: fighters[flyIdx].getLife(), jev: fighters[1 - flyIdx].getLife() };
            const res = {
                round, winner, reason, flySide: flyIdx === 0 ? 'left' : 'right',
                seconds: +((performance.now() - start) / 1000).toFixed(1), life, hits, damage: dmg, actions: actionCounts,
                jev: { decisions: jev.decisions, calls: jev.calls, errors: jev.errors, stale: jev.stale, lead_ms: Math.round(jev.leadOf()),
                    pred_err_px: Math.round(jev.median(jev.predErr)), mode: JEV_MODE, move_source: moveSource,
                    ...(jev.policy ? { local: { ...jev.local, agree: undefined, agreement: jev.agreement(), taught: jev.policy.trained } } : {}),
                    median_latency_ms: Math.round([...jev.lat].sort((a, b) => a - b)[jev.lat.length >> 1] || 0) },
                combat, moves,
                ...(LOG_FLY ? { fly_log: flyLog } : {}),
                ...(TRACE ? { trace } : {}),
                fly: { decisions: fly.decisions, skipped: fly.skipped, stale: fly.stale,
                    reaction_ms: Math.round(fly.median(fly.reactions)), lag_ms: Math.round(fly.lagOf()),
                    ...(THRESH_LR ? { thresholds: { ...fly.thresh } } : {}), learning: LEARN ? { strength: fly.strength, rewards: { ...fly.rewards } } : null,
                    compute_ms: Math.round(fly.ms.reduce((a, b) => a + b, 0) / Math.max(1, fly.ms.length)) },
            };
            const flawless = winner !== 'draw' && life[winner] >= 100;
            if (winner === 'draw') announce('Draw', 'Time');
            else announce(`${NAME[winner]} wins`, flawless ? 'Flawless victory' : reason === 'time' ? 'Time' : '');
            audio.roundEnd();
            if (reason === 'KO') audio.ko(); else audio.bell();
            setTimeout(() => {
                audio.say(...(reason === 'time' ? ['time'] : []), winner === 'draw' ? 'draw' : `${winner}_wins`, ...(flawless ? ['flawless'] : []));
                audio.music('menu');
            }, 700);
            setTimeout(() => {
                try { mk.reset(); } catch (e) { console.warn(e); }
                resolve(res);
            }, 2600);
        };

        const options = {
            arena: { container: $('arena'), arena: mk.arenas.types.THRONE_ROOM },
            fighters: names.map(name => ({ name })),
            callbacks: {
                attack(f, o, lost) {
                    const i = fighters.indexOf(o);
                    const attacker = who(1 - i);
                    hits[attacker]++;
                    combat[attacker].hits++;
                    const guarded = NOMINAL[f.getMove().type] && lost < NOMINAL[f.getMove().type] * 0.5;
                    if (guarded) {
                        combat[attacker].blocked++;
                        totals[attacker].blocked++;
                        audio.block(panOf(o));
                    } else {
                        audio.hit(/kick/.test(f.getMove().type) ? 'kick' : 'punch', panOf(o), lost / 10);
                        audio.grunt(who(i), panOf(o), lost / 10);
                    }
                    dmg[attacker] += lost;
                    totals[attacker].hits++;
                    totals[attacker].dmg += lost;
                    renderTotals();
                    lifeBars[i].style.width = `${o.getLife()}%`;
                    if (who(i) === 'fly') fly.hitUntil = performance.now() + 200;
                    // Dopamine as reward prediction error: an incoming strike is expected to cost its full
                    // damage, so blocking it is better than expected (+) and eating it is worse (-).
                    // Landing a hit is + its damage. Scale: a full 10-damage kick = 1.
                    const full = NOMINAL[f.getMove().type] ?? lost;
                    fly.reward(attacker === 'fly' ? lost / 10 : (lost < full * 0.5 ? (full - lost) : -lost) / 10);
                },
                'game-end'(dead) {
                    finish(who(1 - fighters.indexOf(dead)), 'KO');
                },
            },
        };

        mk.game = new Headless(options);
        const promise = new mk.Promise();
        if (round === 1) announce('Loading fighters', '', { small: true, blink: true });
        promise.ready(async () => {
            fighters = mk.game.fighters;
            $('screen').classList.remove('idle');
            announce(`Round ${round}`);
            audio.bell();
            audio.say(`round${round}`);
            await Promise.all([sleep(1200), audio.voiceIdle(3000)]);
            announce('Fight!');
            audio.fight();
            audio.say('fight');
            audio.music('fight');
            await sleep(650);
            announce('');

            const flyF = fighters[flyIdx], jevF = fighters[1 - flyIdx];
            let prevFly = null, prevJev = null, ticks = 0, lastSecond = ROUND_S;
            start = performance.now();
            timer = setInterval(() => {
                if (ended) return;
                const now = performance.now();
                const elapsed = (now - start) / 1000, left = ROUND_S - elapsed;
                const second = Math.max(0, Math.ceil(left));
                $('timer').textContent = second;
                $('timer').classList.toggle('low', left <= 10);
                if (left <= 10) audio.music('tense');
                if (second !== lastSecond) {
                    if (second <= 5 && second > 0) audio.tick();
                    lastSecond = second;
                }
                if (left <= 0) {
                    const lf = flyF.getLife(), lj = jevF.getLife();
                    finish(lf === lj ? 'draw' : lf > lj ? 'fly' : 'jev', 'time');
                    return;
                }
                const vf = view(flyF, jevF, prevFly), vj = view(jevF, flyF, prevJev);
                prevFly = vf; prevJev = vj;

                const flyState = LOG_FLY ? jev.state(flyF, jevF, vf, left) : null;
                if (ticks % FLY_EVERY) fly.observe(vf, now);  // handicapped: still watch, just don't decide
                else fly.tick(vf, now, (action, z, drive) => {
                    if (ended) return;
                    actionCounts.fly[action] = (actionCounts.fly[action] || 0) + 1;
                    const va = view(flyF, jevF, null);
                    if (TRACE) trace.push({ t: +((performance.now() - start) / 1000).toFixed(2), flyX: flyF.getX(), jevX: jevF.getX(),
                        flyMove: flyF.getMove().type, jevMove: jevF.getMove().type, oppRight: va.oppRight, action,
                        drive: Object.fromEntries(Object.entries(drive).map(([k, x]) => [k, Math.round(x)])),
                        z: Object.fromEntries(Object.entries(z).map(([k, x]) => [k, +x.toFixed(1)])) });
                    if (LOG_FLY && flyLog.length < 20000) flyLog.push({ x: Array.from(features(flyState), q => +q.toFixed(4)), a: action });
                    const started = applyAction(flyF, action, va);
                    if (started) fly.executed(action);
                    noteAttack('fly', action, started, va, flyF);
                });

                jev.tick(jevF, flyF, vj, left, (action, res, source) => {
                    if (ended) return;
                    actionCounts.jev[action] = (actionCounts.jev[action] || 0) + 1;
                    moveSource[source] = (moveSource[source] || 0) + 1;
                    // re-derive direction at apply time: the fighters may have crossed while Jev was thinking
                    const vj2 = view(jevF, flyF, null);
                    noteAttack('jev', action, applyAction(jevF, action, vj2), vj2, jevF);
                });

                ticks++;
                if (ticks % 10 === 0) {
                    $('flyRate').textContent = (fly.decisions / elapsed).toFixed(1);
                    $('jevRate').textContent = (jev.decisions / elapsed).toFixed(1);
                }
            }, TICK_MS);
        });
        mk.game.init(promise);
    });
}

// ---------- saved brains + telemetry ----------
// Every match is saved to telemetry/mk/ (rewritten after each round). With learning on it includes the
// fly's learned synapse strengths, and the best-scoring ones can seed a later match.
let brainList = [];

async function loadBrainList() {
    try {
        const r = await fetch(`/api/mk/brains?fingerprint=${encodeURIComponent(fly.meta.fingerprint)}`);
        brainList = r.ok ? (await r.json()).brains : [];
    } catch {
        brainList = [];
    }
    renderBrainSelect();
}

function renderBrainSelect() {
    const sel = $('brainSelect');
    // mid-match (menu locked) keep what's running; otherwise the viewer's pick, the URL, then the last pick
    let want = sel.disabled ? sel.value : sel.dataset.choice || params.get('brain');
    if (!want) {
        try { want = localStorage.getItem('mk.brain'); } catch { /* storage unavailable */ }
    }
    sel.innerHTML = '';
    const add = (value, label) => {
        const o = document.createElement('option');
        o.value = value;
        o.textContent = label;
        sel.appendChild(o);
    };
    add('wired', 'Wired: the connectome as mapped');
    brainList.forEach((b, i) => {
        const s = b.score || {};
        const when = b.started_at ? new Date(b.started_at).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) : '';
        add(b.id, `${i === 0 ? 'Best learned' : 'Learned'}: ${s.fly}–${s.jev}${s.draw ? ` (${s.draw} drawn)` : ''} vs Jev, ${when}`);
    });
    if (want === 'best') want = brainList[0]?.id;
    sel.value = [...sel.options].some(o => o.value === want) ? want : 'wired';
}

async function startBrain(choice) {
    if (choice === 'wired') {
        fly.resetLearning();
        return 'wired';
    }
    try {
        const r = await fetch(`/api/mk/brain/${encodeURIComponent(choice)}`);
        const b = r.ok ? await r.json() : null;
        if (b && b.weights_b64 && b.fingerprint === fly.meta.fingerprint && await fly.setWeights(fromB64(b.weights_b64))) {
            if (b.thresholds) fly.thresh = { ...fly.thresh, ...b.thresholds };
            return choice;
        }
    } catch (e) {
        console.warn('saved brain', e);
    }
    fly.resetLearning();
    $('brainSelect').value = 'wired';
    return 'wired';
}

async function saveTelemetry(record) {
    try {
        record.updated_at = new Date().toISOString();
        const w = await fly.getWeights();
        record.circuit.fingerprint = w.fingerprint;
        // weights only when the fly learned this match; a frozen replay would just duplicate its source
        record.learned = LEARN ? { strength: w.strength, encoding: 'float32le base64, learned/wired per plastic synapse',
            weights_b64: toB64(w.ratios), thresholds: THRESH_LR ? { ...fly.thresh } : null } : null;
        const r = await fetch('/api/mk/telemetry', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(record) });
        if (!r.ok) console.warn('telemetry not saved', r.status);
    } catch (e) {
        console.warn('telemetry not saved', e);
    }
}

async function runMatch(fly, jev) {
    $('startBtn').disabled = true;
    $('brainSelect').disabled = true;
    score.fly = score.jev = score.draw = 0;
    for (const t of Object.values(totals)) { t.hits = 0; t.dmg = 0; t.blocked = 0; }
    const brainStart = await startBrain($('brainSelect').value);
    jev.log = [];
    renderTotals();
    $('scoreFly').textContent = $('scoreJev').textContent = '0';
    results.length = 0;
    renderHistory();
    if (jev.policy) jev.local = { moves: 0, sure: 0, corrections: 0, agree: [] };
    const record = {
        id: `${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}-${Math.random().toString(36).slice(2, 6)}`,
        version: 1,
        started_at: new Date().toISOString(),
        complete: false,
        config: {
            rounds: ROUNDS, round_seconds: ROUND_S, seed: SEED, learn: LEARN, lr: Number(params.get('lr')) || null,
            announcer: params.get('announcer') || 'daniel', brain_start: brainStart, all_rounds: Boolean(params.get('all')),
            jev_mode: JEV_MODE, teach: TEACH, premotor: PREMOTOR, baseline: Number(params.get('baseline')) || 0,
            thresh_lr: THRESH_LR, circuit: CIRCUIT,
            ...(EVOLVE ? { generation: window.__mk.generation, lineage: LINEAGE } : {}),
            fly_side: 'left', fly_every: FLY_EVERY, lag: LAG, lead: LEAD,
        },
        circuit: { ...fly.meta },
        ...(LOG_JEV ? { jev_log: jev.log } : {}),
        score: { ...score },
        rounds: results,
    };
    const need = Math.floor(ROUNDS / 2) + 1;
    for (let round = 1; round <= ROUNDS; round++) {
        $('roundInfo').textContent = `Round ${round}, best of ${ROUNDS}`;
        $('status').textContent = '';
        await Promise.all([fly.rest(10), audio.voiceIdle()]);
        const res = await runRound(round, fly, jev);
        results.push(res);
        score[res.winner]++;
        renderHistory();
        $('scoreFly').textContent = score.fly;
        $('scoreJev').textContent = score.jev;
        console.log('round', JSON.stringify(res));
        record.score = { ...score };
        await saveTelemetry(record);
        if (!params.get('all') && (score.fly >= need || score.jev >= need)) break;
    }
    await audio.voiceIdle();
    $('screen').classList.add('idle');
    renderMedals(0);
    $('leftName').textContent = NAME.fly;
    $('rightName').textContent = NAME.jev;
    $('leftLife').style.width = $('rightLife').style.width = '100%';
    $('timer').textContent = '';
    const tally = `${score.fly}–${score.jev}${score.draw ? `, ${score.draw} drawn` : ''}`;
    const champ = score.fly === score.jev ? null : score.fly > score.jev ? 'fly' : 'jev';
    if (!champ) announce('Draw', `Match ${tally}`);
    else announce(`${NAME[champ]} wins`, `Match ${tally}`);
    audio.victory();
    audio.say(champ ? `${champ}_match` : 'match_draw');
    $('roundInfo').textContent = `Final, best of ${ROUNDS}`;
    $('status').textContent = 'Press start or Enter for a rematch';
    record.complete = true;
    record.score = { ...score };
    if (jev.policy) record.local_policy = { weights: jev.policy.export(), taught: jev.policy.trained, distilled_from: jev.policy.distilledFrom || 0 };
    await saveTelemetry(record);
    window.__mk.matchId = record.id;
    await loadBrainList();
    window.__mk.done = true;
    window.__mk.score = { ...score };
    $('startBtn').disabled = false;
    $('brainSelect').disabled = false;
}

// ---------- sound ----------
function renderSound() {
    const b = $('soundBtn');
    b.textContent = audio.muted ? 'Sound off' : 'Sound on';
    b.setAttribute('aria-pressed', String(!audio.muted));
}
$('soundBtn').addEventListener('click', e => {
    e.stopPropagation();
    audio.unlock();
    audio.setMuted(!audio.muted);
    renderSound();
});
// browsers only allow audio after an interaction; any click or key press turns it on
for (const type of ['pointerdown', 'keydown']) document.addEventListener(type, () => audio.unlock());
document.addEventListener('keydown', e => {
    if (e.key.toLowerCase() === 'm' && e.target === document.body) {
        audio.setMuted(!audio.muted);
        renderSound();
    }
});
renderSound();

// ---------- boot ----------
$('roundInfo').textContent = `Best of ${ROUNDS}`;
$('status').textContent = 'Warming up the fly brain';
renderHistory();
// ?evolve=N: each match inherits the best brain so far (or the wiring, with ?lineage=0)
async function runSeries(fly, jev) {
    for (let gen = 1; gen <= EVOLVE; gen++) {
        window.__mk.generation = gen;
        $('roundInfo').textContent = `Generation ${gen} of ${EVOLVE}`;
        if (LINEAGE) {
            await loadBrainList();
            $('brainSelect').value = brainList[0]?.id || 'wired';
        } else {
            $('brainSelect').value = 'wired';
        }
        await runMatch(fly, jev);
        console.log('generation', gen, 'of', EVOLVE, 'score', JSON.stringify(window.__mk.score));
    }
    window.__mk.seriesDone = true;
}

const fly = new FlyAgent();
const jev = new JevAgent();
if (JEV_MODE === 'rules') {
    $('jevSub').textContent = 'Hand-written rules on the same state, at the same 100 ms clock';
    $('jevHybrid').hidden = false;
} else if (JEV_MODE !== 'api') {
    LocalPolicy.load().then(policy => {
        jev.policy = policy;
        $('jevSub').textContent = JEV_MODE === 'local'
            ? `Local policy only: ${policy.w.length} weights distilled from ${policy.distilledFrom || 0} Jev decisions`
            : `Hybrid: local policy moves on the 100 ms clock, Jev corrects and teaches it`;
        $('jevHybrid').hidden = false;
    });
}
if (LAG === 'auto') fly.lagOf = () => (jev.lat.length ? jev.median() : LAG_FALLBACK);
fly.ready.then(async meta => {
    $('flySub').textContent = (CIRCUIT === 'rewired' ? 'REWIRED CONTROL — ' : '') + `maleCNS v1.0: ${meta.neurons.toLocaleString()} neurons, ${(meta.synapses / 1e6).toFixed(1)}M synapses, ` +
        (LEARN ? `learns from hits during the match` : 'reflexes only, no learning');
    if (!LEARN) $('flyLearnHead').hidden = true;
    $('brainSelect').addEventListener('change', e => {
        e.target.dataset.choice = e.target.value;
        try { localStorage.setItem('mk.brain', e.target.value); } catch { /* storage unavailable */ }
    });
    await loadBrainList();  // before any auto-start, so ?brain=best has something to pick
    const start = () => {
        if ($('startBtn').disabled) return;
        audio.unlock();
        audio.coin();
        audio.say('title');
        runMatch(fly, jev);
    };
    $('startBtn').disabled = false;
    $('startBtn').onclick = start;
    // arcade shortcuts: click the screen or press Enter
    $('screen').addEventListener('click', start);
    document.addEventListener('keydown', e => { if (e.key === 'Enter' && e.target === document.body) start(); });
    audio.music('menu');
    if (EVOLVE) runSeries(fly, jev);
    else if (params.get('auto')) start();
    else {
        $('status').textContent = `Best of ${ROUNDS}`;
        announce('Press start', '', { small: true, blink: true });
    }
});
