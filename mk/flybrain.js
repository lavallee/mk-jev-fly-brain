// maleCNS "fighter" brain: conductance-based LIF network running in the browser.
// Port of src/model.py (Shiu et al. 2024-style LIF). Wiring = maleCNS v1.0
// synapse counts (scripts/build_malecns_fighter.py); physiology = assumptions.

export const LIF = {
    dt: 0.25,           // ms
    C_m: 100, g_leak: 10, E_leak: -65, V_th: -50, V_reset: -65, refrac: 2,
    E_exc: 0, E_inh: -75, tau_exc: 3, tau_inh: 8,
    weight_scale: 0.05,  // nS per synapse
    background_rate: 10, background_weight: 10,
    stim_weight: 25,
};

export const SENSORS = ['loom_L', 'loom_R', 'target_L', 'target_R', 'object_L', 'object_R', 'recede', 'p1', 'taste', 'body_touch'];
export const MOTORS = ['fwd', 'back', 'jump', 'punch', 'kick', 'wing'];

// Dopamine-like three-factor plasticity on excitatory synapses onto the motor pools.
//   eligibility: a post spike tags each input synapse by exp(-(t_post - t_pre) / tau_pre); tags decay with tau_elig
//   action gate: an efference copy marks the pool whose move was actually executed (decays with tau_elig);
//                only synapses onto that pool can change, so co-active pools don't share the credit
//   reward r (reward prediction error) changes w by lr * r * (tag * gate / largest) * w0, clamped to [w_min, w_max] * w0
// Flies do reward learning mainly in the mushroom body; putting it on motor inputs is a modelling assumption.
export const PLASTICITY = { tau_pre: 20, tau_elig: 1000, lr: 0.2, w_min: 0.1, w_max: 3, baseline: 0 };
// baseline > 0 subtracts a running average of recent dopamine from each signal, so a fighter that is
// losing throughout still learns which of its moves went least badly instead of suppressing all of
// them. The value is the averaging rate (0.02 = the last ~50 rewards).

// mulberry32: seeded so a match can be replayed
function rng(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export class FlyBrain {
    constructor(circuit, { seed = 1, params = {}, readout = MOTORS } = {}) {
        this.p = { ...LIF, ...params };
        this.c = circuit;
        const n = circuit.neurons.length;
        this.n = n;
        this.rand = rng(seed);

        // CSR by presynaptic neuron (edges are pre-sorted by the builder)
        const m = circuit.pre.length;
        this.rowStart = new Int32Array(n + 1);
        for (let k = 0; k < m; k++) this.rowStart[circuit.pre[k] + 1]++;
        for (let i = 0; i < n; i++) this.rowStart[i + 1] += this.rowStart[i];
        this.post = Int32Array.from(circuit.post);
        this.w = new Float32Array(m);
        for (let k = 0; k < m; k++) this.w[k] = circuit.w[k] * this.p.weight_scale;
        this.w0 = Float32Array.from(this.w);
        this.preOf = new Int32Array(m);
        for (let i = 0; i < n; i++) for (let k = this.rowStart[i]; k < this.rowStart[i + 1]; k++) this.preOf[k] = i;
        this.plastic = null;
        this.sign = Int8Array.from(circuit.neurons.map(x => x[3]));

        this.V = new Float32Array(n).fill(this.p.E_leak);
        this.gE = new Float32Array(n);
        this.gI = new Float32Array(n);
        this.ref = new Float32Array(n);
        this.spikeCount = new Uint16Array(n);   // spikes in the current window
        this.lastSpike = new Float32Array(n).fill(-1e9);
        this.t = 0;

        this.groups = {};
        for (const [k, idx] of Object.entries(circuit.groups)) this.groups[k] = Int32Array.from(idx);
        this.drive = Object.fromEntries(SENSORS.map(s => [s, 0]));  // Hz per neuron, keyed by group
        this.readout = readout;

        // per-pool baseline (running mean/var of window rate) for the z-scored readout
        this.base = Object.fromEntries(readout.map(k => [k, { mean: 0, var: 1, n: 0 }]));
    }

    setDrive(d) { Object.assign(this.drive, d); }

    // Back to rest: membrane potentials, conductances and refractory clocks, not the (possibly learned) weights.
    resetState() {
        this.V.fill(this.p.E_leak);
        this.gE.fill(0);
        this.gI.fill(0);
        this.ref.fill(0);
        if (this.plastic) { this.plastic.elig.fill(0); this.plastic.gate.fill(0); }
    }

    // Make the excitatory synapses onto these groups plastic. With `premotor`, also those onto the
    // N neurons that drive the pools hardest, so learning can re-weight the paths feeding a pool and
    // not just the last step. Each premotor neuron is credited to the pool it drives most, which is
    // what the efference-copy gate needs.
    enablePlasticity(groupNames = MOTORS, { premotor = 0 } = {}) {
        const n = this.n, groupOf = new Int8Array(n).fill(-1);
        groupNames.forEach((g, gi) => { for (const i of this.groups[g] || []) groupOf[i] = gi; });
        if (premotor > 0) {
            const drive = new Float64Array(n * groupNames.length);
            for (let k = 0; k < this.post.length; k++) {
                const g = groupOf[this.post[k]];
                if (g >= 0 && this.sign[this.preOf[k]] > 0) drive[this.preOf[k] * groupNames.length + g] += this.w0[k];
            }
            const total = new Float64Array(n);
            for (let i = 0; i < n; i++) {
                if (groupOf[i] >= 0) continue;
                for (let g = 0; g < groupNames.length; g++) total[i] += drive[i * groupNames.length + g];
            }
            const ranked = [...total.keys()].filter(i => total[i] > 0).sort((a, b) => total[b] - total[a]).slice(0, premotor);
            for (const i of ranked) {
                let best = 0;
                for (let g = 1; g < groupNames.length; g++) {
                    if (drive[i * groupNames.length + g] > drive[i * groupNames.length + best]) best = g;
                }
                groupOf[i] = best;
            }
        }
        const incoming = Array.from({ length: n }, () => []);
        for (let k = 0; k < this.post.length; k++) {
            const j = this.post[k];
            if (groupOf[j] >= 0 && this.sign[this.preOf[k]] > 0) incoming[j].push(k);
        }
        const inStart = new Int32Array(n + 1);
        for (let j = 0; j < n; j++) inStart[j + 1] = inStart[j] + incoming[j].length;
        const k = new Int32Array(inStart[n]), grp = new Int8Array(inStart[n]);
        for (let j = 0; j < n; j++) incoming[j].forEach((kk, o) => { k[inStart[j] + o] = kk; grp[inStart[j] + o] = groupOf[j]; });
        this.plastic = { groups: groupNames, inStart, k, grp, elig: new Float32Array(k.length), gate: new Float32Array(groupNames.length) };
        return k.length;
    }

    // Efference copy: this pool's move was just executed.
    executed(group) {
        const pl = this.plastic;
        const g = pl ? pl.groups.indexOf(group) : -1;
        if (g >= 0) pl.gate[g] = 1;
    }

    // Deliver a dopamine-like reward. Returns each group's mean synaptic strength relative to the wiring.
    reward(r) {
        const pl = this.plastic;
        if (!pl) return null;
        if (PLASTICITY.baseline > 0) {
            this.rewardMean = (this.rewardMean ?? 0) + PLASTICITY.baseline * (r - (this.rewardMean ?? 0));
            r -= this.rewardMean;
        }
        let top = 0;
        for (let q = 0; q < pl.elig.length; q++) {
            const e = pl.elig[q] * pl.gate[pl.grp[q]];
            if (e > top) top = e;
        }
        if (top > 0 && r !== 0) {
            const { lr, w_min, w_max } = PLASTICITY;
            for (let q = 0; q < pl.elig.length; q++) {
                const e = pl.elig[q] * pl.gate[pl.grp[q]];
                if (e <= 0) continue;
                const kk = pl.k[q], w0 = this.w0[kk];
                this.w[kk] = Math.min(w_max * w0, Math.max(w_min * w0, this.w[kk] + lr * r * (e / top) * w0));
            }
        }
        return this.strength();
    }

    strength() {
        const pl = this.plastic;
        const sum = pl.groups.map(() => 0), sum0 = pl.groups.map(() => 0);
        for (let q = 0; q < pl.k.length; q++) { sum[pl.grp[q]] += this.w[pl.k[q]]; sum0[pl.grp[q]] += this.w0[pl.k[q]]; }
        return Object.fromEntries(pl.groups.map((g, i) => [g, sum0[i] ? sum[i] / sum0[i] : 1]));
    }

    resetPlasticity() {
        this.rewardMean = 0;
        if (!this.plastic) return;
        this.w.set(this.w0);
        this.plastic.elig.fill(0);
        this.plastic.gate.fill(0);
    }

    // Learned state: each plastic synapse's strength relative to the wiring, in plastic-synapse order.
    getRatios() {
        const pl = this.plastic, r = new Float32Array(pl.k.length);
        for (let q = 0; q < pl.k.length; q++) r[q] = this.w[pl.k[q]] / this.w0[pl.k[q]];
        return r;
    }

    setRatios(r) {
        const pl = this.plastic;
        if (!pl || r.length !== pl.k.length) return false;
        for (let q = 0; q < pl.k.length; q++) this.w[pl.k[q]] = this.w0[pl.k[q]] * r[q];
        pl.elig.fill(0);
        pl.gate.fill(0);
        return true;
    }

    // Identifies the plastic synapse layout (FNV-1a over each synapse's pre/post neuron), so saved
    // weights are only loaded into the circuit they were learned on.
    fingerprint() {
        const pl = this.plastic;
        let h = 0x811c9dc5;
        const mix = x => { h ^= x; h = Math.imul(h, 16777619) >>> 0; };
        for (let q = 0; q < pl.k.length; q++) { mix(this.preOf[pl.k[q]]); mix(this.post[pl.k[q]]); }
        return `${this.n}n-${pl.k.length}s-${h.toString(16).padStart(8, '0')}`;
    }

    // Advance `ms` of simulated time. Returns per-pool mean rate (Hz/neuron) over the window.
    step(ms) {
        const p = this.p, n = this.n, dt = p.dt;
        const dE = Math.exp(-dt / p.tau_exc), dI = Math.exp(-dt / p.tau_inh);
        const invC = dt / p.C_m;
        const steps = Math.round(ms / dt);
        const bgP = p.background_rate * dt / 1000;
        this.spikeCount.fill(0);
        const { V, gE, gI, ref, rowStart, post, w, sign, rand, lastSpike } = this;
        const pl = this.plastic;

        for (let s = 0; s < steps; s++) {
            // background Poisson: skip-sample the Bernoulli events (geometric gaps)
            for (let i = Math.floor(Math.log(1 - rand()) / Math.log(1 - bgP)); i < n;
                i += 1 + Math.floor(Math.log(1 - rand()) / Math.log(1 - bgP))) {
                gE[i] += p.background_weight;
            }
            // sensory Poisson drive
            for (const key in this.drive) {
                const rate = this.drive[key];
                if (!(rate > 0) || !this.groups[key]) continue;
                const pr = rate * dt / 1000;
                const g = this.groups[key];
                for (let j = 0; j < g.length; j++) if (rand() < pr) gE[g[j]] += p.stim_weight;
            }
            for (let i = 0; i < n; i++) {
                gE[i] *= dE;
                gI[i] *= dI;
                if (ref[i] > 0) { V[i] = p.V_reset; ref[i] -= dt; continue; }
                let v = V[i] + (p.g_leak * (p.E_leak - V[i]) + gE[i] * (p.E_exc - V[i]) + gI[i] * (p.E_inh - V[i])) * invC;
                if (v >= p.V_th) {
                    v = p.V_reset;
                    ref[i] = p.refrac;
                    this.spikeCount[i]++;
                    this.lastSpike[i] = this.t;
                    if (pl && pl.inStart[i + 1] > pl.inStart[i]) {
                        for (let q = pl.inStart[i]; q < pl.inStart[i + 1]; q++) {
                            const gap = this.t - lastSpike[this.preOf[pl.k[q]]];
                            if (gap < 5 * PLASTICITY.tau_pre) pl.elig[q] += Math.exp(-gap / PLASTICITY.tau_pre);
                        }
                    }
                    const target = sign[i] > 0 ? gE : gI;
                    for (let k = rowStart[i]; k < rowStart[i + 1]; k++) target[post[k]] += w[k];
                }
                V[i] = v < -120 ? -120 : v;
            }
            this.t += dt;
        }
        if (pl) {
            const decay = Math.exp(-ms / PLASTICITY.tau_elig);
            for (let q = 0; q < pl.elig.length; q++) pl.elig[q] *= decay;
            for (let g = 0; g < pl.gate.length; g++) pl.gate[g] *= decay;
        }
        const rates = {};
        for (const k in this.groups) {
            const g = this.groups[k];
            let c = 0;
            for (let j = 0; j < g.length; j++) c += this.spikeCount[g[j]];
            rates[k] = g.length ? c / g.length / (ms / 1000) : 0;
        }
        return rates;
    }

    // z-score each motor pool against its own running baseline. Baseline only
    // updates on calibration windows (no sensory drive), so fighting can't erase it.
    calibrate(rates) {
        for (const k of this.readout) {
            const b = this.base[k];
            b.n++;
            const d = rates[k] - b.mean;
            b.mean += d / b.n;
            b.var += (d * (rates[k] - b.mean) - b.var) / b.n;
        }
    }

    z(rates) {
        const out = {};
        for (const k of this.readout) {
            const b = this.base[k];
            out[k] = (rates[k] - b.mean) / Math.sqrt(Math.max(b.var, 0.25));
        }
        return out;
    }
}
