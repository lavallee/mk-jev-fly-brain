// A local fighting policy small enough to answer on the fly's 100 ms clock.
//
// Jev sees a JSON state and returns a probability for each of the seven moves. `features()` turns
// that same state into a fixed vector, `LocalPolicy` is a softmax over it (one weight per feature
// per move), and `learn()` is one gradient step towards Jev's distribution. Training on Jev's
// probabilities rather than its pick teaches far more per example.
//
// The encoder lives here so the logged training data and the in-game policy can never drift apart:
// matches log the vector this file produced, and the policy consumes the same vector.

export const ACTIONS = ['stand', 'walk_forward', 'walk_backward', 'jump_away', 'punch', 'kick', 'block'];

// mk.js move types collapsed to what matters for a decision
const MOVE_CLASSES = ['stand', 'walk_forward', 'walk_backward', 'jump', 'attack', 'block', 'hurt'];

function moveClass(move = '') {
    if (/punch|kick|uppercut/.test(move)) return 'attack';
    if (/block/.test(move)) return 'block';
    if (/endure|knock|fall/.test(move)) return 'hurt';
    if (/jump/.test(move)) return 'jump';
    if (move === 'walking') return 'walk_forward';
    if (move === 'walking-backward') return 'walk_backward';
    return 'stand';
}

export const FEATURE_NAMES = [
    'bias', 'my_life', 'opp_life', 'life_lead', 'distance', 'in_range', 'closing', 'busy', 'time_left',
    ...MOVE_CLASSES.map(m => `me_${m}`), ...MOVE_CLASSES.map(m => `opp_${m}`), 'opp_attacking',
];

// Same units the state uses, scaled to roughly 0..1 so one learning rate suits every feature.
export function features(state) {
    const me = state.you || {}, opp = state.opponent || {};
    const x = new Float32Array(FEATURE_NAMES.length);
    const mine = moveClass(me.action), theirs = moveClass(opp.action);
    x[0] = 1;
    x[1] = (me.life || 0) / 100;
    x[2] = (opp.life || 0) / 100;
    x[3] = ((me.life || 0) - (opp.life || 0)) / 100;
    x[4] = Math.min(1, (state.distance_px || 0) / 600);
    x[5] = state.in_attack_range ? 1 : 0;
    x[6] = Math.max(-1, Math.min(1, (state.closing_px_per_s || 0) / 300));
    x[7] = me.busy ? 1 : 0;
    x[8] = Math.min(1, (state.time_left_s || 0) / 60);
    x[9 + MOVE_CLASSES.indexOf(mine)] = 1;
    x[9 + MOVE_CLASSES.length + MOVE_CLASSES.indexOf(theirs)] = 1;
    x[9 + 2 * MOVE_CLASSES.length] = opp.attacking ? 1 : 0;
    return x;
}

export class LocalPolicy {
    constructor(weights = null, { lr = 0.05 } = {}) {
        this.n = FEATURE_NAMES.length;
        this.lr = lr;
        this.w = weights ? Float32Array.from(weights) : new Float32Array(this.n * ACTIONS.length);
        this.trained = 0;
    }

    // { action, probabilities, confidence } — confidence is how far the best move leads the runner-up
    predict(x) {
        const logits = new Float32Array(ACTIONS.length);
        for (let a = 0; a < ACTIONS.length; a++) {
            let sum = 0;
            for (let i = 0; i < this.n; i++) sum += this.w[a * this.n + i] * x[i];
            logits[a] = sum;
        }
        const max = Math.max(...logits);
        let total = 0;
        const p = new Float32Array(ACTIONS.length);
        for (let a = 0; a < ACTIONS.length; a++) { p[a] = Math.exp(logits[a] - max); total += p[a]; }
        for (let a = 0; a < ACTIONS.length; a++) p[a] /= total;
        const order = [...p].map((v, a) => [v, a]).sort((u, v) => v[0] - u[0]);
        return {
            action: ACTIONS[order[0][1]],
            probabilities: Object.fromEntries(ACTIONS.map((name, a) => [name, p[a]])),
            confidence: order[0][0] - order[1][0],
            p,
        };
    }

    // One step towards Jev's distribution (softmax cross-entropy gradient).
    learn(x, target) {
        const { p } = this.predict(x);
        for (let a = 0; a < ACTIONS.length; a++) {
            const err = p[a] - (target[ACTIONS[a]] ?? 0);
            if (!err) continue;
            for (let i = 0; i < this.n; i++) this.w[a * this.n + i] -= this.lr * err * x[i];
        }
        this.trained++;
    }

    export() { return Array.from(this.w, v => +v.toFixed(5)); }

    // Weights distilled offline by scripts/train_local_policy.py; an untrained policy if absent.
    static async load(url = 'policy_weights.json', options = {}) {
        try {
            const r = await fetch(url);
            if (!r.ok) return new LocalPolicy(null, options);
            const { w, decisions } = await r.json();
            const policy = new LocalPolicy(w, options);
            policy.distilledFrom = decisions || 0;
            return policy;
        } catch {
            return new LocalPolicy(null, options);
        }
    }
}
