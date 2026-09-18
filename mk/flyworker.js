// Runs the maleCNS fly brain off the main thread. The game posts sensory drive
// every 100 ms; the worker simulates 100 ms of spiking and returns motor-pool z-scores.
// If the brain falls behind, the game skips ticks rather than queueing them.

import { FlyBrain, SENSORS, MOTORS, PLASTICITY } from './flybrain.js';

let FLY_CIRCUIT;  // loaded on init: the connectome, or the rewired control

const TICK_MS = 100;
let brain, rasterIdx;

const zero = () => Object.fromEntries(SENSORS.map(s => [s, 0]));

// Resting baseline for the z-scored readout: measured once on the fresh brain, then frozen.
// (Re-measuring it between rounds averaged in leftover fight activity and drifted the readout.)
function calibrate(windows) {
    brain.setDrive(zero());
    for (let i = 0; i < 5; i++) brain.step(TICK_MS);
    for (let i = 0; i < windows; i++) brain.calibrate(brain.step(TICK_MS));
}

// Between rounds: settle the network back to rest without touching the baseline or learned weights.
function rest(windows) {
    brain.setDrive(zero());
    brain.resetState();
    for (let i = 0; i < windows; i++) brain.step(TICK_MS);
}

// Each pool's dynamic range: its mean z while every sensory channel is driven at
// 100 Hz. Dividing by it stops tiny pools (TTMn is 2 neurons) from winning on z-scale alone.
function gains(zFloor, windows = 10) {
    brain.setDrive(Object.fromEntries(SENSORS.map(s => [s, 100])));
    for (let i = 0; i < 3; i++) brain.step(TICK_MS);
    const acc = Object.fromEntries(MOTORS.map(k => [k, 0]));
    for (let i = 0; i < windows; i++) {
        const z = brain.z(brain.step(TICK_MS));
        for (const k of MOTORS) acc[k] += z[k] / windows;
    }
    return Object.fromEntries(MOTORS.map(k => [k, Math.max(acc[k], zFloor)]));
}

function rasterSample() {
    const order = ['loom_L', 'loom_R', 'object_L', 'object_R', 'target_L', 'target_R', 'p1', 'taste', 'body_touch', ...MOTORS];
    rasterIdx = [];
    const motor = [];
    for (const k of order) {
        const g = brain.groups[k] || [];
        const take = Math.min(g.length, 8);
        for (let j = 0; j < take; j++) {
            rasterIdx.push(g[Math.floor(j * g.length / take)]);
            motor.push(MOTORS.includes(k) ? 1 : 0);
        }
    }
    return motor;
}

onmessage = async ({ data: m }) => {
    if (m.type === 'init') {
        ({ FLY_CIRCUIT } = await import(m.circuit === 'rewired' ? './fly_circuit_rewired.js' : './fly_circuit.js'));
        brain = new FlyBrain(FLY_CIRCUIT, { seed: m.seed });
        if (m.lr) PLASTICITY.lr = m.lr;
        if (m.baseline) PLASTICITY.baseline = m.baseline;
        // plastic synapses exist even with learning off, so a saved brain can still be loaded
        const plastic = brain.enablePlasticity(MOTORS, { premotor: m.premotor || 0 });
        calibrate(25);
        const gain = gains(m.zFloor);
        rest(10);
        const synapses = FLY_CIRCUIT.w.reduce((a, b) => a + b, 0);
        postMessage({
            type: 'ready', gain, rasterMotor: rasterSample(),
            meta: { neurons: FLY_CIRCUIT.neurons.length, connections: FLY_CIRCUIT.pre.length, synapses, plastic, fingerprint: brain.fingerprint() },
        });
    } else if (m.type === 'executed') {
        brain.executed(m.pool);
    } else if (m.type === 'reward') {
        const strength = brain.reward(m.r);
        if (strength) postMessage({ type: 'learned', strength, r: m.r });
    } else if (m.type === 'resetLearning') {
        brain.resetPlasticity();
        if (brain.plastic) postMessage({ type: 'learned', strength: brain.strength(), r: 0 });
    } else if (m.type === 'getWeights') {
        const ratios = brain.getRatios();
        postMessage({ type: 'weights', id: m.id, ratios, strength: brain.strength(), fingerprint: brain.fingerprint() }, [ratios.buffer]);
    } else if (m.type === 'setWeights') {
        const ok = brain.setRatios(new Float32Array(m.ratios));
        postMessage({ type: 'weightsSet', id: m.id, ok });
        postMessage({ type: 'learned', strength: brain.strength(), r: 0 });
    } else if (m.type === 'rest') {
        rest(m.windows);
        postMessage({ type: 'rested' });
    } else if (m.type === 'step') {
        brain.setDrive(m.drive);
        const t0 = performance.now();
        const z = brain.z(brain.step(TICK_MS));
        const spikes = Uint8Array.from(rasterIdx, i => Math.min(255, brain.spikeCount[i]));
        postMessage({ type: 'step', id: m.id, z, spikes, ms: performance.now() - t0 }, [spikes.buffer]);
    }
};
