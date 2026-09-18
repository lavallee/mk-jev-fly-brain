// Drive one sensory channel at a time and report every readout pool's z-score.
//   node scripts/js/probe_channels.mjs <fly_circuit.js> [rate]
import { pathToFileURL } from 'node:url';
import { FlyBrain } from '../../app/static/mk/flybrain.js';

const file = process.argv[2] ?? 'app/static/mk/fly_circuit.js';
const rate = Number(process.argv[3] ?? 100);
const { FLY_CIRCUIT: C } = await import(pathToFileURL(file));
const READ = ['fwd', 'back', 'jump', 'punch', 'kick', 'wing'].filter(k => C.groups[k]?.length);
const CHANNELS = {
    loom: ['loom_L', 'loom_R'], object: ['object_L', 'object_R'], target: ['target_L', 'target_R'],
    recede: ['recede'], p1: ['p1'], taste: ['taste'], body_touch: ['body_touch'],
};
for (const g of Object.keys(C.groups)) if (g.startsWith('probe:')) CHANNELS[g.slice(6)] = [g];
const brain = new FlyBrain(C, { seed: 7, readout: READ });
const zero = () => Object.fromEntries(Object.values(CHANNELS).flat().map(k => [k, 0]));
brain.setDrive(zero());
for (let i = 0; i < 5; i++) brain.step(100);
for (let i = 0; i < 30; i++) brain.calibrate(brain.step(100));
console.log(`${file}  rate=${rate} Hz`);
console.log('channel'.padEnd(15) + READ.map(k => k.padStart(9)).join(''));
for (const [name, groups] of Object.entries(CHANNELS)) {
    if (!groups.every(g => C.groups[g]?.length)) continue;
    brain.setDrive(zero());
    for (let i = 0; i < 10; i++) brain.step(100);
    brain.setDrive({ ...zero(), ...Object.fromEntries(groups.map(g => [g, rate])) });
    const acc = Object.fromEntries(READ.map(k => [k, 0]));
    for (let i = 0; i < 10; i++) { const z = brain.z(brain.step(100)); for (const k of READ) acc[k] += z[k] / 10; }
    console.log(name.slice(0, 14).padEnd(15) + READ.map(k => acc[k].toFixed(1).padStart(9)).join(''));
}
