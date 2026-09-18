# Model Kombat

A fly brain fights a language model in a 1990s arcade fighter, and the fight is instrumented well
enough to say what each side is contributing.

The **fly** is a 12,000-neuron sensory → descending → motor subgraph of the
[maleCNS v1.0](https://male-cns.janelia.org/) connectome — real wiring, real synapse counts, real
transmitter signs — simulated spike by spike in your browser at 10 decisions a second. The fight
drives named cell types (looming detectors LPLC2 and LC4, motion LC9, target size LC10a,
male-specific P1, foreleg taste and body-touch bristles) and the busiest motor pool picks the move
(DNp09 walks forward, MDN walks back, the giant-fibre neuron TTMn jumps, leg motor neurons punch
and kick, wing motor neurons raise a guard).

**Jev** is [TypeSafe](https://typesafe.ai/)'s System One model, which reads the fight as JSON and
answers with one typed choice among the same seven moves, about five times a second.

The game is [mk.js](https://github.com/mgechev/mk.js). Nothing in the fly is trained in advance;
what learning there is happens during the fight, through a dopamine-like rule.

**[EXPERIMENTS.md](EXPERIMENTS.md) is the interesting part** — twelve experiments in the order they
were run, failures and wrong predictions included. The standings and charts behind them,
generated from 84 archived matches, are at
**[lavallee.github.io/mk-jev-fly-brain](https://lavallee.github.io/mk-jev-fly-brain/)** — there is
no single fly and no single Jev, so each learning setup and each source of decisions is entered as
its own competitor.

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_game_assets.py     # mk.js sprites (not redistributable here — see CREDITS.md)
python server.py                        # -> http://127.0.0.1:5000/mk/
```

That runs everything that needs no account: the fly brain, the hand-written rule bot and the local
policy distilled from Jev. To use Jev itself, put a TypeSafe key in `.env`:

```bash
echo TYPESAFE_API_KEY=sk-... > .env
```

Sound effects and the announcer are generated or fetched separately (see `CREDITS.md`); without
them the game still has synthesized music and hits, and logs two 404s for the assets it looked
for.

## Things to try

```
/mk/?jev=rules                      # the fly against five lines of hand-written rules
/mk/?jev=local                      # against a 168-weight policy distilled from Jev — no API calls
/mk/?jev=api                        # against Jev itself, with its live JSON and probabilities shown
/mk/?jev=hybrid                     # the local policy moving fast, Jev correcting and teaching it
/mk/?evolve=8&learn=1&plastic=500&baseline=0.02&thresh=0.15&jev=rules
                                    # eight matches, each inheriting the best brain so far
/mk/?circuit=rewired                # the control: same neurons and degrees, connections shuffled
```

Every match writes a JSON file to `telemetry/mk/`, including the fly's learned synapse strengths,
and the *Starting brain* menu can begin a later match from any of them. Once you have matches of
your own:

```bash
python -m scripts.report_telemetry     # -> docs/index.html, charts from your own telemetry
```

## What we found

- **Reaction time is the fly's whole edge.** Slow it to Jev's ~350 ms and its damage output falls
  70% and it never wins. It converts decisions into moves no faster than Jev does — an attack locks
  a fighter for 400–600 ms — but it can answer inside the 80–160 ms window when a strike is already
  in flight, and Jev cannot.
- **Jev fits in 168 weights.** A softmax over 24 features, trained on its probabilities, agrees with
  it on 93% of the calls it was confident about and answers in 4 µs. Run at that speed it *beats the
  model it copied* (6.14 damage/s vs 5.69), and a hybrid — local speed, Jev correcting live — beats
  both at 7.66.
- **A five-line rule bot matches Jev** (4.92 damage/s). This game rewards simple positional rules;
  what separates the contenders is how fast and how consistently they apply them.
- **Learning needed two fixes before it did anything.** An efference copy, so dopamine could tell the
  escape jump from the wing guard when both fire together; and a reward baseline, so a losing fly
  compares its moves against each other instead of suppressing all of them.
- **Order of opponents decides the outcome.** Meeting the strongest opponent cold, the fly learns
  not to fight (0–23, zero damage). Arriving with a brain evolved against the rule bot, it wins 5–2,
  twice.
- **The connectome supplies timing, not competence.** A rewired control — same neurons, degrees,
  signs and weights, connections shuffled — still learns to beat the rule bot 23–1. It just cannot
  hit: it swings more, connects more, and does half the damage, because its hits land on a raised
  guard. Real wiring lands 5.6 damage per hit, shuffled wiring 2.3.

Two caveats worth carrying with any of it: Jev is handed the game's own hit predicate
(`in_attack_range`), plus both life totals and the clock, while the fly gets modelled sensory drives
and never learns the score; and most experiments are single runs.

## How it fits together

| | |
|---|---|
| `scripts/build_malecns_fighter.py` | builds `mk/fly_circuit.js` from the public maleCNS files (`--rewire` for the control) |
| `mk/flybrain.js`, `mk/flyworker.js` | the spiking simulation and its dopamine learning, in a Web Worker |
| `mk/fight.js` | senses, motor readout, the match loop, telemetry |
| `jev.py`, `server.py` | the TypeSafe proxy, telemetry storage and saved brains |
| `mk/policy.js`, `scripts/train_local_policy.py` | the distilled local policy, and fitting it |
| `scripts/probe_channels.mjs` (in `scripts/js/`) | drive one sense, see which motor pools answer |
| `scripts/report_telemetry.py` | reads every match JSON, writes `docs/index.html` — the charts above |

The physiology, the sensory encodings and the pool→move mapping are assumptions, documented in the
source. The wiring is not.
