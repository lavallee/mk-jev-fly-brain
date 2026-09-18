# Model Kombat: what we tested, in the order we tested it

A record of the experiments behind `/mk`, where a 12,000-neuron slice of the **maleCNS v1.0**
connectome fights **Jev** (TypeSafe's System One model) in mk.js. Written as we went, including the
arms that failed and the predictions that were wrong.

Two things are worth stating before the log, because they frame everything in it:

- **The information is asymmetric.** Jev is handed `in_attack_range` — the game's own hit predicate —
  plus both life totals and the clock. The fly gets modelled visual and tactile drives and never
  learns the score. This was never a clean contest between two intelligences.
- **Most arms are n=1.** The headline result is replicated twice; almost nothing else is. Each time
  the fly failed we changed the machinery until it stopped failing, which is a search over our own
  design space with the fly as substrate.

---

## 1. Building the circuit

**Question.** Which neurons should be in the fight at all?

**What we did.** Extracted a sensory → descending → motor subgraph from maleCNS: looming detectors
(LPLC2, LC4), motion (LC9), target size (LC10a), male-specific P1, foreleg taste and body touch
bristles on the input side; DNp09 (forward walking), MDN (backward), TTMn (giant-fibre jump),
front- and hind-leg and wing motor neurons on the output side.

**Finding — the selection method changes the biology you see.** An early cut at 8,000 neurons
produced a strong body-touch → MDN response (z ≈ 50–70) that vanished entirely in a
20,000-neuron reference build. We replaced the arbitrary cut with a flow ranking (input-fraction
influence forward from each sense times influence back from each pool, normalised per route) and
settled at 12,000 neurons, which matches the 20,000 reference on every robust response.

**Finding — the connectome routes senses selectively.** Looming drives the escape jump at z 81 and
almost nothing else. Motion drives forward walking at z 103. Body touch drives the kick pool at
9.9. No visual cue reaches the legs strongly. That last fact closes off a whole strategy later.

**Failure worth recording.** The first simulation pinned the entire network at ~440 Hz: the
synaptic weight scale was too strong for a dense recurrent graph. Cutting it from 0.6 to 0.05 nS
per synapse produced sane, selective responses.

## 2. First fight: Jev 5–2

Seven rounds, sides alternating, no learning. The fly landed **more hits** (103 to 85) and dealt
**less damage** (493 to 694), because Jev blocked: a guarded hit is worth 2 damage instead of 10.

Two bugs surfaced, both ours, both instructive:

- **The fly jumped into walls when it stood on the right.** Distance was computed from sprite
  widths, which grow as a limb extends, so an opponent's punch *read as a lunge* — but only from
  one side. Measuring from the fixed collision box fixed it, and Jev's `distance_px` had the same
  fault.
- **A two-neuron pool won every argument.** TTMn is two neurons, so its z-score swamped the
  130-neuron leg pools. Each pool is now scaled by its response to full sensory drive.

We also replaced "approach speed in pixels" with the **angular expansion** of the opponent's image,
with a threshold: a slow walk-in stays under it, a lunge or a strike saturates it. That is what a
looming detector actually responds to.

## 3. Learning, and the credit-assignment bug that hid it

**First attempt.** Dopamine-like three-factor plasticity on the ~25,000 excitatory synapses onto
the motor pools. Result: no help at all (Jev 6–1, 7–0). At a higher learning rate it was actively
destructive — blocking collapsed to zero.

**Diagnosis.** The escape jump and the wing guard are both driven by the looming detectors and fire
together, so every reward or punishment hit both. Learning could not separate the reflex that was
hurting from the one that was helping.

**Fix — an efference copy.** The brain now receives a copy of the move that was actually executed,
and only synapses onto that pool can change. Dopamine became a reward *prediction error*: blocking
a strike counts as better than expected, eating one as worse.

**Result.** The fly won **5–2 in two separate 7-round matches**, unlearning futile escape jumps
(jump synapses 0.89×) and building a guard (wings up to 1.6×).

**Instrumentation bug found along the way.** The resting baseline the z-scores are measured against
was being re-measured half a second after each round ended, absorbing leftover fight activity. That
flattered early rounds and produced a spurious "wins early, fades late" pattern. The baseline is
now measured once on a fresh brain and frozen.

## 4. How much is the fly's speed worth?

The fly decides 10 times a second; Jev manages 5–7 with a ~350 ms round trip. Is that the whole
story? Twenty valid rounds (three dropped to a TypeSafe outage), learning off:

| Fly setting | Reaction | Attacks/s | Fly damage/s | Rounds |
|---|---|---|---|---|
| Instant (normal) | 0 ms | 0.65 | 4.64 | 1–4 |
| `?flyevery=2` (half rate) | 0 ms | 0.34 | 3.20 | 0–5 |
| `?lag=150` | 150 ms | 0.54 | 3.40 | 0–4 |
| `?lag=auto` (Jev's latency) | 386 ms | 0.18 | **1.37** | **0–6** |

**Finding.** Both fighters convert to the same ~2 moves/s, because an attack locks a fighter for
400–600 ms; the fly's extra decisions are ignored 48% of the time against Jev's 20%. But reaction
speed is the fly's entire edge: at parity its damage output falls by 70% and it never wins. Roughly
1 damage/s per 100 ms of added delay.

## 5. Compensating Jev's latency — which made Jev worse

**Idea.** Rather than blunting the fly, cancel Jev's handicap: send it the fight as it will be when
its answer lands (distance and range projected forward, each fighter's remaining commitment, the
opponent's move marked unknown if it will have ended).

| Jev | Damage/s | Rounds |
|---|---|---|
| As-is | **5.67** | 7–1 |
| Geometry projected only | 4.87 | 6–2 |
| Fully projected | **4.56** | **3–5** |

**Finding, measured rather than guessed.** The state Jev acts on is *not stale*: the median
distance error between what it was told and reality when its answer landed is **0 px**, because
both fighters are standing or locked most of the time. Projecting ahead only adds error (median 3.8
px) and makes it attack at moments that have passed. Latency hurts Jev only inside the 80–160 ms
attack windows, which linear extrapolation cannot predict.

## 6. Distilling Jev into 168 weights

**Setup.** `?log=jev` records every decision as (feature vector, the probability Jev gave each
move). One match produced 928 examples. We fitted a softmax over 24 features — training on Jev's
*probabilities*, not its picks.

**Fidelity.** 77% agreement on held-out decisions, **93% on the calls Jev was confident about**,
KL 0.046. It predicts in **4 µs** against Jev's ~350 ms.

**The surprise.** It doesn't just match Jev, it beats it, purely by acting sooner:

| Jev's side | API calls/s | Damage/s | Rounds vs fly |
|---|---|---|---|
| Jev API | 6.9 | 5.69 | 7–1 |
| Local policy only | **0** | 6.14 | 8–0 |
| Hybrid (local moves, Jev corrects and teaches live) | 6.8 | **7.66** | 8–0 |

The hybrid is the fairness fix the handicap flags could not produce: at 10 decisions/s on both
sides, judgment beat reflexes 8–0.

## 7. The rule-bot control

**Question.** Is any of this about intelligence, or does this game simply reward simple rules?

**What we did.** Wrote a five-line opponent: block an in-range attacker, strike in range, else
close. Same state, same 100 ms clock.

**Finding.** It beats the wired fly **6–0 at 4.92 damage/s** — the same band as Jev's API (5.69) and
the distilled policy (6.14). Three very different systems, one performance band.

## 8. Reading each fighter back as rules

Fitting the same 24-feature model to each fighter's own moves:

| | Predicted | Policy in plain terms |
|---|---|---|
| Jev | 77% | Walk when **not in range**, strike **in range**, block when the **opponent attacks** |
| Fly, as wired | 83% | Walk when **far**, **jump away when close**, kick when **hurt** |
| Fly, after 7 generations | 94% | Kick when **close and closing**, block when the opponent **guards** or it is **ahead**, **no jumping** |

**Finding.** Dopamine learning rediscovered roughly the hand-written rules. The wired fly's fatal
habit was the escape jump — a real circuit doing exactly its job, which happens to be useless
against someone who keeps punching. Run on the connectome's clock, those learned rules beat the
rule bot **6–0 (10.57 vs 2.11 damage/s)**, reversing the wired fly's 0–6.

## 9. Cumulative learning, and the reward baseline that unlocked it

**First attempt.** Inherit the best brain each match and keep learning. Against the fast local
policy the fly went to **zero damage by generation 2**, with every pool driven below its wired
strength.

**Diagnosis.** A losing fly has every move punished, so it suppresses all of them, so it does less,
so it gets hit more.

**Fix.** Judge each dopamine signal against a running average of recent ones (`?baseline=0.02`), so
the *relative* difference between moves survives a losing streak. Also extended plasticity to the
synapses onto the 500 neurons that drive the pools hardest (25k → 60k plastic synapses, +4 ms per
tick).

**Result**, 8 generations × 3 rounds:

| Opponent | Record | Fly damage/s | Opponent |
|---|---|---|---|
| Jev API | **24–0** | 6.1–8.2 (was 4.79) | 2.2–4.7 (was 5.69) |
| Distilled local policy | **23–1** | 8.8 (was 2.2) | 1.1 (was 6.14) |
| Hybrid | **0–23** | 0.0 from gen 2 | 7.3 → 2.3 |

Inheritance helps early then plateaus: generation 1 scored 4.88 damage/s, generations 3+ about 8.8.

## 10. Trying to teach it timing

**Question.** Can it learn to strike while the opponent recovers?

**Attempt 1 — give it the missing sense.** An extending limb and a retracting one are both motion to
the fly. We added a receding-motion channel and picked its cell type by wiring: **LC11 reaches no
motor pool at all** (every |z| < 1); **LPLC4**, the visual type with the strongest excitatory reach
into the hind-leg pool, lands on the **wings** (z 11.6), not the legs (0.8).

**Finding.** In this subgraph no visual cue has a strong path to the legs. The kick pool is
reachable from body touch (9.9) and looming (8.1) — which is exactly why the wired fly's rule was
"kick when hurt". Its counter-attack is a *touch* reflex, not a visual one. The `recede` channel is
wired up and fires, but it can only raise a guard.

**Attempt 2 — change the policy class.** `?thresh=0.15` makes the bar each pool must clear
learnable: landing a hit lowers it, taking damage raises it. It expresses the one thing pool gains
cannot — whether a reflex is worth acting on at all.

**Result against the hybrid**: the fly raised its own bar for kicking (4 → 10.5) and jumping
(4 → 9.2), stopped attacking (0.02 attacks/s) and played for the clock. **Losses 17 → 8, draws
7 → 16, life left at the end 29 → 67.** Never a win. A stalemate is the best answer available to a
policy class that can only choose *whether* to fire a reflex.

## 11. Curriculum: the hybrid is beatable after all

**Prediction (wrong).** We told the user to expect the fly to lose this.

**What happened.** A fly that arrives with a brain already evolved against the rule bot, and keeps
learning, beats the hybrid **5–2 over seven 60 s rounds — twice**, 5.14 and 5.16 damage/s against
5.47 and 6.06, attacking 1.6 times a second where the from-scratch fly attacked 0.02. A brain that
has *already fought the hybrid* does better still: 5–2 at **6.52 damage/s**, out-damaging it.

**Finding.** Nothing about the machinery changed — only the order of opponents. Meeting the hybrid
cold, every opening is punished and the fly learns not to fight. Arriving with a guard and a low
kick threshold, it can fight. Its thresholds then track the score *within* a match: the kick bar
sits at 1.5–2.0 while winning and jumps to 4.6–6.1 after each loss.

## 12. The rewired control: how much is the connectome doing?

**Question.** Is the wiring doing work, or is our hand-built readout?

**What we did.** Shuffled who connects to whom while preserving every neuron, its in-degree and
out-degree, its transmitter sign and every synapse weight. Same senses, pools, readout, learning
rule, opponents.

| Against the rule bot | Real connectome | Rewired |
|---|---|---|
| Record | 23–1 | **23–1** |
| Damage/s | **7.59** | 3.89 |
| Attacks/s | 1.54 | 1.92 |
| Hits landed/s | 1.36 | 1.67 |
| **Damage per hit** | **5.6** | **2.3** |

| Against the hybrid, 7 rounds | Real | Rewired |
|---|---|---|
| Record | **5–2, twice** (every round a KO) | 2 wins, 1 loss, 4 draws |
| Damage/s | 5.14 / 5.16 | 2.46 |

**Finding.** A scrambled brain still learns to fight — so the fighting competence comes mostly from
the readout, the reward and the curriculum we built, and would survive scrambling the connectome.
But it cannot hit: it swings *more*, connects *more*, and does half the damage, because 2.3 damage
per hit is what a blocked hit is worth. It attacks into a raised guard. Rewired, every sense drives
every pool in proportion to its population; the real circuit routes selectively.

**The competence is ours. The timing is the connectome's.**

---

## 13. The round robin: every competitor against every other

Until now each fly had only ever been entered against the opponent that had just beaten its
predecessor, so the match-up grid was a staircase with fourteen holes in it. Filling them took 56
more matches, played headlessly by `scripts/run_matches.py` — the real page in headless Chromium,
one arm at a time, with the saved brains parked between arms so each curriculum starts from the
connectome instead of inheriting whatever was best in the archive.

Rounds won–drawn–lost by the fly, 140 matches, 433 rounds:

| | Rule bot | Jev API | Local policy | Hybrid |
|---|---|---|---|---|
| Wired (no learning) | 6–0–7 | 2–0–14 | 0–0–16 | 0–0–20 |
| In-match learning | 0–0–4 | 4–0–2 | 2–0–2 | 0–0–4 |
| In-match learning + baseline | 2–0–5 | 3–0–2 | 2–0–3 | 0–0–4 |
| Curriculum | 16–0–1 | 24–0–0 | 26–6–43 | 0–7–17 |
| Curriculum + thresholds | 23–0–1 | **4–0–16** | 18–0–0 | 0–16–8 |
| Reloaded brain | 6–0–0 | 5–0–4 | 6–0–0 | 21–0–8 |
| Rewired control | 23–0–1 | **0–1–10** | 0–6–5 | 2–4–1 |

Three things the blank cells had been hiding.

**The rewired control's 23–1 was about its schedule, not its wiring.** It had only ever played the
rule bot. Against Jev it goes 0–1–10, against the local policy 0–6–5. Shuffled wiring beats the
weakest opponent in the field and nothing else, and even there it wins by stalling: 2.3 damage per
landed hit against the real circuit's 5.5. Experiment 12 read one match-up as if it were general.
The claim that survives is narrower — *the connectome is what converts a decision into a hit that
lands*; what it is not is a general-purpose advantage that shows up against every opponent.

**Learnable thresholds are opponent-specific.** They sweep the local policy (18–0, 2–0 in all eight
generations) and hold the rule bot at 23–1, and they collapse against Jev — 4–16, where the plain
curriculum had gone 24–0. Raising a threshold buys patience; Jev is the one opponent that punishes
patience, because it closes distance faster than the fly re-decides.

**Transfer beats training length.** The reloaded brain — a curriculum brain carried in cold — is the
only fly with a winning record against all four (6–0, 5–4, 6–0, 21–8). The same connectome learning
from scratch inside a single match loses 0–4 to the rule bot the reloaded brain sweeps 6–0.

Two cells are thinner than the rest: the rewired arms against Jev and the local policy got 5 and 4
generations before hitting a wall-clock cap, not 8.

---

## What would change our minds

- **More replication.** Most arms are single runs. The 5–2 is twice; the rewired control has one
  arm per opponent, two of them short.
- **Symmetric information.** Give both sides the same inputs, or deny Jev the game's hit predicate,
  and the comparison becomes meaningful rather than illustrative.
- **A learned readout.** Ours is a ~12-parameter controller (per-pool gains and thresholds) sitting
  on a 12,000-neuron simulation. Fixing it and allowing only synaptic plasticity would test the
  connectome harder; learning it end-to-end would test it less.
- **Other rewiring nulls.** We preserved degrees, signs and weights. Preserving distance-dependence
  or cell-type block structure would be a stricter control.
- **A ladder.** Rule bot → API → local → hybrid, carrying the brain up each rung, against a fly
  trained on any single opponent.

## Reproducing

```bash
python -m scripts.build_malecns_fighter                  # the circuit (12k neurons)
python -m scripts.build_malecns_fighter --rewire 7 \
    --out app/static/mk/fly_circuit_rewired.js           # the control
python -m app.server                                     # then open /mk
```

Flags used above, all off by default, all recorded per match in `telemetry/mk/*.json`:
`?lag`, `?flyevery`, `?lead`, `?plastic`, `?baseline`, `?thresh`, `?evolve`, `?lineage`,
`?jev=api|hybrid|local|rules`, `?circuit=rewired`, `?log=jev|fly`, `?brain=<id>|best|wired`.
