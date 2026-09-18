"""Build the maleCNS "fighter" circuit for the mk.js face-off (/mk).

Extracts a sensory -> descending -> motor subgraph from the Janelia/FlyEM
**maleCNS v1.0** connectome (adult male brain + ventral nerve cord) and writes
it to mk/fly_circuit.js for the in-browser LIF simulation.

Public downloads (no neuPrint token needed), put in data/malecns/:
  https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
    body-annotations-male-cns-v1.0-minconf-0.5.feather
    body-neurotransmitters-male-cns-v1.0.feather
    connectome-weights-male-cns-v1.0-minconf-0.5.feather   (152M edges, ~1 GB)

Channels (all cell types are named maleCNS v1.0 annotations; nothing invented):

  sensory (driven by the game)          motor readout (drives the fighter)
  -----------------------------         -----------------------------------
  loom_L/R   LPLC2 + LC4 (looming)      fwd    DNp09 (forward walking)
  target_L/R LC10a (male target track)  back   MDN   (moonwalker, backward walking)
  object_L/R LC9 (object motion; top VPN input to DNp09)
  recede     LPLC4, driven by the opponent's image contracting: the cue that a limb is being
             withdrawn and the attacker is briefly open
  p1         P1 = male-specific pC1     jump   TTMn  (giant-fiber jump motor neuron)
  taste      front-leg taste bristles   punch  front-leg motor neurons (fl)
  body_touch notum + mid-leg tactile
             bristles (-> AN17A026 -> MDN)
                                        kick   hind-leg motor neurons (hl)
                                        wing   wing motor neurons (wm, threat)

Intermediate neurons: ranked by sensory->motor flow (see flow_rank): input-fraction
influence spread forward from each sensory group times influence spread back
from each motor group. The top neurons fill the --max-neurons budget.

    python -m scripts.build_malecns_fighter
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.feather as pf
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "malecns")
OUT = os.path.join(ROOT, "mk", "fly_circuit.js")
ANN = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT = "body-neurotransmitters-male-cns-v1.0.feather"
W = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

INHIBITORY = {"gaba", "glutamate", "histamine"}  # Shiu et al. 2024 convention


def sensors(a: pd.DataFrame) -> dict[str, pd.DataFrame]:
    t = a["type"].fillna("")
    vis = a[t.isin(["LPLC2", "LC4"])]
    tgt = a[t == "LC10a"]
    obj = a[t == "LC9"]  # top visual input to DNp09 (forward walking) in maleCNS
    # LPLC4 has the strongest excitatory reach into the hind-leg (kick) pool of any visual type,
    # which is what this cue has to be able to move; the encoding below is ours, not its known tuning.
    recede = a[t == "LPLC4"]
    # P1 = male-specific pC1 clusters (Yu 2010 pMP4 / Rideout 2010 pC1)
    p1 = a[t.str.startswith("pC1") & a["dimorphism"].fillna("").str.contains("male")]
    # front-leg bristles are annotated gustatory: males taste each other with their forelegs
    taste = a[(a["superclass"] == "vnc_sensory") & (a["subclass"] == "leg bristle")
              & (a["entryNerve"] == "ProLN")]
    # thorax (notum) + mid-leg tactile bristles: they converge on AN17A026, a top excitatory input to MDN
    body_touch = a[(a["class"] == "mechanosensory_tactile")
                   & ((a["subclass"] == "notum")
                      | ((a["subclass"] == "mechanosensory bristle") & (a["entryNerve"] == "MesoLN")))]
    return {
        "loom_L": vis[vis.somaSide == "L"], "loom_R": vis[vis.somaSide == "R"],
        "target_L": tgt[tgt.somaSide == "L"], "target_R": tgt[tgt.somaSide == "R"],
        "object_L": obj[obj.somaSide == "L"], "object_R": obj[obj.somaSide == "R"],
        "recede": recede,
        "p1": p1, "taste": taste, "body_touch": body_touch,
    }


def motors(a: pd.DataFrame) -> dict[str, pd.DataFrame]:
    t = a["type"].fillna("")
    mn = a[a["superclass"] == "vnc_motor"]
    return {
        "fwd": a[t == "DNp09"],
        "back": a[t == "MDN"],
        "jump": a[t == "TTMn"],
        "punch": mn[(mn.subclass == "fl") & (t[mn.index] != "TTMn")],
        "kick": mn[mn.subclass == "hl"],
        "wing": mn[(mn.subclass == "wm") & (t[mn.index] != "TTMn")],
    }


def load_edges(min_w: int) -> pd.DataFrame:
    tbl = pf.read_table(os.path.join(DATA, W), memory_map=True)
    tbl = tbl.filter(pc.greater_equal(tbl["weight"], min_w))
    return tbl.to_pandas()


def flow_rank(e: pd.DataFrame, sens: dict, mots: dict, depth: int) -> pd.Series:
    """Score every neuron by the sensory->motor signal it carries.

    W[post, pre] = synapses / total input synapses of post (input fraction).
    f_g = sum_h W^h s_g     forward influence of sensory group g (unit total drive)
    b_k = sum_h (W^T)^h m_k  backward influence onto motor group k (unit total weight)
    score = max over routes (g, k) of f_g * b_k / max(f_g * b_k)

    Every sense->motor route is normalised to its own best carrier, so a thin
    route (body touch -> AN17A026 -> MDN) keeps its relays instead of losing
    the budget to the heaviest routes (looming -> giant fiber -> wings/jump).
    """
    ids = np.unique(np.concatenate([e.body_pre.to_numpy(), e.body_post.to_numpy()]))
    ix = pd.Series(np.arange(len(ids)), index=ids)
    pre, post = ix[e.body_pre].to_numpy(), ix[e.body_post].to_numpy()
    w = e.weight.to_numpy(np.float64)
    tot_in = np.bincount(post, weights=w, minlength=len(ids))
    W = sp.csr_matrix((w / tot_in[post], (post, pre)), shape=(len(ids), len(ids)))
    WT = W.T.tocsr()

    def spread(M, group):
        acc = np.zeros(len(ids))
        members = ix.reindex(group.bodyId.astype(np.int64)).dropna().astype(int).to_numpy()
        if not len(members):
            return acc
        v = np.zeros(len(ids))
        v[members] = 1.0 / len(members)
        for _ in range(depth):
            v = M @ v
            acc += v
        return acc

    fs = [spread(W, g) for g in sens.values()]
    bs = [spread(WT, m) for m in mots.values()]
    score = np.zeros(len(ids))
    for f in fs:
        for b in bs:
            route = f * b
            top = route.max()
            if top > 0:
                np.maximum(score, route / top, out=score)
    return pd.Series(score, index=ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-w", type=int, default=3, help="min synapses for kept edges")
    ap.add_argument("--depth", type=int, default=3, help="hops of forward/backward flow")
    ap.add_argument("--max-neurons", type=int, default=12000)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--rewire", type=int, default=0, metavar="SEED",
                    help="control: shuffle who connects to whom, keeping every degree, sign and weight")
    ap.add_argument("--probe-types", default="", help="semicolon-separated cell types to add as extra sensory groups (for probing)")
    args = ap.parse_args()

    a = pd.read_feather(os.path.join(DATA, ANN))
    a = a[a["status"].isin(["Traced", "Anchor"])].reset_index(drop=True)
    nt = pd.read_feather(os.path.join(DATA, NT))[["body", "consensus_nt", "predicted_nt"]]
    nt = nt.set_index("body")

    sens, mots = sensors(a), motors(a)
    for t in filter(None, args.probe_types.split(";")):
        sens[f"probe:{t}"] = a[a["type"] == t]
    for k, v in {**sens, **mots}.items():
        print(f"  {k:9s} {len(v):4d}  {sorted(v['type'].dropna().unique())[:6]}")
    sens_ids = {int(b) for v in sens.values() for b in v.bodyId}
    mot_ids = {int(b) for v in mots.values() for b in v.bodyId}

    print("loading edges ...")
    e = load_edges(args.edge_w)
    print(f"  {len(e):,} edges >= {args.edge_w} synapses")
    score = flow_rank(e, sens, mots, args.depth)
    keep = set(sens_ids) | set(mot_ids)
    # the neurons carrying the most sensory->motor flow; cap keeps the browser sim real-time
    for v in score[score > 0].sort_values(ascending=False).index:
        if len(keep) >= args.max_neurons:
            break
        keep.add(int(v))
    print(f"  {int((score > 0).sum()):,} neurons carry sensor->motor flow within {args.depth}+{args.depth} hops; keeping {len(keep):,}")

    ids = sorted(keep)
    index = {b: i for i, b in enumerate(ids)}
    sub = e[e.body_pre.isin(keep) & e.body_post.isin(keep)]
    sub = sub[sub.body_pre != sub.body_post]
    print(f"  {len(sub):,} edges in circuit")

    ann = a.set_index("bodyId")
    groups = {k: [index[int(b)] for b in v.bodyId if int(b) in index] for k, v in {**sens, **mots}.items()}
    neurons = []
    n_inh = 0
    for b in ids:
        r = ann.loc[b] if b in ann.index else None
        ntv = nt.loc[b] if b in nt.index else None
        tr = None
        if ntv is not None:
            tr = ntv["consensus_nt"] if isinstance(ntv["consensus_nt"], str) else ntv["predicted_nt"]
        sign = -1 if (isinstance(tr, str) and tr in INHIBITORY) else 1
        # motor neurons are glutamatergic onto muscle (excitatory at the NMJ); they have no CNS outputs here
        if r is not None and r["superclass"] == "vnc_motor":
            sign = 1
        n_inh += sign < 0
        neurons.append([
            str(r["type"]) if r is not None and isinstance(r["type"], str) else "",
            str(r["somaSide"]) if r is not None and isinstance(r["somaSide"], str) else "",
            str(r["superclass"]) if r is not None and isinstance(r["superclass"], str) else "",
            sign,
        ])
    print(f"  {n_inh} inhibitory (GABA/Glu/His) of {len(ids)}")

    pre = sub.body_pre.map(index).to_numpy(np.int32)
    post = sub.body_post.map(index).to_numpy(np.int32)
    w = sub.weight.to_numpy(np.int32)

    if args.rewire:
        # Null model: permuting the post column keeps every in-degree and out-degree exactly (each
        # neuron appears as often as before, on either side) and leaves weights and transmitter signs
        # untouched. Only which neuron talks to which is destroyed. Self-loops and repeat pairs are
        # re-drawn. If a fly wired this way learns as well as the real one, the connectome is
        # decoration and the readout is doing the work.
        rng = np.random.default_rng(args.rewire)
        post = rng.permutation(post)
        for _ in range(20):
            seen = set()
            bad = []
            for k in range(len(pre)):
                key = (pre[k], post[k])
                if pre[k] == post[k] or key in seen:
                    bad.append(k)
                else:
                    seen.add(key)
            if not bad:
                break
            post[bad] = rng.permutation(post[bad])
        print(f"  rewired (seed {args.rewire}): {len(bad):,} self/duplicate edges left of {len(pre):,}")
    order = np.argsort(pre, kind="stable")
    payload = {
        "dataset": "maleCNS v1.0 (Janelia FlyEM), minconf 0.5" + (f" — REWIRED control, seed {args.rewire}" if args.rewire else ""),
        "source": "https://male-cns.janelia.org/download/",
        "params": {k: v for k, v in vars(args).items() if k != "out"},
        "bodyIds": [str(b) for b in ids],
        "neurons": neurons,  # [type, somaSide, superclass, sign]
        "groups": groups,
        "pre": pre[order].tolist(), "post": post[order].tolist(), "w": w[order].tolist(),
    }
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("// Generated by scripts/build_malecns_fighter.py from maleCNS v1.0. Do not edit.\n")
        f.write("export const FLY_CIRCUIT = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
