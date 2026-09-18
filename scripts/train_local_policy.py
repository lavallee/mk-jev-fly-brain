"""Distil Jev into a local policy small enough to answer on the fly's 100 ms clock.

Reads the decisions logged by matches played with `?log=jev` (telemetry/mk/*.json): for each one,
the feature vector `mk/policy.js` produced and the probability Jev gave every move.
Trains a softmax regression on those probabilities (soft targets teach far more per example than
the winning label alone) and writes the weights for the page to load.

    python -m scripts.train_local_policy             # train on Jev, report, write weights
    python -m scripts.train_local_policy --report    # evaluate the saved weights only
    python -m scripts.train_local_policy --source fly --rules   # read the fly's own policy

`--source fly` fits the same model to the fly's chosen moves (logged with `?log=fly`), so the
connectome's behaviour can be read as the same kind of rules and compared with Jev's.

Held-out split is temporal: the last rounds of each match are the test set, so a policy can't score
well by memorising positions it will meet again.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEMETRY = os.path.join(ROOT, "telemetry", "mk")
WEIGHTS = os.path.join(ROOT, "mk", "policy_weights.json")
ACTIONS = ["stand", "walk_forward", "walk_backward", "jump_away", "punch", "kick", "block"]


def load(source: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Jev logs a probability per move; the fly logs only the move it made (a one-hot target)."""
    xs, ps, src = [], [], []
    for path in sorted(glob.glob(os.path.join(TELEMETRY, "*.json"))):
        rec = json.load(open(path))
        entries = rec.get("jev_log", []) if source == "jev" else [e for r in rec.get("rounds", []) for e in r.get("fly_log", [])]
        for entry in entries:
            xs.append(entry["x"])
            if "p" in entry:
                ps.append([entry["p"].get(a, 0.0) for a in ACTIONS])
            else:
                ps.append([1.0 if a == entry["a"] else 0.0 for a in ACTIONS])
            src.append(rec["id"])
    if not xs:
        raise SystemExit(f"no logged {source} decisions: play a match with ?log={source} first")
    return np.array(xs, np.float64), np.array(ps, np.float64), src


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train(X: np.ndarray, P: np.ndarray, epochs: int = 4000, lr: float = 0.5, l2: float = 1e-4) -> np.ndarray:
    W = np.zeros((X.shape[1], len(ACTIONS)))
    m, v, t = np.zeros_like(W), np.zeros_like(W), 0
    for _ in range(epochs):  # Adam on the soft-target cross-entropy
        grad = X.T @ (softmax(X @ W) - P) / len(X) + l2 * W
        t += 1
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        W -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W


def report(name: str, W: np.ndarray, X: np.ndarray, P: np.ndarray) -> None:
    Q = softmax(X @ W)
    agree = (Q.argmax(1) == P.argmax(1)).mean()
    kl = float((P * np.log((P + 1e-9) / (Q + 1e-9))).sum(1).mean())
    conf = P.max(1) - np.sort(P, axis=1)[:, -2]
    sure = conf > 0.3
    print(f"  {name:10s} n={len(X):5d}  top-1 agreement {agree:5.1%}  KL {kl:.3f}"
          f"  on Jev's confident calls {(Q.argmax(1) == P.argmax(1))[sure].mean():5.1%} (n={sure.sum()})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="evaluate the saved weights instead of training")
    ap.add_argument("--source", default="jev", choices=["jev", "fly"], help="whose decisions to fit")
    ap.add_argument("--rules", action="store_true", help="print the strongest feature per move")
    ap.add_argument("--test-frac", type=float, default=0.25, help="last fraction of each match held out")
    args = ap.parse_args()

    X, P, src = load(args.source)
    matches = sorted(set(src))
    train_idx, test_idx = [], []
    for match in matches:  # temporal split inside each match
        idx = [i for i, s in enumerate(src) if s == match]
        cut = int(len(idx) * (1 - args.test_frac))
        train_idx += idx[:cut]
        test_idx += idx[cut:]
    Xtr, Ptr, Xte, Pte = X[train_idx], P[train_idx], X[test_idx], P[test_idx]
    print(f"{len(X)} decisions from {len(matches)} match(es): {len(Xtr)} train / {len(Xte)} held out")

    if args.report:
        W = np.array(json.load(open(WEIGHTS))["w"]).reshape(len(ACTIONS), X.shape[1]).T
    else:
        W = train(Xtr, Ptr)

    # baseline: always answer with the commonest move
    base = np.zeros_like(W)
    base[0, int(np.bincount(Ptr.argmax(1), minlength=len(ACTIONS)).argmax())] = 10
    report("baseline", base, Xte, Pte)
    report("train", W, Xtr, Ptr)
    report("held out", W, Xte, Pte)

    if args.rules:
        names = json.load(open(os.path.join(ROOT, "mk", "feature_names.json")))
        share = P.argmax(1)
        print(f"\n  what {args.source} does, as rules (strongest features per move):")
        for a, action in enumerate(ACTIONS):
            used = (share == a).mean()
            if used < 0.01:
                continue
            top = np.argsort(-np.abs(W[1:, a]))[:3] + 1  # skip the bias
            print(f"    {action:14s} {used:5.1%} of moves   " + ", ".join(f"{names[i]} {W[i, a]:+.2f}" for i in top))

    if args.report or args.source == "fly":
        return

    if True:  # only a policy fitted to Jev is written out for the page to play with
        json.dump({"actions": ACTIONS, "features": X.shape[1], "decisions": len(X),
                   "w": [round(v, 5) for v in W.T.flatten()]}, open(WEIGHTS, "w"))
        print(f"wrote {WEIGHTS} ({W.size} weights)")
        strongest = np.argsort(-np.abs(W), axis=None)[:8]
        names = json.load(open(os.path.join(ROOT, "mk", "feature_names.json")))
        print("  strongest weights:", ", ".join(
            f"{names[i // len(ACTIONS)]}->{ACTIONS[i % len(ACTIONS)]} {W.flatten()[i]:+.2f}" for i in strongest))


if __name__ == "__main__":
    main()
