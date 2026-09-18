"""Jev (TypeSafe System One) as a fighter.

The browser sends the structured game state from Jev's side of the arena; this
module asks Jev one typed Choice over the same action set the fly brain's motor
readout can produce, and returns the chosen action with its full probability
distribution. The API key stays server-side.

Key lookup: $TYPESAFE_API_KEY, else TYPESAFE_API_KEY in .env next to this file.
"""

from __future__ import annotations

import http.client
import json
import os
import threading
import time

API_HOST = "api.typesafe.ai"
API_PATH = "/v1/systemone"
MODEL = "jev-latest"
ROOT = os.path.dirname(os.path.abspath(__file__))

# Same vocabulary as the fly's motor readout (app/static/mk/fight.js ACTIONS).
ACTIONS = {
    "stand": "Hold still and do nothing this instant",
    "walk_forward": "Step toward the opponent to close the distance",
    "walk_backward": "Step away from the opponent to open distance",
    "jump_away": "Leap backward away from the opponent; evades attacks but deals no damage",
    "punch": "Punch: 8 damage; only lands if the opponent is in attack range when the fist extends",
    "kick": "High kick: 10 damage; slightly slower than a punch; only lands if the opponent is in attack range",
    "block": "Guard: any hit taken while guarding does only 20% damage; deals no damage",
}

QUESTION = {
    "type": "choice",
    "instructions": {
        "task": "You control `you` in a one-on-one 2D fighting game, deciding the next action roughly ten times a second. Pick the action that best helps `you` win right now.",
        "win_condition": "Reduce the opponent's life to 0 before yours reaches 0. If `time_left_s` runs out, the fighter with more life wins.",
        "mechanics": "Attacks only land when `in_attack_range` is true at the moment of impact. While `you.busy` is true, a new action is ignored until the current move finishes. An opponent who is `attacking` while in range will hit you unless you block or get away.",
        "timing": "When `lead_ms` is present the state has been projected that far ahead, to the moment your action takes effect, so decide for that moment rather than for now. `you.free_in_ms` is how much longer you stay committed to your current move after that, `opponent.still_committed_when_you_act` says whether the opponent is still locked in the action shown, and `opponent.action` is \"unknown\" when its current move will have finished by then.",
    },
    "criteria": ACTIONS,
}

_local = threading.local()
_key: str | None = None


def _api_key() -> str:
    global _key
    if _key:
        return _key
    _key = os.environ.get("TYPESAFE_API_KEY")
    if not _key:
        env = os.path.join(ROOT, ".env")
        if os.path.exists(env):
            for line in open(env):
                k, _, v = line.strip().partition("=")
                if k.strip() == "TYPESAFE_API_KEY":
                    _key = v.strip().strip("'\"")
    if not _key:
        raise RuntimeError("TYPESAFE_API_KEY not set (env or .env); ?jev=local and ?jev=rules need no key")
    return _key


def _conn() -> http.client.HTTPSConnection:
    c = getattr(_local, "conn", None)
    if c is None:
        c = http.client.HTTPSConnection(API_HOST, timeout=10)
        _local.conn = c
    return c


def choose(state: dict) -> dict:
    """Ask Jev for one action. Returns {action, probabilities, confidence, latency_ms, usage}."""
    body = json.dumps({"model": MODEL, "state": state, "questions": {"action": QUESTION}})
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    t0 = time.perf_counter()
    for attempt in range(2):  # one retry on a dropped keep-alive connection
        try:
            c = _conn()
            c.request("POST", API_PATH, body=body, headers=headers)
            r = c.getresponse()
            raw = r.read()
            break
        except (http.client.HTTPException, OSError):
            _local.conn = None
            if attempt:
                raise
    if r.status != 200:
        raise RuntimeError(f"TypeSafe {r.status}: {raw[:300]!r}")
    data = json.loads(raw)
    ans = data["answers"]["action"]
    return {
        "action": ans["choice"],
        "probabilities": ans["probabilities"],
        "confidence": ans.get("confidence"),
        "latency_ms": round((time.perf_counter() - t0) * 1000),
        "model": data.get("model"),
        "usage": data.get("usage"),
    }
