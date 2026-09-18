"""Model Kombat: a maleCNS fly brain fights Jev in mk.js.

    pip install -r requirements.txt
    python scripts/fetch_game_assets.py      # mk.js sprites (not redistributable here)
    echo TYPESAFE_API_KEY=... > .env         # only needed for ?jev=api and ?jev=hybrid
    python server.py                         # -> http://127.0.0.1:5000/mk/

Serves the page, proxies TypeSafe so the key stays off the client, and stores one JSON file per
match in telemetry/mk/ (which is also where saved fly brains live).
"""

from __future__ import annotations

import json
import os
import re
import time

from flask import Flask, jsonify, request, send_from_directory

import jev

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MK_DIR = os.path.join(BASE_DIR, "mk")
TELEMETRY = os.path.join(BASE_DIR, "telemetry", "mk")
MATCH_ID = re.compile(r"^[0-9A-Za-z_-]{8,64}$")

app = Flask(__name__)


@app.route("/")
@app.route("/mk")
@app.route("/mk/")
def index():
    return send_from_directory(MK_DIR, "index.html")


@app.route("/mk/<path:filename>")
def asset(filename):
    return send_from_directory(MK_DIR, filename)


@app.route("/api/jev/move", methods=["POST"])
def jev_move():
    """Jev picks the next fighter action from the structured game state."""
    try:
        return jsonify(jev.choose(request.get_json(force=True)))
    except Exception as e:  # surfaced in the Jev panel's error count, not as a 500 page
        return jsonify({"error": str(e)}), 502


def _brain_summary(rec: dict) -> dict | None:
    learned = rec.get("learned") or {}
    if not learned.get("weights_b64"):
        return None
    rounds = rec.get("rounds") or []
    return {
        "id": rec["id"], "started_at": rec.get("started_at"), "complete": rec.get("complete", False),
        "score": rec.get("score") or {}, "rounds_played": len(rounds),
        "damage": {w: sum((r.get("damage") or {}).get(w, 0) for r in rounds) for w in ("fly", "jev")},
        "strength": learned.get("strength"), "fingerprint": (rec.get("circuit") or {}).get("fingerprint"),
        "brain_start": (rec.get("config") or {}).get("brain_start", "wired"),
    }


@app.route("/api/mk/telemetry", methods=["POST"])
def telemetry():
    if request.content_length and request.content_length > 25 * 1024 * 1024:
        return jsonify({"error": "telemetry too large"}), 413
    rec = request.get_json(force=True)
    mid = str(rec.get("id", ""))
    if not MATCH_ID.match(mid):
        return jsonify({"error": "bad match id"}), 400
    rec["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    os.makedirs(TELEMETRY, exist_ok=True)
    path = os.path.join(TELEMETRY, f"{mid}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(path + ".tmp", path)
    return jsonify({"ok": True, "id": mid})


@app.route("/api/mk/brains")
def brains():
    """Saved learned brains for this circuit, best first: match margin, win share, damage margin."""
    fingerprint = request.args.get("fingerprint")
    out = []
    for name in os.listdir(TELEMETRY) if os.path.isdir(TELEMETRY) else []:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(TELEMETRY, name)) as f:
                summary = _brain_summary(json.load(f))
        except (OSError, ValueError, KeyError):
            continue
        if summary and summary["complete"] and (not fingerprint or summary["fingerprint"] == fingerprint):
            out.append(summary)

    def rank(b):
        s = b["score"]
        played = max(1, s.get("fly", 0) + s.get("jev", 0) + s.get("draw", 0))
        return (s.get("fly", 0) - s.get("jev", 0), s.get("fly", 0) / played, b["damage"]["fly"] - b["damage"]["jev"])

    out.sort(key=rank, reverse=True)
    return jsonify({"brains": out[:10]})


@app.route("/api/mk/brain/<mid>")
def brain(mid):
    if not MATCH_ID.match(mid):
        return jsonify({"error": "bad match id"}), 400
    path = os.path.join(TELEMETRY, f"{mid}.json")
    if not os.path.exists(path):
        return jsonify({"error": "no such brain"}), 404
    with open(path) as f:
        rec = json.load(f)
    learned = rec.get("learned") or {}
    return jsonify({"id": mid, "fingerprint": (rec.get("circuit") or {}).get("fingerprint"),
                    "weights_b64": learned.get("weights_b64"), "strength": learned.get("strength"),
                    "thresholds": learned.get("thresholds"), "score": rec.get("score")})


if __name__ == "__main__":
    host = os.environ.get("MK_HOST", "127.0.0.1")
    port = int(os.environ.get("MK_PORT", "5000"))
    print(f"Model Kombat on http://{host}:{port}/mk/")
    app.run(host=host, port=port, debug=False)
