"""Bake the Model Kombat announcer with ElevenLabs.

The fighters' names are fixed, so every line is generated once and served as a static file
(mk/voice/<set>/<key>.mp3). The page plays them through its own Web Audio graph, so a
tab-audio capture records the announcer along with the music and effects. Pick a set in the page
with ?announcer=<set>.

    python -m scripts.bake_announcer                              # daniel, announcer mode
    python -m scripts.bake_announcer --voice callum --style classic --no-arena
    python -m scripts.bake_announcer --only fight,title --force
    python -m scripts.bake_announcer --voices --probe             # which voices this key/plan can use

Announcer mode = shouted, all-caps lines with dramatic beats ("ROUND... ONE!"), the most
expressive stability, internal pauses capped so beats stay snappy, and an arena pass: a slight
pitch drop for size, low-end and presence EQ, compression and a short stadium slapback.

Reads ELEVENLABS_API_KEY from the environment or .env at the repo root. Needs ffmpeg (with rubberband).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_ROOT = os.path.join(ROOT, "mk", "voice")
MODEL = "eleven_v3"  # performs [shouting]/[dramatic] tags instead of reading them aloud
SR = 44100

VOICES = {  # ElevenLabs premade voices auditioned for the part
    "daniel": "onwK4e9ZLuTAKqWW03F9",  # steady British broadcaster (default)
    "callum": "N2lVS1w4EtoT3dr4eOWO",  # gravelly, loudest and punchiest "Fight!"
    "adam": "pNInz6obpgDQGcFmaJgB",    # dominant, brash
    "brian": "nPczCjzI2devNBz1zQrb",   # deepest, calmer
    "harry": "SOYHLrjzK2X1ezoPC6cr",   # fierce warrior, drawn-out delivery
}

ROUND_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
STYLES = {
    # the first bake: tags, sentence case
    "classic": {
        "title": "[shouting] Model Kombat!",
        **{f"round{i + 1}": f"[dramatic] Round {w}!" for i, w in enumerate(ROUND_WORDS)},
        "fight": "[shouting] Fight!",
        "time": "[shouting] Time!",
        "fly_wins": "[dramatic] Fly brain wins!",
        "jev_wins": "[dramatic] Jev wins!",
        "draw": "[dramatic] Draw!",
        "flawless": "[shouting] Flawless victory!",
        "fly_match": "[dramatic] Fly brain wins the match!",
        "jev_match": "[dramatic] Jev wins the match!",
        "match_draw": "[dramatic] The match is a draw!",
    },
    # announcer mode: shouted caps; ellipses mark the dramatic beat (capped in post).
    # Names stay out of caps: "JEV" gets spelled out as J-E-V; "Jevv" is read as one syllable.
    "announcer": {
        "title": "[shouting] MODEL... KOMBAT!",
        **{f"round{i + 1}": f"[shouting] ROUND... {w.upper()}!" for i, w in enumerate(ROUND_WORDS)},
        "fight": "[shouting] FIGHT!",
        "time": "[shouting] TIME!",
        "fly_wins": "[shouting] FLY BRAIN... WINS!",
        "jev_wins": "[shouting] Jevv... WINS!",
        "draw": "[shouting] DRAW!",
        "flawless": "[shouting] FLAWLESS... VICTORY!",
        "fly_match": "[shouting] FLY BRAIN... WINS THE MATCH!",
        "jev_match": "[shouting] Jevv... WINS THE MATCH!",
        "match_draw": "[shouting] THE MATCH... IS A DRAW!",
    },
}

MAX_PAUSE = 0.35      # s: longest silence kept inside a line
SILENCE_DB = -45      # frames quieter than this count as silence
ARENA = ",".join([
    "rubberband=pitch=0.92",                              # ~1.4 semitones down: a bigger voice, same energy
    "highpass=f=70",
    "lowshelf=g=4:f=160",                                  # chest
    "equalizer=f=2800:t=q:w=1.0:g=3",                      # presence / cut-through
    "acompressor=threshold=-20dB:ratio=4:attack=4:release=150:makeup=3",
    "aecho=0.9:0.45:55|110:0.22|0.1",                      # short stadium slapback
])


def _api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY")
    env = os.path.join(ROOT, ".env")
    if not key and os.path.exists(env):
        for line in open(env):
            k, _, v = line.strip().partition("=")
            if k.strip() == "ELEVENLABS_API_KEY":
                key = v.strip().strip("'\"")
    if not key:
        raise SystemExit("ELEVENLABS_API_KEY not set (env or .env)")
    return key


def synthesize(text: str, voice_id: str, key: str, stability: float | None = None) -> bytes:
    """stability for eleven_v3: 0.0 Creative (most expressive), 0.5 Natural, 1.0 Robust."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"
    payload = {"text": text, "model_id": MODEL}
    if stability is not None:
        payload["voice_settings"] = {"stability": stability}
    body = json.dumps(payload).encode()
    for attempt in range(4):
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"xi-api-key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise SystemExit(f"ElevenLabs {e.code} for {text!r}: {e.read()[:300]!r}")
    raise AssertionError("unreachable")


def list_voices(key: str, probe: bool) -> None:
    """Print the account's voices by category. Premade voices always work over the API. On a free
    plan, library voices ("professional") return 402 paid_plan_required and cloned voices return
    401 subscription_required. --probe confirms the non-premade ones with a 3-character generation each."""
    req = urllib.request.Request("https://api.elevenlabs.io/v2/voices?page_size=100", headers={"xi-api-key": key})
    voices = json.load(urllib.request.urlopen(req, timeout=30))["voices"]
    for v in sorted(voices, key=lambda v: (v.get("category") != "premade", v.get("category") or "", v["name"])):
        status = ""
        if probe and v.get("category") != "premade":
            body = json.dumps({"text": "Hi.", "model_id": "eleven_multilingual_v2"}).encode()
            r = urllib.request.Request(f"https://api.elevenlabs.io/v1/text-to-speech/{v['voice_id']}", data=body, method="POST",
                                       headers={"xi-api-key": key, "Content-Type": "application/json"})
            try:
                urllib.request.urlopen(r, timeout=60).read()
                status = "usable"
            except urllib.error.HTTPError as e:
                detail = json.loads(e.read() or b"{}").get("detail", {})
                status = f"blocked ({e.code} {detail.get('code', '')})"
        elif v.get("category") == "premade":
            status = "usable"
        print(f"  {v['voice_id']}  {v.get('category', ''):12s} {status:32s} {v['name']}")


def decode(raw: bytes) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         input=raw, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def tighten(x: np.ndarray) -> np.ndarray:
    """Trim leading/trailing silence and shorten every internal pause to MAX_PAUSE (10 ms crossfades)."""
    hop = int(0.01 * SR)
    frames = len(x) // hop
    rms = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2)) for i in range(frames)])
    loud = 20 * np.log10(rms + 1e-9) > SILENCE_DB
    if not loud.any():
        return x
    first, last = int(np.argmax(loud)), frames - int(np.argmax(loud[::-1]))
    x = x[max(0, (first - 2) * hop):min(len(x), (last + 6) * hop)]
    loud = loud[max(0, first - 2):last + 6]

    keep = int(MAX_PAUSE / 0.01)
    fade = hop
    out, cursor, i = [], 0, 0
    while i < len(loud):
        if loud[i]:
            i += 1
            continue
        j = i
        while j < len(loud) and not loud[j]:
            j += 1
        if j - i > keep and j < len(loud):
            cut_start = (i + keep // 2) * hop
            cut_end = (j - keep // 2) * hop
            head = x[cursor:cut_start].copy()
            tail_start = x[cut_end:cut_end + fade]
            ramp = np.linspace(0, 1, len(tail_start), dtype=np.float32)
            head[-len(tail_start):] = head[-len(tail_start):] * (1 - ramp) + tail_start * ramp
            out.append(head)
            cursor = cut_end + fade
        i = j
    out.append(x[cursor:])
    return np.concatenate(out)


def polish(raw: bytes, dest: str, arena: bool) -> None:
    x = tighten(decode(raw))
    chain = (ARENA + "," if arena else "") + "loudnorm=I=-15:TP=-1.5:LRA=11"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    "-af", chain, "-ar", str(SR), "-ac", "1", "-b:a", "128k", dest],
                   input=x.astype(np.float32).tobytes(), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="daniel", choices=sorted(VOICES))
    ap.add_argument("--style", default="announcer", choices=sorted(STYLES))
    ap.add_argument("--stability", type=float, default=0.0, help="eleven_v3: 0.0 creative, 0.5 natural, 1.0 robust")
    ap.add_argument("--no-arena", dest="arena", action="store_false", help="skip the arena processing pass")
    ap.add_argument("--set", default=None, help="output set name (default: the voice name)")
    ap.add_argument("--out", default=None, help="output directory (overrides --set)")
    ap.add_argument("--only", default="", help="comma-separated line keys")
    ap.add_argument("--force", action="store_true", help="regenerate lines that already exist")
    ap.add_argument("--voices", action="store_true", help="list the account's voices and exit")
    ap.add_argument("--probe", action="store_true", help="with --voices: test-generate with each non-premade voice")
    args = ap.parse_args()

    key = _api_key()
    if args.voices:
        list_voices(key, args.probe)
        return

    lines = STYLES[args.style]
    out = args.out or os.path.join(VOICE_ROOT, args.set or args.voice)
    os.makedirs(out, exist_ok=True)
    wanted = [k for k in args.only.split(",") if k] or list(lines)
    unknown = set(wanted) - set(lines)
    if unknown:
        raise SystemExit(f"unknown line keys: {sorted(unknown)}")

    manifest_path = os.path.join(out, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {"lines": {}}
    for k in wanted:
        dest = os.path.join(out, f"{k}.mp3")
        if os.path.exists(dest) and not args.force:
            print(f"  skip  {k}")
            continue
        polish(synthesize(lines[k], VOICES[args.voice], key, args.stability), dest, args.arena)
        manifest["lines"][k] = {"text": lines[k]}
        print(f"  baked {k:11s} {os.path.getsize(dest) / 1024:5.1f} KB  {lines[k]}")

    manifest.update({"voice": args.voice, "voice_id": VOICES[args.voice], "model": MODEL, "style": args.style,
                     "stability": args.stability, "arena": args.arena, "max_pause_s": MAX_PAUSE})
    manifest["lines"] = {k: manifest["lines"][k] for k in lines if k in manifest["lines"]}
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"wrote {manifest_path} ({len(manifest['lines'])} lines)")


if __name__ == "__main__":
    main()
