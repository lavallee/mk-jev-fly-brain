"""Cut the downloaded sound-effect packs into one-shots for Model Kombat.

Sources live in mk/effects/ (as downloaded); this writes tight, level-matched mono
one-shots to mk/effects/bank/ with a manifest the page loads. Re-runnable: the bank is
rebuilt from the sources every time, so edits to the sources or to the settings below just need

    python -m scripts.slice_effects

Each one-shot starts 5 ms before its transient (so hits land on the frame they happen), ends when
the tail falls away, and is peak-normalised. Credits: see mk/effects-CREDITS.md.
"""

from __future__ import annotations

import json
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "mk", "effects")
BANK = os.path.join(SRC, "bank")
SR = 44100

# category -> source file. Packs are split into one-shots, single sounds pass through.
SOURCES = {
    "punch": "storegraphic-punch-and-hits-310521.mp3",
    "kick": "freesound_community-kick-sounds-38706.mp3",
    "kick_bright": "khoamthanh-kick-bright-medium-504170.mp3",
    "grunt_ough": "freesound_community-ough-47202.mp3",
    "grunt_umph": "freesound_community-umph-47201.mp3",
}
ON_DB = -38        # a burst starts when it rises above this
OFF_DB = -50       # ...and ends when it falls below this
MIN_LEN = 0.06     # s: ignore clicks
GAP = 0.15         # s: silence that separates two one-shots
PEAK = 0.89        # ~ -1 dBFS
MIN_BODY = 0.05    # s: an impact needs some body, not just a click
MIN_RMS_DB = -23   # ...and some weight; weak fragments of a pack are dropped
LATE_PEAK = 0.08   # s: a clip that peaks later than this is a wind-up; re-cut it to start at the impact


def decode(path: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def bursts(x: np.ndarray) -> list[tuple[int, int]]:
    hop = int(0.005 * SR)
    db = 20 * np.log10(np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, len(x) - hop, hop)]) + 1e-9)
    out, start, quiet = [], None, 0
    for i, d in enumerate(db):
        if d > ON_DB and start is None:
            start = i
            quiet = 0
        elif start is not None:
            quiet = quiet + 1 if d < OFF_DB else 0
            if quiet * 0.005 > GAP:
                out.append((start, i - quiet))
                start = None
    if start is not None:
        out.append((start, len(db)))
    spans = [(max(0, int((a * 0.005 - 0.005) * SR)), min(len(x), int((b * 0.005 + 0.12) * SR))) for a, b in out]
    return [(a, b) for a, b in spans if (b - a) / SR >= MIN_LEN]


def write(x: np.ndarray, dest: str) -> float:
    x = x * (PEAK / max(1e-6, float(np.abs(x).max())))
    fade = min(int(0.01 * SR), len(x) // 4)
    x[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)  # no click at the tail
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    "-b:a", "128k", dest], input=x.astype(np.float32).tobytes(), check=True)
    return len(x) / SR


def main() -> None:
    os.makedirs(BANK, exist_ok=True)
    for name in os.listdir(BANK):
        if name.endswith((".mp3", ".json")):
            os.remove(os.path.join(BANK, name))

    manifest: dict[str, list[dict]] = {}
    for category, filename in SOURCES.items():
        path = os.path.join(SRC, filename)
        if not os.path.exists(path):
            print(f"  missing {filename}")
            continue
        x = decode(path)
        spans = bursts(x)
        kind = category.split("_")[0]
        for i, (a, b) in enumerate(spans, 1):
            clip = x[a:b].copy()
            if kind in ("punch", "kick"):
                peak = int(np.argmax(np.abs(clip)))
                if peak / SR > LATE_PEAK:  # drop the wind-up so the hit lands on the frame it happens
                    clip = clip[max(0, peak - int(0.005 * SR)):].copy()
                body = np.flatnonzero(np.abs(clip) > 0.05)
                rms_db = 20 * np.log10(np.sqrt(np.mean(clip ** 2)) + 1e-9)
                if not len(body) or (body[-1] - body[0]) / SR < MIN_BODY or rms_db < MIN_RMS_DB:
                    print(f"  drop  {category}_{i:02d}  (thin: {rms_db:.1f} dB)")
                    continue
            key = f"{category}_{i:02d}" if len(spans) > 1 else category
            seconds = write(clip, os.path.join(BANK, f"{key}.mp3"))
            manifest.setdefault(kind, []).append({"key": key, "seconds": round(seconds, 3), "source": filename})
            print(f"  {key:16s} {seconds:5.2f}s  from {filename} @ {a / SR:5.2f}s")

    json.dump({"categories": manifest}, open(os.path.join(BANK, "manifest.json"), "w"), indent=2)
    total = sum(len(v) for v in manifest.values())
    print(f"wrote {BANK} ({total} one-shots: " + ", ".join(f"{k} {len(v)}" for k, v in manifest.items()) + ")")


if __name__ == "__main__":
    main()
