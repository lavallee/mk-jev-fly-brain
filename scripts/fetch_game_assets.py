"""Fetch the mk.js sprites and arena backgrounds this game draws with.

They are not in this repository. mk.js is MIT-licensed and its source file is vendored here, but
its `game/images/` folder contains Mortal Kombat character and arena art, which belongs to its
rights holders and is not ours to redistribute. This script downloads it from the upstream mk.js
repository into mk/images/ (gitignored) for local use.

    python scripts/fetch_game_assets.py

Without it the page loads but the fighters never appear ("LOADING FIGHTERS" stays up).
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "mk", "images")
TARBALL = "https://codeload.github.com/mgechev/mk.js/tar.gz/refs/heads/master"
INSIDE = "game/images/"


def main() -> None:
    if os.path.isdir(DEST) and os.listdir(DEST):
        print(f"{DEST} already populated; delete it to re-fetch")
        return
    print(f"downloading mk.js from {TARBALL} ...")
    with urllib.request.urlopen(TARBALL, timeout=120) as r:
        blob = r.read()
    count = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = member.name.split("/", 1)
            if len(parts) != 2 or not parts[1].startswith(INSIDE) or not member.isfile():
                continue
            rel = parts[1][len(INSIDE):]
            if ".." in rel or rel.startswith("/"):
                continue
            out = os.path.join(DEST, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with tar.extractfile(member) as src, open(out, "wb") as dst:
                dst.write(src.read())
            count += 1
    if not count:
        sys.exit("no images found in the tarball; has the upstream layout changed?")
    print(f"wrote {count} files to {DEST}")
    print("Sprites and arenas are Mortal Kombat assets (Midway/NetherRealm), used here for local play.")


if __name__ == "__main__":
    main()
