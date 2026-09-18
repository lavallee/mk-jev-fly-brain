"""Play matches headlessly, so a match-up can be filled in without sitting at the arcade.

Drives the real page in headless Chromium — same worker, same simulation, same telemetry — and
waits for the match (or the whole `?evolve=N` series) to finish before starting the next.

    python -m scripts.run_matches "jev=rules&learn=1&plastic=500&baseline=0.02"
    python -m scripts.run_matches --repeat 2 "jev=local&learn=1&plastic=500"
    python -m scripts.run_matches --plan plan.txt        # one query string per line, '#' comments

Rounds run in real time, so budget ~2 minutes per 3-round match at `seconds=40`.
"""

from __future__ import annotations

import argparse
import sys
import time

from urllib.parse import parse_qsl, urlencode

from playwright.sync_api import sync_playwright

DEFAULTS = "auto=1&rounds=3&seconds=40"


def run(page, base: str, query: str, timeout_s: float) -> dict:
    merged = dict(parse_qsl(DEFAULTS)) | dict(parse_qsl(query))  # the run's own params win
    url = f"{base}/mk/?{urlencode(merged)}"
    print(f"  {url}", flush=True)
    page.goto(url, wait_until="domcontentloaded")
    page.evaluate("try { localStorage.setItem('mk.muted', '1') } catch {}")
    series = "evolve=" in query
    flag = "seriesDone" if series else "done"
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_s:
        state = page.evaluate(
            "() => ({ done: !!(window.__mk && window.__mk.%s), score: window.__mk && window.__mk.score,"
            " gen: window.__mk && window.__mk.generation, id: window.__mk && window.__mk.matchId })" % flag)
        note = f"gen {state['gen']} " if state.get("gen") else ""
        if state.get("score") and f"{note}{state['score']}" != last:
            last = f"{note}{state['score']}"
            print(f"    {int(time.time() - t0):4d}s  {last}", flush=True)
        if state["done"]:
            return state
        page.wait_for_timeout(4000)
    print("    timed out", flush=True)
    return {"done": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="*", help="URL query strings, one per match-up")
    ap.add_argument("--plan", help="file with one query string per line")
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=2400, help="seconds per run")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    queries = list(args.queries)
    if args.plan:
        queries += [ln.strip() for ln in open(args.plan) if ln.strip() and not ln.startswith("#")]
    if not queries:
        raise SystemExit("nothing to run")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed, args=["--mute-audio", "--autoplay-policy=no-user-gesture-required"])
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        page = ctx.new_page()
        page.on("pageerror", lambda e: print(f"    page error: {e}", file=sys.stderr, flush=True))
        for i, q in enumerate(queries, 1):
            for rep in range(args.repeat):
                print(f"[{i}/{len(queries)}{f' rep {rep + 1}' if args.repeat > 1 else ''}]", flush=True)
                run(page, args.base, q, args.timeout)
        browser.close()


if __name__ == "__main__":
    main()
