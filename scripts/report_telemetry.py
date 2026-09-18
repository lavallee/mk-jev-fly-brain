"""Turn the match telemetry into a single self-contained HTML report.

Reads every match JSON under the telemetry directory (including archived runs) and writes one page
with the learning curves, the rewired control, what the fly learned, the outcomes and what each
side actually does. No dependencies, no build step, no network: inline SVG and a little CSS.

    python -m scripts.report_telemetry                          # -> docs/index.html
    python -m scripts.report_telemetry --telemetry ../other/telemetry --out /tmp/report.html

Charts follow the usual rules: categorical hues in fixed order, one axis, a legend and direct
labels for every series, a table view behind each chart, and a dark mode stepped for its own
surface rather than flipped.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import time
from collections import defaultdict
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPPONENTS = {  # internal name -> (label, categorical slot)
    "rules": ("Rule bot", 1),
    "api": ("Jev API", 2),
    "local": ("Local policy", 3),
    "hybrid": ("Hybrid", 4),
}
POOLS = {"wing": ("Wings (guard)", 1), "jump": ("TTMn (escape jump)", 2), "kick": ("Hind legs (kick)", 3)}
ACTIONS = ["stand", "walk_forward", "walk_backward", "jump_away", "punch", "kick", "block"]

# There is no one fly and no one Jev. Each side fields several competitors, and a match is only
# ever fly-vs-Jev, so the league is bipartite: the two columns of the standings never play
# each other, and each row's record is against whichever opponents it happened to be entered against.
SIDE_NOTE = {
    "rules": "five lines of hand-written positional rules",
    "api": "TypeSafe System One, reading the fight as JSON ~5 times a second",
    "local": "168 weights distilled from Jev's probabilities, microseconds per decision",
    "hybrid": "the local policy moving at fly speed, Jev correcting and teaching it live",
    "Wired (no learning)": "the connectome as mapped, plasticity off",
    "In-match learning": "dopamine learning inside a single match, starting from the wired brain",
    "In-match learning + baseline": "the same, comparing each move against its own recent average",
    "Curriculum": "each match inherits the best brain saved so far",
    "Curriculum + thresholds": "a curriculum, plus thresholds the fly can learn to raise and lower",
    "Reloaded brain": "a curriculum-trained brain, entered against a harder opponent",
    "Rewired control": "connections shuffled; neurons, degrees, signs and weights kept",
}
FLY_ORDER = ["Wired (no learning)", "In-match learning", "In-match learning + baseline",
             "Curriculum", "Curriculum + thresholds", "Reloaded brain", "Rewired control"]


def fly_variant(cfg: dict) -> str:
    """Which fly this is. The line between variants is where the learning setup differs."""
    if cfg.get("circuit") == "rewired":
        return "Rewired control"
    if not cfg.get("learn"):
        return "Wired (no learning)"
    if cfg.get("lineage") or cfg.get("generation"):
        return "Curriculum + thresholds" if cfg.get("thresh_lr") else "Curriculum"
    if cfg.get("brain_start") not in (None, "wired"):
        return "Reloaded brain"
    return "In-match learning" + (" + baseline" if cfg.get("baseline") else "")


def records(matches: list[dict]) -> dict:
    """Round records per (fly variant, Jev variant), plus each competitor's damage per second."""
    cells: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"matches": 0, "w": 0, "d": 0, "l": 0, "fly_dps": [], "jev_dps": []})
    for m in matches:
        cfg = m.get("config") or {}
        cell = cells[(fly_variant(cfg), cfg.get("jev_mode") or "api")]
        cell["matches"] += 1
        for r in m["rounds"]:
            cell["w" if r["winner"] == "fly" else "d" if r["winner"] == "draw" else "l"] += 1
            cell["fly_dps"].append(r["damage"]["fly"] / max(0.1, r["seconds"]))
            cell["jev_dps"].append(r["damage"]["jev"] / max(0.1, r["seconds"]))
    return cells


# ---------- data ----------

def load(telemetry: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(telemetry, "**", "*.json"), recursive=True)):
        try:
            rec = json.load(open(path))
        except (OSError, ValueError):
            continue
        if rec.get("complete") and rec.get("rounds"):
            out.append(rec)
    return out


def per_second(rounds: list[dict], pick) -> float:
    return mean(pick(r) / max(0.1, r["seconds"]) for r in rounds)


def generation_series(matches: list[dict], circuit: str, value, variant: str | None = None) -> dict[str, dict[int, float]]:
    """{opponent: {generation: mean value}} for one circuit, and one fly variant if named.

    Arms that differ in their learning setup are different competitors, so averaging them into one
    curve would hide the thing the curve is for.
    """
    buckets: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for m in matches:
        cfg = m.get("config") or {}
        gen = cfg.get("generation")
        this_circuit = "rewired" if cfg.get("circuit") == "rewired" else "real"
        if not gen or this_circuit != circuit or (variant and fly_variant(cfg) != variant):
            continue
        v = value(m)
        if v is not None:
            buckets[cfg.get("jev_mode", "api")][gen].append(v)
    return {k: {g: mean(vs) for g, vs in sorted(gens.items())} for k, gens in buckets.items()}


# ---------- svg ----------

def line_chart(series: list[dict], *, width=720, height=300, x_label="", y_label="", y_zero=True) -> str:
    """series: [{name, slot, points: [(x, y)], format}] - one axis, legend + direct end labels."""
    pad = {"l": 56, "r": 128, "t": 36, "b": 40}
    xs = [x for s in series for x, _ in s["points"]]
    ys = [y for s in series for _, y in s["points"]]
    if not xs:
        return "<p>no data</p>"
    x0, x1 = min(xs), max(xs)
    y0, y1 = (0 if y_zero else min(ys)), max(ys) * 1.08
    if y1 <= y0:
        y1 = y0 + 1
    px = lambda x: pad["l"] + (x - x0) / max(1e-9, x1 - x0) * (width - pad["l"] - pad["r"])
    py = lambda y: height - pad["b"] - (y - y0) / max(1e-9, y1 - y0) * (height - pad["t"] - pad["b"])

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    ticks = 4
    for i in range(ticks + 1):
        v = y0 + (y1 - y0) * i / ticks
        y = py(v)
        parts.append(f'<line class="grid" x1="{pad["l"]}" x2="{width - pad["r"]}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text class="tick" x="{pad["l"] - 10}" y="{y + 4:.1f}" text-anchor="end">{v:.3g}</text>')
    for x in sorted(set(xs)):
        parts.append(f'<text class="tick" x="{px(x):.1f}" y="{height - pad["b"] + 20}" text-anchor="middle">{x:g}</text>')
    if y_label:
        parts.append(f'<text class="axis-label" x="6" y="13">{html.escape(y_label)}</text>')
    if x_label:
        parts.append(f'<text class="axis-label" x="{width - pad["r"]}" y="{height - 6}" text-anchor="end">{html.escape(x_label)}</text>')

    for s in series:
        pts = s["points"]
        d = " ".join(f'{"M" if i == 0 else "L"}{px(x):.1f},{py(y):.1f}' for i, (x, y) in enumerate(pts))
        parts.append(f'<path class="line" style="stroke:var(--series-{s["slot"]})" d="{d}"/>')
        for x, y in pts:
            parts.append(f'<circle class="dot" style="fill:var(--series-{s["slot"]})" cx="{px(x):.1f}" cy="{py(y):.1f}" r="4">'
                         f'<title>{html.escape(s["name"])} · {x:g}: {y:.2f}</title></circle>')
        lx, ly = pts[-1]
        parts.append(f'<text class="series-label" x="{px(lx) + 12:.1f}" y="{py(ly) + 4:.1f}">{html.escape(s["name"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def stacked_bars(rows: list[dict], *, width=720, bar=26, gap=16) -> str:
    """rows: [{label, parts: [(name, value, css_var)]}] - composition per row, 2px surface gaps."""
    pad_l, pad_r = 132, 96
    total_max = max(sum(v for _, v, _ in r["parts"]) for r in rows) or 1
    height = len(rows) * (bar + gap) + gap
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    for i, row in enumerate(rows):
        y = gap + i * (bar + gap)
        x = pad_l
        total = sum(v for _, v, _ in row["parts"]) or 1
        parts.append(f'<text class="row-label" x="{pad_l - 12}" y="{y + bar * 0.7:.0f}" text-anchor="end">{html.escape(row["label"])}</text>')
        for name, value, var in row["parts"]:
            w = value / total_max * (width - pad_l - pad_r)
            if w <= 0:
                continue
            parts.append(f'<rect class="seg" x="{x:.1f}" y="{y}" width="{max(0, w - 2):.1f}" height="{bar}" rx="3" style="fill:{var}">'
                         f'<title>{html.escape(row["label"])} · {html.escape(name)}: {value:g}</title></rect>')
            if w > 34:
                parts.append(f'<text class="in-bar" x="{x + w / 2 - 1:.1f}" y="{y + bar * 0.68:.0f}" text-anchor="middle">{value:g}</text>')
            x += w
        parts.append(f'<text class="row-total" x="{x + 10:.1f}" y="{y + bar * 0.7:.0f}">{total:g} rounds</text>')
    parts.append("</svg>")
    return "".join(parts)


def grouped_bars(categories: list[str], series: list[dict], *, width=720, height=260) -> str:
    """series: [{name, slot, values}] aligned to categories."""
    pad = {"l": 46, "r": 16, "t": 16, "b": 46}
    top = max((v for s in series for v in s["values"]), default=1) * 1.1 or 1
    inner = (width - pad["l"] - pad["r"]) / max(1, len(categories))
    bw = min(26, (inner - 10) / len(series))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    for i in range(5):
        v = top * i / 4
        y = height - pad["b"] - (v / top) * (height - pad["t"] - pad["b"])
        parts.append(f'<line class="grid" x1="{pad["l"]}" x2="{width - pad["r"]}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text class="tick" x="{pad["l"] - 8}" y="{y + 4:.1f}" text-anchor="end">{v:.0f}%</text>')
    for ci, cat in enumerate(categories):
        cx = pad["l"] + inner * (ci + 0.5)
        for si, s in enumerate(series):
            v = s["values"][ci]
            h = (v / top) * (height - pad["t"] - pad["b"])
            x = cx - (len(series) * bw) / 2 + si * bw
            parts.append(f'<rect class="seg" x="{x:.1f}" y="{height - pad["b"] - h:.1f}" width="{bw - 2:.1f}" height="{h:.1f}" rx="3" '
                         f'style="fill:var(--series-{s["slot"]})"><title>{html.escape(s["name"])} · {html.escape(cat)}: {v:.0f}%</title></rect>')
        parts.append(f'<text class="tick" x="{cx:.1f}" y="{height - pad["b"] + 18:.0f}" text-anchor="middle">{html.escape(cat.replace("_", " "))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def standings_table(cells: dict) -> str:
    """One row per competitor, each record read from that competitor's own side."""
    rows = []
    for side, keys, flip in (("fly", FLY_ORDER, False), ("jev", list(OPPONENTS), True)):
        entries = []
        for key in keys:
            mine = [(k, c) for k, c in cells.items() if k[0 if side == "fly" else 1] == key]
            if not mine:
                continue
            w = sum(c["l" if flip else "w"] for _, c in mine)
            d = sum(c["d"] for _, c in mine)
            l = sum(c["w" if flip else "l"] for _, c in mine)
            dps = [v for _, c in mine for v in c["jev_dps" if flip else "fly_dps"]]
            played = w + d + l
            faced = sorted({k[1] if side == "fly" else k[0] for k, _ in mine},
                            key=lambda o: list(OPPONENTS).index(o) if side == "fly" else FLY_ORDER.index(o))
            entries.append({"name": OPPONENTS[key][0] if flip else key, "key": key, "side": side,
                            "w": w, "d": d, "l": l, "played": played, "matches": sum(c["matches"] for _, c in mine),
                            "rate": w / max(1, played), "dps": mean(dps) if dps else 0,
                            "faced": [OPPONENTS[o][0] if side == "fly" else o for o in faced]})
        rows += sorted(entries, key=lambda e: -e["rate"])

    out = ['<table class="standings"><thead><tr><th>Competitor</th><th>Rounds</th><th>W</th><th>D</th>'
           '<th>L</th><th>Win rate</th><th>Damage&nbsp;/&nbsp;s</th><th class="recordcol">Record</th></tr></thead><tbody>']
    side_label = {"fly": "fly brain", "jev": "opponent"}
    for r in rows:
        bar = "".join(
            f'<i style="width:{100 * v / max(1, r["played"]):.1f}%;background:{c}"></i>'
            for v, c in ((r["w"], "var(--pole-cool)"), (r["d"], "var(--neutral)"), (r["l"], "var(--pole-warm)")))
        out.append(
            f'<tr class="side-{r["side"]}"><td><b>{html.escape(r["name"])}</b> <span class="tag">{side_label[r["side"]]}</span>'
            f'<span class="desc">{html.escape(SIDE_NOTE.get(r["key"], ""))}</span>'
            f'<span class="sched">played {html.escape(", ".join(r["faced"]))}</span></td>'
            f'<td>{r["played"]}</td><td>{r["w"]}</td><td>{r["d"]}</td><td>{r["l"]}</td>'
            f'<td>{r["rate"]:.0%}</td><td>{r["dps"]:.2f}</td>'
            f'<td class="recordcol"><span class="recordbar" title="{r["w"]} won, {r["d"]} drawn, {r["l"]} lost '
            f'across {r["matches"]} matches">{bar}</span></td></tr>')
    out.append("</tbody></table>")
    return "".join(out)


def head_to_head(cells: dict, *, width=720, cell_h=54) -> str:
    """Fly variants down, opponents across. Tint carries the margin; the record is written in."""
    flies = [f for f in FLY_ORDER if any(k[0] == f for k in cells)]
    jevs = [j for j in OPPONENTS if any(k[1] == j for k in cells)]
    pad_l, pad_t, gap = 186, 30, 3
    cw = (width - pad_l) / max(1, len(jevs))
    height = pad_t + len(flies) * cell_h
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    for ci, j in enumerate(jevs):
        parts.append(f'<text class="tick" x="{pad_l + cw * (ci + 0.5):.1f}" y="{pad_t - 12}" text-anchor="middle">'
                     f'{html.escape(OPPONENTS[j][0])}</text>')
    for ri, f in enumerate(flies):
        y = pad_t + ri * cell_h
        parts.append(f'<text class="row-label" x="{pad_l - 14}" y="{y + cell_h / 2 + 4:.0f}" text-anchor="end">{html.escape(f)}</text>')
        for ci, j in enumerate(jevs):
            x = pad_l + cw * ci
            c = cells.get((f, j))
            if not c:
                parts.append(f'<rect class="cell-empty" x="{x + gap:.1f}" y="{y + gap}" width="{cw - gap * 2:.1f}" '
                             f'height="{cell_h - gap * 2}" rx="4"/>')
                continue
            n = c["w"] + c["d"] + c["l"] or 1
            margin = (c["w"] - c["l"]) / n
            fill = "var(--neutral)" if abs(margin) < 0.1 else ("var(--pole-cool)" if margin > 0 else "var(--pole-warm)")
            parts.append(
                f'<rect x="{x + gap:.1f}" y="{y + gap}" width="{cw - gap * 2:.1f}" height="{cell_h - gap * 2}" rx="4" '
                f'style="fill:{fill};fill-opacity:{0.14 + 0.5 * abs(margin):.2f}">'
                f'<title>{html.escape(f)} vs {html.escape(OPPONENTS[j][0])}: {c["w"]} won, {c["d"]} drawn, '
                f'{c["l"]} lost over {c["matches"]} matches</title></rect>'
                f'<text class="cell-record" x="{x + cw / 2:.1f}" y="{y + cell_h / 2:.0f}" text-anchor="middle">'
                f'{c["w"]}–{c["d"]}–{c["l"]}</text>'
                f'<text class="cell-sub" x="{x + cw / 2:.1f}" y="{y + cell_h / 2 + 15:.0f}" text-anchor="middle">'
                f'{c["matches"]} match{"es" if c["matches"] != 1 else ""}</text>')
    parts.append("</svg>")
    return "".join(parts)


def legend(items: list[tuple[str, str]]) -> str:
    dots = "".join(f'<span class="key"><i style="background:{var}"></i>{html.escape(name)}</span>' for name, var in items)
    return f'<div class="legend">{dots}</div>'


def table(headers: list[str], rows: list[list], caption: str) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<details class="tableview"><summary>Table: {html.escape(caption)}</summary>'
            f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></details>')


def figure(title: str, note: str, chart: str, legend_html: str, table_html: str) -> str:
    return (f'<figure><h3>{html.escape(title)}</h3><p class="note">{note}</p>'
            f'{legend_html}{chart}{table_html}</figure>')


# ---------- report ----------

def build(matches: list[dict]) -> str:
    rounds_all = [r for m in matches for r in m["rounds"]]
    jev_log = sum(len(m.get("jev_log", [])) for m in matches)
    fly_log = sum(len(r.get("fly_log", [])) for m in matches for r in m["rounds"])

    # damage per landed hit, real vs rewired, as the headline pair
    def dmg_per_hit(circuit: str) -> float:
        want = "Rewired control" if circuit == "rewired" else "Curriculum + thresholds"
        rs = [r for m in matches for r in m["rounds"]
              if (m.get("config") or {}).get("jev_mode") == "rules" and fly_variant(m.get("config") or {}) == want]
        hits = sum(r["hits"]["fly"] for r in rs)
        return sum(r["damage"]["fly"] for r in rs) / max(1, hits)

    real_dph, rewired_dph = dmg_per_hit("real"), dmg_per_hit("rewired")

    # 1. learning curves by opponent (real circuit)
    dps = lambda m: per_second(m["rounds"], lambda r: r["damage"]["fly"])
    curves = generation_series(matches, "real", dps, variant="Curriculum")
    series1, rows1 = [], []
    for key, (label, slot) in OPPONENTS.items():
        pts = sorted(curves.get(key, {}).items())
        if len(pts) < 2:
            continue
        series1.append({"name": label, "slot": slot, "points": pts})
        rows1.append([label] + [f"{v:.2f}" for _, v in pts])
    gens = sorted({g for s in series1 for g, _ in s["points"]})

    # 2. the rewired control, against the same opponent
    real_th = generation_series(matches, "real", dps, variant="Curriculum + thresholds")
    rew = generation_series(matches, "rewired", dps, variant="Rewired control")
    series2 = []
    if "rules" in real_th:
        series2.append({"name": "Real connectome", "slot": 1, "points": sorted(real_th["rules"].items())})
    if "rules" in rew:
        series2.append({"name": "Rewired control", "slot": 2, "points": sorted(rew["rules"].items())})

    # 3. what the fly learned: synapse strength by pool, per generation, against the rule bot
    pool_pts: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for m in matches:
        cfg = m.get("config") or {}
        gen, strength = cfg.get("generation"), ((m.get("learned") or {}).get("strength") or {})
        if gen and strength and cfg.get("jev_mode") == "rules" and fly_variant(cfg) == "Curriculum + thresholds":
            for pool in POOLS:
                if pool in strength:
                    pool_pts[pool][gen].append(strength[pool])
    series3 = [{"name": POOLS[p][0], "slot": POOLS[p][1], "points": sorted((g, mean(v)) for g, v in gs.items())}
               for p, gs in pool_pts.items()]

    # 4. outcomes by opponent, both circuits
    rows4 = []
    for circuit in ("real", "rewired"):
        for key, (label, _) in OPPONENTS.items():
            rs = [r for m in matches for r in m["rounds"]
                  if (m.get("config") or {}).get("jev_mode") == key
                  and ("rewired" if (m.get("config") or {}).get("circuit") == "rewired" else "real") == circuit]
            if not rs:
                continue
            won = sum(1 for r in rs if r["winner"] == "fly")
            drew = sum(1 for r in rs if r["winner"] == "draw")
            lost = len(rs) - won - drew
            rows4.append({"label": f"{label}{'' if circuit == 'real' else ' · rewired'}",
                          "parts": [("Fly won", won, "var(--pole-cool)"), ("Drawn", drew, "var(--neutral)"), ("Fly lost", lost, "var(--pole-warm)")]})

    # 5. what each side does, from the logged decisions
    def mix(entries: list[str]) -> list[float]:
        total = len(entries) or 1
        return [100 * entries.count(a) / total for a in ACTIONS]
    jev_actions = [e["a"] for m in matches for e in m.get("jev_log", [])]
    fly_actions = [e["a"] for m in matches for r in m["rounds"] for e in r.get("fly_log", [])]
    series5 = [s for s in ({"name": "Jev", "slot": 2, "values": mix(jev_actions)},
                           {"name": "Fly brain", "slot": 1, "values": mix(fly_actions)}) if sum(s["values"]) > 0]

    stat = lambda v, l, s="": f'<div class="stat"><b>{v}</b><span>{l}</span><i>{s}</i></div>'
    cells = records(matches)
    figures = [
        figure("Standings",
               "Not one fly and one Jev: each side fields several competitors, separated by what "
               "changes about them — for the fly, its learning setup and its wiring; for the other side, where "
               "the decisions come from. Rounds, from each competitor's own point of view. Nobody played a full "
               "schedule, so read this alongside the grid below: a good record against the rule bot is not the "
               "same achievement as a good record against the hybrid.",
               standings_table(cells),
               legend([("Won", "var(--pole-cool)"), ("Drawn", "var(--neutral)"), ("Lost", "var(--pole-warm)")]), ""),
        figure("Who actually played whom",
               "Every competitor against every other, as rounds won–drawn–lost by the fly. Read down a column "
               "to see what the fly's learning setup is worth against a fixed opponent; read across a row to see "
               "how far that setup carries. The wiring alone loses to the whole field; a brain carried in from an "
               "earlier curriculum is the only fly with a winning record against all four.",
               head_to_head(cells),
               legend([("Fly ahead", "var(--pole-cool)"), ("Level", "var(--neutral)"), ("Fly behind", "var(--pole-warm)")]),
               table(["Fly brain", "Opponent", "Matches", "Won", "Drawn", "Lost"],
                     [[f, OPPONENTS[j][0], c["matches"], c["w"], c["d"], c["l"]]
                      for (f, j), c in sorted(cells.items(), key=lambda kv: (FLY_ORDER.index(kv[0][0]), kv[0][1]))],
                     "every match-up")),
    ]
    if series1:
        figures.append(figure(
            "The fly learns, against every opponent but one",
            "Damage dealt per second by the fly, averaged over each generation's rounds. Each generation inherits the "
            "best brain saved so far and keeps learning. Against the hybrid — the local policy moving at the fly's own "
            "speed with Jev correcting it live — learning from scratch converges on not fighting at all.",
            line_chart(series1, x_label="generation", y_label="fly damage / second"),
            legend([(s["name"], f'var(--series-{s["slot"]})') for s in series1]),
            table(["Opponent"] + [f"gen {g}" for g in gens], rows1, "fly damage per second by generation")))
    if len(series2) == 2:
        figures.append(figure(
            "Scrambling the connectome: still learns, stops landing clean hits",
            f"The control keeps every neuron, every in- and out-degree, every transmitter sign and every synapse weight, "
            f"and shuffles only which neuron connects to which. It still learns to beat the rule bot — at "
            f"{rewired_dph:.1f} damage per landed hit against {real_dph:.1f}, which is what a blocked hit is worth.",
            line_chart(series2, x_label="generation", y_label="fly damage / second"),
            legend([(s["name"], f'var(--series-{s["slot"]})') for s in series2]),
            table(["Circuit"] + [f"gen {g}" for g, _ in series2[0]["points"]],
                  [[s["name"]] + [f"{v:.2f}" for _, v in s["points"]] for s in series2], "real vs rewired, damage per second")))
    if series3:
        figures.append(figure(
            "What learning changes, synapse by synapse",
            "Mean strength of the excitatory synapses onto each motor pool, relative to the wiring (1.0 = as mapped). "
            "The guard grows; the escape jump — a real circuit doing its job, useless against someone who keeps "
            "punching — is suppressed.",
            line_chart(series3, x_label="generation", y_label="strength vs wiring", y_zero=False),
            legend([(s["name"], f'var(--series-{s["slot"]})') for s in series3]),
            table(["Pool"] + [f"gen {g}" for g, _ in series3[0]["points"]],
                  [[s["name"]] + [f"{v:.2f}" for _, v in s["points"]] for s in series3], "learned strength by pool")))
    if rows4:
        figures.append(figure(
            "Every round played, by opponent",
            "All rounds in the telemetry, including the runs where the fly was learning from scratch and lost badly.",
            stacked_bars(rows4),
            legend([("Fly won", "var(--pole-cool)"), ("Drawn", "var(--neutral)"), ("Fly lost", "var(--pole-warm)")]),
            table(["Opponent", "Fly won", "Drawn", "Fly lost"],
                  [[r["label"]] + [p[1] for p in r["parts"]] for r in rows4], "round outcomes")))
    if series5:
        figures.append(figure(
            "What each side actually does",
            f"Share of decisions by move, from {jev_log + fly_log:,} logged decisions. Jev spreads itself across "
            "closing, striking and blocking; the fly, as wired, spends a quarter of its moves jumping away.",
            grouped_bars(ACTIONS, series5),
            legend([(s["name"], f'var(--series-{s["slot"]})') for s in series5]),
            table(["Move"] + [s["name"] for s in series5],
                  [[a] + [f'{s["values"][i]:.0f}%' for s in series5] for i, a in enumerate(ACTIONS)], "decision mix")))

    return PAGE.format(
        generated=time.strftime("%d %B %Y"),
        stats=(stat(f"{len(matches):,}", "matches") + stat(f"{len(rounds_all):,}", "rounds")
               + stat(f"{jev_log + fly_log:,}", "logged decisions")
               + stat(f"{real_dph:.1f}", "damage per hit", "real wiring, curriculum vs rule bot")
               + stat(f"{rewired_dph:.1f}", "damage per hit", "rewired control, same curriculum")),
        figures="\n".join(figures))


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Kombat — telemetry</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1: #fcfcfb; --surface-2: #f4f4f1; --line: #e2e2dd;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #6f6e6a;
    --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
    --pole-cool: #2a78d6; --pole-warm: #eb6834; --neutral: #a8a7a1;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) {{
      color-scheme: dark;
      --surface-1: #1a1a19; --surface-2: #232322; --line: #393936;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #9a9a92;
      --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
      --pole-cool: #3987e5; --pole-warm: #d95926; --neutral: #6f6e6a;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--surface-1); color: var(--text-primary);
    font: 16px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
  main {{ max-width: 860px; margin: 0 auto; padding: 40px 20px 64px; }}
  h1 {{ font-size: 2rem; line-height: 1.15; margin: 0 0 8px; letter-spacing: -0.02em; }}
  .sub {{ color: var(--text-secondary); margin: 0 0 28px; max-width: 62ch; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin: 0 0 40px; }}
  .stat {{ background: var(--surface-2); border-radius: 8px; padding: 12px 14px; }}
  .stat b {{ display: block; font-size: 1.6rem; line-height: 1.1; font-variant-numeric: tabular-nums; }}
  .stat span {{ display: block; font-size: .82rem; color: var(--text-secondary); }}
  .stat i {{ display: block; font-size: .72rem; color: var(--text-muted); font-style: normal; }}
  figure {{ margin: 0 0 44px; }}
  h3 {{ font-size: 1.12rem; margin: 0 0 4px; }}
  .note {{ color: var(--text-secondary); font-size: .92rem; margin: 0 0 14px; max-width: 68ch; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 6px; font-size: .84rem; color: var(--text-secondary); }}
  .key {{ display: inline-flex; align-items: center; gap: 6px; }}
  .key i {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; }}
  .chart {{ width: 100%; height: auto; overflow: visible; }}
  .grid {{ stroke: var(--line); stroke-width: 1; }}
  .tick {{ fill: var(--text-muted); font-size: 11px; }}
  .axis-label {{ fill: var(--text-secondary); font-size: 11px; }}
  .line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
  .dot {{ stroke: var(--surface-1); stroke-width: 2; }}
  .series-label, .row-label, .row-total {{ fill: var(--text-secondary); font-size: 12px; }}
  .row-label {{ fill: var(--text-primary); }}
  .in-bar {{ fill: #fff; font-size: 11px; font-variant-numeric: tabular-nums; }}
  .seg {{ stroke: var(--surface-1); stroke-width: 2; }}
  .cell-record {{ fill: var(--text-primary); font-size: 13px; font-variant-numeric: tabular-nums; }}
  .cell-sub {{ fill: var(--text-muted); font-size: 10.5px; }}
  .cell-empty {{ fill: var(--surface-2); }}
  .standings {{ font-size: .9rem; }}
  .standings th {{ color: var(--text-muted); font-weight: 500; font-size: .78rem; }}
  .standings td {{ padding: 9px 8px; vertical-align: top; }}
  .standings td:first-child {{ width: 46%; }}
  .standings .tag {{ font-size: .68rem; color: var(--text-muted); border: 1px solid var(--line);
    border-radius: 99px; padding: 1px 7px; margin-left: 6px; white-space: nowrap; }}
  .standings .desc {{ display: block; color: var(--text-secondary); font-size: .8rem; margin-top: 2px; }}
  .standings .sched {{ display: block; color: var(--text-muted); font-size: .74rem; margin-top: 3px; }}
  .standings tr.side-jev td:first-child b {{ font-style: italic; }}
  .recordcol {{ width: 128px; }}
  .recordbar {{ display: flex; gap: 2px; height: 10px; margin-top: 4px; }}
  .recordbar i {{ display: block; height: 100%; border-radius: 2px; }}
  .tableview {{ margin-top: 10px; font-size: .86rem; color: var(--text-secondary); }}
  .tableview summary {{ cursor: pointer; }}
  table {{ border-collapse: collapse; margin-top: 10px; width: 100%; }}
  th, td {{ text-align: right; padding: 5px 8px; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }}
  th:first-child, td:first-child {{ text-align: left; }}
  footer {{ color: var(--text-secondary); font-size: .88rem; border-top: 1px solid var(--line); padding-top: 18px; }}
  a {{ color: var(--series-1); }}
</style>
</head>
<body>
<main>
  <h1>Model Kombat: what the telemetry says</h1>
  <p class="sub">A 12,000-neuron slice of the <a href="https://male-cns.janelia.org/">maleCNS</a> connectome fights a
  language model in mk.js. Every match writes a JSON file; this page is generated from all of them by
  <code>scripts/report_telemetry.py</code>. Generated {generated}.</p>
  <div class="stats">{stats}</div>
  {figures}
  <footer>
    <p>Two caveats travel with every number here. Jev is handed the game's own hit predicate
    (<code>in_attack_range</code>) plus both life totals and the clock, while the fly gets modelled sensory drives and
    never learns the score — this was never a clean contest between two intelligences. And most configurations were run
    once; the headline result was replicated twice.</p>
    <p>Method, failures and wrong predictions: <a href="https://github.com/lavallee/mk-jev-fly-brain/blob/main/EXPERIMENTS.md">EXPERIMENTS.md</a> ·
    code: <a href="https://github.com/lavallee/mk-jev-fly-brain">github.com/lavallee/mk-jev-fly-brain</a></p>
  </footer>
</main>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", default=os.path.join(ROOT, "telemetry"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "index.html"))
    args = ap.parse_args()

    matches = load(args.telemetry)
    if not matches:
        raise SystemExit(f"no complete matches under {args.telemetry}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(build(matches))
    print(f"wrote {args.out} from {len(matches)} matches "
          f"({sum(len(m['rounds']) for m in matches)} rounds)")


if __name__ == "__main__":
    main()
