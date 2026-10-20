"""Regenerate the result charts from results/data/ (standard library only).

    python results/figures/make_figures.py

Writes pass_rate_over_time.svg and team_size_over_time.svg next to this file. Colors are slots 1-3 of a
categorical palette validated for color-vision deficiency; every series is also labelled directly.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dd"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # M1 first run, M1 rerun, M0
FONT = "font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def frame(width, height, x0, x1, y, ticks, unit, ylabel):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="{FONT}">',
           f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>']
    for v in ticks:
        out.append(f'<line x1="{x0}" x2="{x1}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}"/>'
                   f'<text x="{x0 - 8}" y="{y(v) + 4:.1f}" text-anchor="end" font-size="11" fill="{MUTED}">{v}{unit}</text>')
    out.append(f'<text x="{x0}" y="18" font-size="12" fill="{MUTED}">{ylabel}</text>')
    return out


def x_axis(out, x, base, x1):
    for m in range(0, 361, 60):
        out.append(f'<text x="{x(m):.1f}" y="{base + 18}" text-anchor="middle" font-size="11" fill="{MUTED}">{m}</text>')
    out.append(f'<text x="{x1}" y="{base + 34}" text-anchor="end" font-size="12" fill="{MUTED}">minutes into the 6-hour run</text>')


def label(out, x, y, text, anchor="start", color=INK, size=12):
    # A surface-colored halo keeps labels legible where they cross grid or reference lines.
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" fill="{color}" '
               f'stroke="{SURFACE}" stroke-width="4" paint-order="stroke" stroke-linejoin="round">{text}</text>')


def pass_rate():
    W, H, X0, X1, TOP, BASE = 820, 420, 60, 600, 50, 360
    def x(m):
        return X0 + (X1 - X0) * m / 360

    def y(v):
        return TOP + (BASE - TOP) * (1 - v / 100)

    out = frame(W, H, X0, X1, y, range(0, 101, 20), "%", "pass rate on pandoc")
    x_axis(out, x, BASE, X1)
    for m, text in ((107, "M1 rerun: freeze-prep (107 min)"), (143, "M1 first run: freeze (143), Stop hook (145)")):
        out.append(f'<line x1="{x(m):.1f}" x2="{x(m):.1f}" y1="{y(100):.1f}" y2="{y(0):.1f}" stroke="{MUTED}" stroke-dasharray="2 4"/>')
    label(out, x(107) - 4, y(97), "M1 rerun: freeze-prep at 107 min", "end", MUTED, 11)
    label(out, x(143) + 4, y(97), "M1 first run: freeze at 143, Stop hook at 145", "start", MUTED, 11)
    curve = [(int(r["minute"]), float(r["pass_rate_percent"])) for r in csv.DictReader(open(DATA / "m1_rerun_pass_rate_curve.csv"))]
    held = list(csv.DictReader(open(DATA / "team_held_out_measurements.csv")))
    rerun_held = [(float(r["minute"]), float(r["pass_rate_percent"])) for r in held if r["run"] == "M1 rerun"]
    first_held = [(float(r["minute"]), float(r["pass_rate_percent"])) for r in held if r["run"] == "M1 first run"]
    first_line = [p for p in first_held if p in {(83.8, 41.46), (142.6, 53.57), (208.1, 56.2)}]

    def polyline(points, color, dashed=False):
        d = " ".join(f"{x(m):.1f},{y(v):.1f}" for m, v in points)
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')

    polyline([(0, 0)] + curve, ORANGE)
    for m, v in curve:
        out.append(f'<circle cx="{x(m):.1f}" cy="{y(v):.1f}" r="4" fill="{ORANGE}" stroke="{SURFACE}" stroke-width="2"/>')
    polyline(rerun_held, ORANGE, dashed=True)
    polyline(first_line, BLUE, dashed=True)
    for points, color in ((rerun_held, ORANGE), (first_held, BLUE)):
        for m, v in points:
            out.append(f'<circle cx="{x(m):.1f}" cy="{y(v):.1f}" r="4" fill="{SURFACE}" stroke="{color}" stroke-width="2"/>')
    out.append(f'<circle cx="{x(240):.1f}" cy="{y(61.09):.1f}" r="6" fill="{BLUE}" stroke="{SURFACE}" stroke-width="2"/>')
    out.append(f'<circle cx="{x(360):.1f}" cy="{y(54.26):.1f}" r="6" fill="{AQUA}" stroke="{SURFACE}" stroke-width="2"/>')
    label(out, x(107) + 8, y(81.69) - 10, "M1 rerun, team's own 'held-out' set (also used to steer fixes)")
    label(out, x(240) + 9, y(61.09) - 8, "M1 first run, official at 4 h: 61.1%")
    label(out, x(215), y(46), "M1 first run, freshly sealed held-out sets", "middle")
    label(out, x(83.8), y(41.46) + 22, "41.5% while the team's own grid was 100%", "middle")
    label(out, X1 + 10, y(54.37) - 4, "M1 rerun, official: 54.4%")
    label(out, X1 + 10, y(54.37) + 14, "M0, official at 6 h: 54.3%")
    out.append(f'<text x="{X0}" y="{H - 8}" font-size="11" fill="{MUTED}">Solid lines and filled dots: official score on 5,214 hidden tests. '
               'Dashed lines and hollow dots: case sets the team built itself.</text>')
    out.append("</svg>")
    (HERE / "pass_rate_over_time.svg").write_text("\n".join(out) + "\n")


def team_size():
    W, H, X0, X1, TOP, BASE = 820, 400, 100, 640, 40, 260
    def x(m):
        return X0 + (X1 - X0) * m / 360

    def y(v):
        return TOP + (BASE - TOP) * (1 - v / 16)

    out = frame(W, H, X0, X1, y, range(0, 17, 4), "", "active workers")
    data = json.load(open(DATA / "team_size_timeline.json"))
    colors = {"M0": AQUA, "M1 first run": BLUE, "M1 rerun": ORANGE}
    for name, run in data.items():
        values = run["active_workers_per_10_minutes"]
        d = f"M{x(0):.1f},{y(values[0]):.1f}"
        for i, v in enumerate(values):
            if v is None:  # after the run's last request
                break
            d += f" V{y(v):.1f} H{x(min(360, 10 * (i + 1))):.1f}"
        out.append(f'<path d="{d}" fill="none" stroke="{colors[name]}" stroke-width="2"/>')
    for row, (name, run) in enumerate(data.items()):
        ry = BASE + 40 + row * 22
        out.append(f'<line x1="{X0}" x2="{X1}" y1="{ry}" y2="{ry}" stroke="{GRID}"/>')
        label(out, X0 - 8, ry + 4, name, "end", MUTED, 11)
        for m in run["stop_hook_blocks_minutes"]:
            out.append(f'<line x1="{x(m):.1f}" x2="{x(m):.1f}" y1="{ry - 6}" y2="{ry + 6}" stroke="{colors[name]}" stroke-width="2"/>')
        label(out, X1 + 10, ry + 4, f"{len(run['stop_hook_blocks_minutes'])} Stop hook block{'s' if len(run['stop_hook_blocks_minutes']) != 1 else ''}", "start", MUTED, 11)
    x_axis(out, x, BASE + 84, X1)
    label(out, x(250), y(14.5), "M1 first run: back to 16 after the block", "middle")
    label(out, x(250), y(2.6), "M1 rerun: 1–2 workers from 113 min")
    label(out, x(300), y(8.3) - 6, "M0: 5–9 workers", "middle")
    out.append(f'<text x="{X0}" y="{H - 8}" font-size="11" fill="{MUTED}">Ticks: times the Stop hook refused the master\'s attempt '
               'to end the session early. The M1 first run stopped at 4 h when the API balance ran out.</text>')
    out.append("</svg>")
    (HERE / "team_size_over_time.svg").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    pass_rate()
    team_size()
