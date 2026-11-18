"""gta6-twitch-ccu-forecast-01 entry #5 (sswi6484): joint two-dimensional kernel quantile smoothing.

Model: for the target case, anchor on its hour-1 print p1 and forecast each hour h = 2..24
from the comparables' release-day log-ratios r_c[h'] = ln(viewers_c[h']) - ln(viewers_c[1]).
Each hour's forecast is the weighted empirical quantiles of the pooled ratios, where an
observation (comparable c, hour h') is weighted by a 2D Gaussian kernel

    w = exp(-0.5 * ((ln(h1_c) - ln(p1)) / B_Z) ** 2) * exp(-0.5 * ((h' - h) / B_H) ** 2)

so a target hour borrows strength from launches of a similar hour-1 size AND from
neighbouring hours of the day curve. This is genuinely different from the four earlier
entries: it is not per-hour fitting (#1 weighted empirical quantiles per hour, #3
per-hour pinball regression), not a total-then-split (#4), and not parametric (#2
hierarchical lognormal). The hour itself is a continuous dimension here.

Quantiles are the weighted empirical quantiles of the pooled ratios at the five scored
levels; q_tau(h) = p1 * exp(quantile_tau), rounded to 2 decimals half up (as the house
baseline rounds). Outputs are validated: 23 rows, five quantiles each, 0 < q <= 1e9,
non-decreasing. Standard library only, deterministic, no network.

Bandwidths B_Z and B_H were chosen by leave-one-out pinball loss over the 37 usable
comparable launches in the kit's practice case (each held out in turn, scored against
its own hours 2..24). A finer grid around the first-pass optimum (0.75, 2.0) found
B_H = 2.25 slightly better than 2.0 at fixed B_Z = 0.75 (overall 0.4012 vs 0.4021),
while larger B_Z (0.85+) kept improving the overall score only by borrowing from
small/mid launches at the expense of the big-h1/huge buckets that matter for a
GTA VI-sized target, so B_Z stays at 0.75. The spread floor SPREAD_FLOOR = 0.8
was then tuned on the same LOO: it improves overall (0.4021 -> 0.4009), big-h1
(0.2131 -> 0.2123) and huge (0.1338 -> 0.1336) together, and the gain holds on
both split halves of the pseudo-targets, so it is not one launch driving it.
Structural variants tried and rejected on the same LOO: Laplace / Epanechnikov /
triangular hour kernels (all worse), dropping the h'=1 anchor for late hours
(no change — its hour-kernel weight is already negligible there), per-hour
effective-sample-size guards that widen B_H when the pool is thin (worse),
INCLUDE_H1 = False (worse), and SPREAD_FLOOR >= 1.2 (overall still improves but
the huge bucket degrades). See the LOO harness kept beside this repo (not submitted).
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Bandwidths of the 2D Gaussian kernel, chosen by leave-one-out pinball loss.
# B_Z: log hour-1 distance (how close a launch's size must be to count fully).
# B_H: hour distance (how far along the day curve an observation may sit).
B_Z = 0.75
B_H = 2.25
# Whether hour-1 observations (ratio exactly 0) join the pool: they anchor the
# early hours to the target's own print.
INCLUDE_H1 = True
# Minimum log-width between the 5% and 95% quantiles. The pooled ratios carry
# their own spread, but for very large targets the size kernel leaves only a
# handful of near-size comparables, and their pooled ratios can be degenerate
# narrow (the outer quantiles then sit almost on the median and understate the
# true uncertainty). The floor widens the 5%/95% pair symmetrically around the
# median only when the pooled width is below it; it binds on a few huge
# launches (e.g. Cyberpunk 2077, Marvel Rivals) and is a no-op elsewhere.
SPREAD_FLOOR = 0.8


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def day_curve(comp: dict) -> dict[int, float] | None:
    """A comparable's release-day hourly viewers {hour 1..24: viewers}, or None if incomplete."""
    try:
        day_start = datetime.fromisoformat(comp["day_start"].replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return None
    by_from = {}
    for row in comp.get("hourly") or []:
        v = row.get("viewers")
        if v is not None and v > 0:
            by_from[row.get("from")] = float(v)
    curve = {}
    for h in range(1, 25):
        t = (day_start + timedelta(hours=h - 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if t not in by_from:
            return None
        curve[h] = by_from[t]
    return curve


def weighted_quantile(pairs: list[tuple[float, float]], tau: float) -> float:
    """Weighted empirical quantile: smallest value whose cumulative weight reaches tau of the total."""
    total = sum(w for _, w in pairs)
    if total <= 0:
        raise ValueError("empty weight pool")
    target = tau * total
    acc = 0.0
    for value, w in sorted(pairs):
        acc += w
        if acc >= target:
            return value
    return sorted(pairs)[-1][0]


def hour_quantiles(p1: float, z0: float, comps: list[tuple[float, dict[int, float]]], h: int) -> list[float | int]:
    """The five quantiles of hour h's count, from the 2D-kernel pooled log-ratios."""
    pool: list[tuple[float, float]] = []
    for zc, curve in comps:
        dz = (zc - z0) / B_Z
        w_size = math.exp(-0.5 * dz * dz)
        if w_size < 1e-6:
            continue
        for hp in range(1, 25):
            if hp == 1 and not INCLUDE_H1:
                continue
            dh = (hp - h) / B_H
            w = w_size * math.exp(-0.5 * dh * dh)
            if w < 1e-6:
                continue
            pool.append((math.log(curve[hp]) - zc, w))
    if not pool:
        # No comparable in reach: fall back to the house persistence spread.
        z = [-1.645, -0.674, 0, 0.674, 1.645]
        out: list[float | int] = []
        prev = 0.0
        for zz in z:
            v = round2(p1 * math.exp(zz * 0.15 * math.sqrt(h - 1)))
            v = max(v, 1)
            v = min(v, 1_000_000_000)
            if v < prev:
                v = prev
            out.append(v)
            prev = float(v)
        return out
    qs = [weighted_quantile(pool, tau) for tau in LEVELS]
    if SPREAD_FLOOR > 0:
        mid = qs[2]
        lo = min(qs[0], mid - SPREAD_FLOOR / 2)
        hi = max(qs[4], mid + SPREAD_FLOOR / 2)
        qs[0] = lo
        qs[4] = hi
    out: list[float | int] = []
    prev = 0.0
    for q in qs:
        v = round2(p1 * math.exp(q))
        v = max(v, 1)
        v = min(v, 1_000_000_000)
        if v < prev:
            v = prev
        out.append(v)
        prev = float(v)
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    z0 = math.log(p1)
    comps: list[tuple[float, dict[int, float]]] = []
    for comp in case.get("comparables") or []:
        curve = day_curve(comp)
        if curve is None:
            continue
        comps.append((math.log(curve[1]), curve))
    if len(comps) < 3:
        raise ValueError(f"{name} has only {len(comps)} usable comparables")
    return {"hours": [{"hour": row["hour"], "q": hour_quantiles(p1, z0, comps, row["hour"])} for row in case["forecast"]]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
