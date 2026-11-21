"""gta6-trophy-rarity-forecast-01 entry: per-trophy log-time trend on the logit scale.

Model
-----
For each trophy, the earn rate is a ratio (earners / players tracked).  Over the
launch week the player base grows fast, so early trophies (story chapter 1,
early activities) rise steeply as most players complete them, while late
trophies (final mission, platinum, 100% completion) dilute toward 0 because
few players have reached them yet.  A single flat forecast misses both.

For every trophy we fit, on the case's history, a weighted least-squares line
of logit(rate) against ln(1 + hours since launch).  The line is read at the
target; its slope carries the trophy's own direction (rising or diluting),
and the spread combines a base uncertainty with the slope's standard error,
so trophies whose trend is poorly identified get a wider band.  The total
logit move from the last read is capped, so a steep early slope cannot run
away over a 4.5-day horizon.  Rates are clamped to [0.01, 99.99] before the
logit (the page prints 2 decimals; 0% and 100% have no logit), and a trophy
at 100% stays near 100%.

When the history is flat (the practice cases, launch night before the first
refresh) the slope is 0 and the forecast is the last rate with a wide spread:
there is no trend to extrapolate, and the honest answer is a broad band.

Quantiles use the standard normal's z at each scored level.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Launch reference: the game's release day, the origin of the log-time axis.
LAUNCH = "2026-11-19T00:00:00Z"
# Base logit spread (uncertainty that is there even with a perfect trend).
BASE_SPREAD = 0.9
# Cap on the absolute logit move from the last read to the target.
MAX_MOVE = 3.0


def parse_time(s):
    """ISO 8601 UTC (with fractional seconds) -> epoch seconds."""
    s = s.replace("Z", "+00:00")
    from datetime import datetime

    return datetime.fromisoformat(s).timestamp()


LAUNCH_T = parse_time(LAUNCH)


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def fit_trend(xs, ys):
    """Weighted least squares y = a + b*x. Returns (a, b, se_b, resid_sd).

    Weights rise with recency so a changing trend is followed.  With fewer
    than 3 points, or no variation in x, there is no slope to fit.
    """
    n = len(xs)
    if n < 2:
        return (ys[-1] if ys else 0.0, 0.0, 0.0, 0.0)
    if n == 2:
        # Two points define the line exactly.
        dx = xs[1] - xs[0]
        if abs(dx) <= 1e-12:
            return (ys[-1], 0.0, 0.0, 0.0)
        b = (ys[1] - ys[0]) / dx
        a = ys[0] - b * xs[0]
        return (a, b, 0.0, 0.0)
    # Recency weights: linear ramp, newest worth 3x the oldest.
    w = [1.0 + 2.0 * (i / (n - 1)) for i in range(n)]
    sw = sum(w)
    mx = sum(wi * xi for wi, xi in zip(w, xs)) / sw
    my = sum(wi * yi for wi, yi in zip(w, ys)) / sw
    sxx = sum(wi * (xi - mx) ** 2 for wi, xi in zip(w, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, 0.0)
    sxy = sum(wi * (xi - mx) * (yi - my) for wi, xi, yi in zip(w, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    # Residuals and the slope's standard error.
    resid = [yi - (a + b * xi) for xi, yi in zip(xs, ys)]
    dof = max(n - 2, 1)
    s2 = sum(wi * r * r for wi, r in zip(w, resid)) / (sw * dof / n) if n > 2 else 0.0
    se_b = math.sqrt(s2 / sxx) if sxx > 0 else 0.0
    resid_sd = math.sqrt(sum(r * r for r in resid) / dof) if n > 2 else 0.0
    return (a, b, se_b, resid_sd)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    target_t = parse_time(case["target"])
    x_target = math.log(1.0 + max(0.0, (target_t - LAUNCH_T) / 3600.0))
    last = history[-1]
    x_last = math.log(1.0 + max(0.0, (parse_time(last["at"]) - LAUNCH_T) / 3600.0))

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        xs, ys = [], []
        for read in history:
            rate = read["rates"][i]
            if rate is None:
                continue
            t = parse_time(read["at"])
            xs.append(math.log(1.0 + max(0.0, (t - LAUNCH_T) / 3600.0)))
            ys.append(logit(rate))
        if not ys:
            # No read ever listed this trophy: forecast near zero, wide band.
            y_med, spread = logit(0.01), 2.0
        else:
            a, b, se_b, resid_sd = fit_trend(xs, ys)
            y_last = ys[-1]
            y_pred = a + b * x_target
            # Cap the move from the last read so a steep slope cannot run away.
            move = y_pred - y_last
            if move > MAX_MOVE:
                y_pred = y_last + MAX_MOVE
            elif move < -MAX_MOVE:
                y_pred = y_last - MAX_MOVE
            y_med = y_pred
            # Spread: base + slope uncertainty + residual scatter, growing
            # with how far ahead the target is.
            horizon_factor = math.sqrt(max(0.0, x_target - x_last) + 1.0)
            spread = math.sqrt(
                BASE_SPREAD * BASE_SPREAD
                + (se_b * max(0.0, x_target - x_last)) ** 2
                + (resid_sd * horizon_factor) ** 2
            )
            spread = min(max(spread, 0.5), 3.0)
        q = [round4(100.0 * sigmoid(y_med + z * spread)) for z in Z]
        # Keep the quantiles non-decreasing (sigmoid is monotone, but the
        # rounding can tie; enforce it explicitly).
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
