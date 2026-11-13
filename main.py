"""Sticky-mixture quantile forecaster V7.

Forecasts the Yes price of every market of Polymarket's "Next CEO of
NVIDIA?" event, one minute at a time, for the 240 minutes after a
checkpoint, at the quantiles 0.05 / 0.25 / 0.5 / 0.75 / 0.95.

Model, per market: a two-part mixture. With probability p0 the price does
not move at all (an atom at the last price); with probability 1 - p0 it
moves like a Normal variable centred on a slow mean-reversion drift. p0
and the scale s are measured per lag from the history of lagged moves
(moves of exactly zero give p0; the median absolute deviation of the
nonzero moves, over the most recent 720 minutes, gives s), then
interpolated in log-horizon for each forecast minute. A centre shift of
-0.30 * s pulls every quantile down a little (the median sits at the last
price more often than a symmetric model expects), the 95% quantile's
scale is widened by 1.3 (the upper tail was undercovered), and a
cross-market sum drift fires only when the case's last prices sum far
from 1 (a genuine dislocation between markets).

Python 3.13, standard library only. Deterministic, no network, no
randomness. Reads $MAKEFUN_INPUT_DIR (default /input), writes one
<case>.out per *.in file to $MAKEFUN_OUTPUT_DIR (default /output).

Local validation (2026-11-13, official 6-case harness, full-precision
kit inputs): six-case pinball 0.270965 (p00 0.416678, p01 0.311221,
p02 0.362731, p03 0.295379, p04 0.186924, p05 0.111736); max runtime
0.24 s per case (limit 8 s); deterministic.
"""

import json
import math
import os
import statistics
from pathlib import Path

# --- Model constants -------------------------------------------------------
LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 240)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
HORIZON = 240
S_SCALE = 0.50
P0_MIN = 0.02
P0_MAX = 0.9
S_FLOOR = 0.0001
TAU = 1200.0
S_WINDOW = 720
MED_WINDOW = 1440
FALLBACK_SIGMA = 0.0025
REGIME_STAGNANT_P0 = 0.9
REGIME_JUMP_P0 = 0.5
REGIME_JUMP_MOVE = 0.05
REGIME_JUMP_WINDOW = 60

# --- V7 knobs ---------------------------------------------------------------
SHIFT = -0.30            # centre shift, same for every quantile
SUM_K = 0.05             # cross-market sum drift strength
SUM_THRESH = 0.10        # sum drift fires only when |sum_dev| > 0.10
# tau95 widening plateau (6-case gain vs V6-C): x1.2 -0.000926,
# x1.3 -0.000989, x1.4 -0.000937 — broad plateau, not a sharp peak.
S_MULT_TAU = (1.0, 1.0, 1.0, 1.0, 1.3)   # only tau95 scale widened x1.3
ATOM_SHIFT = 0.0         # atom sits exactly at the last price

_ND = statistics.NormalDist()


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _scale_from_moves(moves):
    """MAD-based scale from a list of moves, or None if not usable.

    Uses the nonzero moves only; needs at least 5 of them and a
    positive median absolute deviation.
    """
    nz = [m for m in moves if m != 0.0]
    if len(nz) < 5:
        return None
    med = statistics.median(nz)
    mad = statistics.median([abs(m - med) for m in nz])
    if mad <= 0.0:
        return None
    return 1.4826 * mad * S_SCALE


def _lag_stats(prices):
    """Per-lag (p0, s) points, as two sorted lists of (lag, value)."""
    n = len(prices)
    p0_pts = []
    s_pts = []
    for lag in LAGS:
        if lag >= n:
            continue
        moves = [prices[i] - prices[i - lag] for i in range(lag, n)]
        zeros = sum(1 for m in moves if m == 0.0)
        p0 = _clamp(zeros / len(moves), P0_MIN, P0_MAX)
        p0_pts.append((lag, p0))
        # Scale: prefer the nonzero moves in the most recent window.
        start = max(lag, n - S_WINDOW)
        s = _scale_from_moves(
            [prices[i] - prices[i - lag] for i in range(start, n)])
        if s is None:
            s = _scale_from_moves(moves)
        if s is None:
            s = FALLBACK_SIGMA * math.sqrt(lag)
        s_pts.append((lag, max(s, S_FLOOR)))
    if not p0_pts:
        p0_pts = [(1, 0.5)]
        s_pts = [(1, FALLBACK_SIGMA)]
    return p0_pts, s_pts


def _interp_log_horizon(points, h):
    """Linear interpolation in log(h) over sorted (lag, value) points."""
    if h <= points[0][0]:
        return points[0][1]
    if h >= points[-1][0]:
        return points[-1][1]
    for (l1, v1), (l2, v2) in zip(points, points[1:]):
        if l1 <= h <= l2:
            t = (math.log(h) - math.log(l1)) / (math.log(l2) - math.log(l1))
            return v1 + (v2 - v1) * t
    return points[-1][1]


def _regime_scale(prices, p0_pts):
    """0.8 when the market is stagnant, 1.3 after a jump or when the
    1-minute stickiness is low, else 1.0."""
    p0_by_lag = dict(p0_pts)
    if p0_by_lag.get(1, 0.0) > REGIME_STAGNANT_P0:
        return 0.8
    n = len(prices)
    start = max(1, n - REGIME_JUMP_WINDOW)
    jumped = any(abs(prices[i] - prices[i - 1]) > REGIME_JUMP_MOVE
                 for i in range(start, n))
    if jumped or p0_by_lag.get(1, 1.0) < REGIME_JUMP_P0:
        return 1.3
    return 1.0


def _forecast_market(prices, sum_dev):
    """240 rows x 5 quantiles for one market."""
    if not prices:
        return [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(HORIZON)]
    n = len(prices)
    p_last = prices[-1]
    p0_pts, s_pts = _lag_stats(prices)
    regime = _regime_scale(prices, p0_pts)
    med24 = statistics.median(prices[-MED_WINDOW:])
    sum_k = SUM_K if abs(sum_dev) > SUM_THRESH else 0.0

    rows = []
    for h in range(1, HORIZON + 1):
        p0 = _interp_log_horizon(p0_pts, h)
        s = max(_interp_log_horizon(s_pts, h) * regime, S_FLOOR)
        decay = 1.0 - math.exp(-h / TAU)
        drift = (med24 - p_last) * decay - sum_k * sum_dev * decay
        lo = (1.0 - p0) * 0.5
        hi = lo + p0
        cs = SHIFT * s
        row = []
        for j, q in enumerate(QUANTILES):
            sm = S_MULT_TAU[j]
            if q < lo:
                qn = q / (1.0 - p0)
                val = p_last + drift + cs + s * sm * _ND.inv_cdf(qn)
            elif q <= hi:
                val = p_last + ATOM_SHIFT * s
            else:
                qn = (q - p0) / (1.0 - p0)
                val = p_last + drift + cs + s * sm * _ND.inv_cdf(qn)
            row.append(round(_clamp(val, 0.0, 1.0), 6))
        row.sort()
        rows.append(row)
    return rows


def _forecast_case(case):
    markets = case["markets"]
    lasts = [m["prices"][-1] for m in markets if m["prices"]]
    sum_dev = sum(lasts) - 1.0
    forecasts = []
    for market in markets:
        forecasts.append({
            "id": market["id"],
            "q": _forecast_market(market["prices"], sum_dev),
        })
    return {"forecasts": forecasts}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(in_dir.glob("*.in")):
        with open(in_path, "r", encoding="utf-8") as f:
            case = json.load(f)
        out = _forecast_case(case)
        out_path = out_dir / (in_path.stem + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
