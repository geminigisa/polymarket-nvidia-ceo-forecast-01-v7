# Sticky-mixture quantile forecaster V7

Forecasts the Yes price of every market of Polymarket's "Next CEO of
NVIDIA?" event, one minute at a time, for the 240 minutes after a
checkpoint, at the quantiles 0.05 / 0.25 / 0.5 / 0.75 / 0.95.

## The model

Prices in this event are sticky: they often sit unchanged for minutes,
then move in a tick, and occasionally jump. V7 models each market as a
two-part mixture. With probability **p0** the price does not move at all
— an atom sitting exactly at the last observed price. With probability
**1 − p0** it moves like a Normal variable around a slow drift.

Both parts are measured from the market's own history, per lag
(1, 2, 4, 8, 16, 32, 64, 128, 240 minutes). p0 is the fraction of lagged
moves that are exactly zero, clamped to [0.02, 0.9]. The scale s comes
from the median absolute deviation of the nonzero moves, measured over
the most recent 720 minutes when there are enough of them, otherwise
over all moves, otherwise a small fallback that grows with the square
root of the lag. For each forecast minute, p0 and s are interpolated in
log-horizon between the measured lags.

The drift pulls the price slowly back toward the median of the last
1440 minutes, with a time constant of 1200 minutes. The scale is
adjusted by regime: 0.8 when the market is fully stagnant, 1.3 after a
recent jump of more than 0.05 or when one-minute stickiness is low,
otherwise 1.0.

## What makes V7 different

Three changes on top of the earlier sticky-mixture design:

1. **Centre shift −0.30 × s.** Every quantile is shifted down by three
   tenths of the scale. The model's median was sitting too high: the
   price lands exactly on the last price far more often than a symmetric
   model expects, so the whole distribution is pulled slightly down.
2. **τ95 widening ×1.3.** Only the 95% quantile's scale is widened.
   The earlier model undercovered the upper tail — the price landed
   above its 95% quantile far more often than one time in twenty —
   because the centre shift drags the upper tail down with it. Widening
   just the top quantile repairs the tail without touching the median.
3. **Gated cross-market sum drift.** The last prices of all markets in
   a case should roughly sum to 1 (one successor will be announced).
   When the sum is far from 1 (|sum − 1| > 0.10) that dislocation is a
   real signal and the drift leans against it at strength 0.05; when
   the sum is close to 1 the correction stays off, so it cannot add
   noise to ordinary cases.

## Interface

Python 3.13, standard library only (`json`, `math`, `os`,
`statistics`, `pathlib`). Deterministic — no randomness, no network,
no files read outside the input directory. Reads
`$MAKEFUN_INPUT_DIR` (default `/input`), writes one `<case>.out` per
`*.in` file to `$MAKEFUN_OUTPUT_DIR` (default `/output`), each a single
JSON object `{"forecasts": [{"id": ..., "q": [[q05, q25, q50, q75, q95]
× 240 rows]}]}`. A market with no price history forecasts 0.5 at every
quantile. Every value is clamped to [0, 1], rounded to 6 decimals, and
each row is sorted so the quantiles never decrease.
