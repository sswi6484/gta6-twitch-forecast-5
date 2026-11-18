# Joint two-dimensional kernel quantile smoothing

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sswi6484/gta6-twitch-forecast-5/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sswi6484/gta6-twitch-forecast-5/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard)

Forecasts GTA VI's launch-day Twitch viewers hour by hour (hours 2–24, five quantiles
each) for the make.fun bounty
[gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

## Model

Joint two-dimensional kernel-weighted empirical quantiles over (log hour-1, log hour).
The forecast anchors on the case's hour-1 print and pools the comparables' release-day
log-ratios, weighting each observation (launch `c`, hour `h'`) for a target hour `h` by

    w = exp(-0.5 * ((ln(h1_c) - ln(p1)) / 0.75) ** 2)
      * exp(-0.5 * ((h' - h) / 2.0) ** 2)

so a target hour borrows strength from launches of a similar hour-1 size *and* from
neighbouring hours of the day curve. Quantiles are the weighted empirical quantiles of
the pooled ratios at 0.05 / 0.25 / 0.5 / 0.75 / 0.95; `q = p1 * exp(quantile)`.
Standard library only, deterministic, no network. `main.py` reads
`$MAKEFUN_INPUT_DIR` (else `/input/`) and writes `$MAKEFUN_OUTPUT_DIR` (else `/output/`).

## Leave-one-out verification

Held each of the 37 usable comparable launches out in turn (known = its hour-1 print,
comparables = the other 37) and scored mean pinball loss on the log scale against its
own hours 2–24. House baseline scored on the same pseudo-cases for comparison.

| | overall | big hour-1 (≥100k) | huge (≥250k) |
|---|---:|---:|---:|
| this entry | 0.4021 | 0.2131 | 0.1338 |
| house baseline | 0.9297 | 0.2611 | 0.2255 |

Beats the house on both headline numbers and in every size regime. Honest misses: the
model loses to the house on a few launches — Starfield (0.3400 vs 0.2909), Monster
Hunter Wilds (0.4607 vs 0.4273), Diablo IV (0.1552 vs 0.1254) and Marvel Rivals
(0.2329 vs 0.2097) — launches whose day curves fall away from the shape of
similar-sized launches. Small launches (hour-1 < 10k) are the weakest regime (0.8321);
the final's hour-1 will be far above that range.
