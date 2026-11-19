# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/agungherman/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/agungherman/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many GTA VI players will earn each PS5 trophy in week one: for
all 55 trophies it predicts the share of tracked players holding the trophy
seven days after release, as five quantiles (5/25/50/75/95%) of the Exophase
earn rate.

The model is a hyperbolic age model. Players join over time, so a trophy's
earn rate follows the *average age* of the player base rather than calendar
time: the tracked-player count N(t) is extrapolated with a saturating
logistic curve fitted over the distinct Exophase snapshots (never a power
law), the average player age A(t) is the integral of N over N, and each
trophy's rate follows r(A) = r_inf * A / (A + tau) with tau chosen on a grid.
The spread combines a random-walk term, a refresh-jump volatility term
(Exophase refreshes in jumps, which dilutes rates), the per-trophy fit error
and the uncertainty in tau. When history is flat or stale, a stale-page
fallback is used instead: a delayed refresh means N is about to jump, which
dilutes rates, so the median is pulled down. Pure Python stdlib, one file,
fully deterministic — no clock, no randomness, no network.

On the kit's practice cases the model scores a mean pinball loss of about
0.53 versus 1.52 for the persistence baseline; on simulated final cases it
averages about 0.39 versus 0.67 for persistence.

## Run it

```sh
python3 main.py
```

The program reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default
`/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (default
`/output`). Each output is one JSON object `{"trophies": [{"id": n, "q":
[q05, q25, q50, q75, q95]}, ...]}` with one row per trophy in case order.

---

<sub>Entered on [make.fun](https://make.fun/?ref=5wyg9qx0&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
