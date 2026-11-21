# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vasm-696/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vasm-696/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI PlayStation 5 trophy's Exophase earn rate at 2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy.

## Model

For every trophy, the earn rate is a ratio (earners / players tracked). Over the launch week the player base grows fast, so early trophies (Chapter 1, early activities) rise steeply as most players complete them, while late trophies (final mission, platinum, 100% completion) dilute toward 0 because few players have reached them yet.

The program fits, per trophy, a weighted least-squares line of logit(rate) against ln(1 + hours since launch) on the case's history, reads the line at the target, and caps the total logit move from the last read. The spread combines a base uncertainty with the slope's standard error and the residual scatter, so trophies whose trend is poorly identified get a wider band. Rates are clamped to [0.01, 99.99] before the logit. A flat history yields the last rate with a wide spread.

## Run it

```sh
python3 main.py
```

`main.py` reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`). Standard library only, no network, deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
