[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Endangdanngz/week-one-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Endangdanngz/week-one-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# Week-One Rarity Forecast — GTA VI trophy earn rates

A forecasting program for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).
For each case it reads the house's recording of Exophase's GTA VI (PS5)
trophy page up to the case's checkpoint and writes, for every trophy, five
quantiles (q05, q25, q50, q75, q95) of its earn rate — the percent of tracked
players holding it — at the case's target. Scoring is the mean pinball loss
on the logit scale, so the model works in logit space throughout.

## Run it

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is self-contained (Python 3.13, standard library only, no
network), deterministic, and runs in ~0.05 s per case against an 8 s limit.
It reads every `*.in` file in `$MAKEFUN_INPUT_DIR` (else `/input`), sorted
by name, and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## The model

Exophase refreshes a game's counts every few hours; between refreshes every
house read is identical. The model therefore starts from the accounting
identity `rate = earners / players`, in logit space:

1. **Segments.** The history is collapsed into distinct page states
   separated by refreshes. All dynamics are measured between segments.
2. **Groups.** Each trophy is classified from its metadata (name,
   description, tier) as story, side content, collectibles, stunts,
   completion, or ultra-hard. Groups provide shrinkage priors.
3. **Trend, two views blended by horizon.** A recency-weighted per-day
   logit drift (right for short horizons) and a log-time slope — logit rate
   against `ln(1 + hours since release)`, whose real-time effect decays as a
   saturating curve ages (right for the 4.5-day final horizon). Both are
   shrunk toward their group's median; positive shifts are scaled by the
   headroom left under 100% and capped; the median also blends in the slope
   fit's own line read at the target, which smooths rounding noise.
4. **Frozen-page branch.** If the whole history is one page state, no
   refresh has been observed and the dominant event inside the horizon is
   the first refresh itself: the tracked-player base can multiply several
   fold (priced from the horizon, how long the page has been frozen, and the
   base's size) while earners grow only by group-typical factors, so most
   rates fall. The branch works with earner counts, gives a wide,
   downward-skewed spread, and treats a 100% rate specially: it can only
   dilute, slowly, so its median sits just under 100 with a fat upper tail.
5. **Spread.** Per trophy, on the logit scale: a random-walk term in
   `sqrt(horizon)`, plus the observed per-refresh jump variance (recency
   weighted, shrunk to a robust pooled value) times the expected number of
   refreshes in the horizon, plus inflation for thin data, stale pages and
   extreme rates. The downside widens when dilution dominates, the upside
   for fast risers.
6. **Guards.** Rates clamp to [0.01, 99.99] before any logit; logits clamp
   to [-10, 10]; the platinum's median is softly capped at 1.05× the best
   gold median. Any failure inside a case degrades that case to persistence
   with a wide spread, then to tier priors — the program always writes a
   valid output.

All tunable constants live in one `CONFIG` dict at the top of `main.py`
(the balanced configuration). Conservative and aggressive variants —
damping, sigma scale, flat-branch spreads — are documented in comments
there.

## Backtest

`backtest.py` (standard library only; never run by the grader) implements
the official pinball-on-logit evaluator, both house baselines (persistence
and the logit-curve fit), and a synthetic launch simulator: the tracked
player base grows 638 → ~150k over seven days, each trophy's earners follow
a saturating curve whose speed varies by group and by a per-seed speed
factor, and the page freezes between refreshes every 2–4 h while the house
reads it every 30 min. Two regimes: a normal launch, and a "stale launch"
whose page is frozen at the real launch-night state until the first refresh
— the situation the practice cases were cut from. Mean pinball loss over
8 seeds × 8 cells (lower is better):

- stale-practice 03:00, 90 min — model 0.582, persistence 0.817, curve 0.817
- stale-practice 03:30, 90 min — model 0.542, persistence 0.831, curve 0.831
- normal, day 0.15, 90 min — model 0.572, persistence 0.812, curve 0.748
- normal, day 1, 1 day — model 0.216, persistence 0.341, curve 0.210
- normal, day 1, 4.52 days — model 0.368, persistence 0.589, curve 0.614
- stale, day 1, 4.52 days — model 0.496, persistence 0.710, curve 0.712
- normal, day 2, 4.52 days — model 0.335, persistence 0.339, curve 0.320
- stale, day 2, 4.52 days — model 0.261, persistence 0.346, curve 0.312
- **overall — model 0.422, persistence 0.598, curve 0.570**

The model beats both baselines overall by ~26–30% and wins six of eight
cells; the two narrow losses (day-1/1-day and normal day-2/final, 3–5%
behind the curve baseline) are that baseline's home turf — smooth,
well-observed histories — and no cell is lost by more than 5%.

On the kit's real practice inputs (whose answers are not published), the
program writes valid 55-row outputs (2,782 / 2,790 bytes), byte-identical
across runs, with every median below the last read's rate — the dilution
the first refresh brings — and bands wide enough to contain a ×6 player
jump. `test_edge.py` runs the entry point over twelve hostile cases (empty
or missing history, one read, all-zero rates, a 100% rate, nulls, unsorted
or truncated reads, malformed entries, zero trophies) and validates every
output's shape; all pass.

## Layout

- `main.py` — the entry: parser, segment dynamics, model, fallbacks.
- `backtest.py` — evaluator, house baselines, synthetic launch simulator.
- `test_edge.py` — edge-case battery over the real entry point.
- `LICENSE` — MIT.
