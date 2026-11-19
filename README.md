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
4. **Earners/players decomposition (rich histories).** With three or more
   page states, the median also blends — in logit space, weighted higher
   the more segments and the cleaner the player fit — with a forecast that
   rebuilds the rate from its parts. Between consecutive page states, the
   marginal earn share (of the players who joined the tracked base, the
   share already holding the trophy) and the already-tracked base's per-day
   progression (the pairs' residual earners growth, decayed over the
   horizon as players finish content) are estimated per trophy and shrunk
   toward their group medians. The player base at the target comes from
   the saturating log-log fit, never projected downward and never past a
   capped multiple of the last count. Earners are floored at 98% of their
   last count and capped at the player count; at horizon zero the
   decomposition reproduces the last read exactly.
5. **Frozen-page branch.** If the whole history is one page state, no
   refresh has been observed and the dominant event inside the horizon is
   the first refresh itself: the tracked-player base can multiply several
   fold (priced from the horizon, how long the page has been frozen, and the
   base's size) while earners grow only by group-typical factors, so most
   rates fall. The branch works with earner counts, gives a wide,
   downward-skewed spread, and treats a 100% rate specially: it can only
   dilute, slowly, so its median sits just under 100 with a fat upper tail.
6. **Spread.** Per trophy, on the logit scale: a random-walk term in
   `sqrt(horizon)`, plus the observed per-refresh jump variance times the
   expected number of refreshes in the horizon, plus inflation for thin
   data, stale pages and extreme rates, plus a term for the disagreement
   between the drift and decomposition medians. The jump variance is
   estimated on recent transitions only — launch-sized jumps say little
   about the jump scale days later — and a young page with a handful of
   states keeps a sigma floor that decays over its first hours, when the
   earn curves are steepest and two page states cannot measure them. The
   downside widens when dilution dominates, the upside for fast risers.
7. **Guards.** Rates clamp to [0.01, 99.99] before any logit; logits clamp
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
8 seeds × 8 cells (lower is better), with the model's 90% coverage
(the share of trophies whose answer rate fell inside [q05, q95]):

- stale-practice 03:00, 90 min — model 0.582, persistence 0.817, curve 0.817, coverage 0.855
- stale-practice 03:30, 90 min — model 0.542, persistence 0.831, curve 0.831, coverage 0.884
- normal, day 0.15, 90 min — model 0.459, persistence 0.812, curve 0.748, coverage 0.980
- normal, day 1, 1 day — model 0.198, persistence 0.341, curve 0.210, coverage 0.982
- normal, day 1, 4.52 days — model 0.305, persistence 0.589, curve 0.614, coverage 0.982
- stale, day 1, 4.52 days — model 0.477, persistence 0.710, curve 0.712, coverage 0.957
- normal, day 2, 4.52 days — model 0.224, persistence 0.339, curve 0.320, coverage 0.982
- stale, day 2, 4.52 days — model 0.230, persistence 0.346, curve 0.312, coverage 0.982
- **overall — model 0.377, persistence 0.598, curve 0.570, coverage 0.950**

The model beats both baselines overall by ~34–37% and wins all eight
cells. The previous iteration of this entry (drift medians only) scored
0.422 overall on the same suite and *lost* the normal day-2/final cell to
the curve baseline (0.335 against 0.320) — that cell is the closest
analogue of the real final (a day-2 checkpoint, a 4.52-day horizon, a rich
history), and the earners/players decomposition is what fixed it: the same
cell now scores 0.224, 30% under the curve baseline, and the day-1/final
cell improved from 0.368 to 0.305. The configuration was also run on a
held-out seed set (9–16) with the same ordering, so the gains are not an
artefact of the tuning seeds.

On coverage: the two frozen-page cells (0.86–0.88) are the flat branch,
unchanged in this iteration. The rich-history cells sit at 0.96–0.98 on
this suite — above the nominal 0.90 — because the simulator's earn curves
are deterministic given a seed, so once the median mechanism (drift +
decomposition) is right, the remaining dispersion is mostly refresh
timing, and the bands honestly sized for a real 4.5-day horizon cover
nearly all of it. The shipped spread was narrowed from the first
iteration's (it covered 0.982 on the real practice) but deliberately kept
wider than the simulator-optimal width: fitting the simulator's
determinism would be overconfidence the real final would punish.

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
