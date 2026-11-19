"""Backtest for the week-one rarity forecast (not run by the grader).

Three parts:

1. An evaluator implementing the official score: mean pinball loss on the
   logit scale over every trophy and quantile level, with rates clamped to
   [0.01, 99.99] before the logit.

2. The two house baselines as reference forecasters: persistence (median =
   last read's rate, random-walk spread 0.5*sqrt(days)) and the curve
   baseline (least squares of logit rate on ln(1 + hours since release) over
   the last 24 h of reads, read at the target, spread = random walk and the
   fit's residual combined; fewer than 3 reads falls back to persistence).

3. A synthetic launch simulator. It invents a plausible Exophase recording:
   the tracked-player base grows from 638 to ~150k over seven days (anchor
   points interpolated in log space), each trophy's earners follow a
   saturating curve whose speed and ceiling depend on the trophy's group,
   earners grow more slowly than the player base early on (so rates dilute
   when the base jumps), and the page only changes at refreshes every 2-4 h,
   freezing between them while the house reads it every 30 minutes. Two
   regimes: "normal" (refreshes from the start) and "stale launch" (the page
   frozen at the real launch-night state - 638 players, first trophy at
   100%, most others near 1% - until the first refresh at 04:30, exactly the
   situation the practice cases were cut from).

Cases are cut from the simulated recordings at several checkpoints and
horizons, every forecaster is scored on each, and the table at the end
compares the model against both baselines cell by cell and reports the
model's 90% coverage (share of answers inside [q05, q95]) per cell. The
model must beat both baselines overall and stay competitive in every cell
before its CONFIG is accepted; the final-like cells (day-2 checkpoint,
4.52-day horizon) carry the most weight because the real final has that
shape.

Usage: python3 backtest.py [--seeds N] [--sweep]
"""

import argparse
import importlib.util
import json
import math
import random
import sys
import time
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT_INPUT = (HERE.parent / "kits" / "gta6-trophy-rarity-forecast-01"
             / "gta6-trophy-rarity-forecast-01-kit" / "input")
RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

spec = importlib.util.spec_from_file_location("entry_main", HERE / "main.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


# ---------------------------------------------------------------------------
# Official score
# ---------------------------------------------------------------------------

def clamp_rate(x):
    return min(99.99, max(0.01, float(x)))


def logit_pct(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def pinball_loss(rows, answer):
    """rows: {trophy_id: [q05..q95]}; answer: {trophy_id: rate}. Mean loss."""
    total = 0.0
    count = 0
    for tid, qs in rows.items():
        if tid not in answer or answer[tid] is None:
            continue
        ly = logit_pct(answer[tid])
        for tau, q in zip(LEVELS, qs):
            d = ly - logit_pct(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            count += 1
    return total / count if count else float("nan")


def rows_of(forecast):
    return {row["id"]: row["q"] for row in forecast["trophies"]}


def coverage90(rows, answer):
    """Share of scored trophies whose answer rate lies inside [q05, q95]."""
    inside = 0
    total = 0
    for tid, qs in rows.items():
        if tid not in answer or answer[tid] is None:
            continue
        total += 1
        if qs[0] <= answer[tid] <= qs[-1]:
            inside += 1
    return inside / total if total else float("nan")


# ---------------------------------------------------------------------------
# House baselines (reference forecasters, from the published formulas)
# ---------------------------------------------------------------------------

def last_rates(case):
    hist = case["history"]
    n = len(case["trophies"])
    out = [None] * n
    for read in hist:
        for i in range(n):
            if read["rates"][i] is not None:
                out[i] = read["rates"][i]
    return out


def persistence_rows(case):
    days = case["horizon_minutes"] / 1440.0
    spread = 0.5 * math.sqrt(days)
    rows = {}
    for trophy, rate in zip(case["trophies"], last_rates(case)):
        lg = logit_pct(rate if rate is not None else 0.0)
        rows[trophy["id"]] = [round(100.0 / (1.0 + math.exp(-(lg + z * spread))), 4) for z in Z]
    return rows


def _ts(text):
    return datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()


def curve_rows(case):
    """The house curve baseline: logit-linear in ln(1 + hours since release)."""
    days = case["horizon_minutes"] / 1440.0
    rows = persistence_rows(case)
    cp = _ts(case["at"])
    target = _ts(case["target"])
    window = [r for r in case["history"] if _ts(r["at"]) >= cp - 24 * 3600]
    for i, trophy in enumerate(case["trophies"]):
        pts = []
        for read in window:
            rate = read["rates"][i]
            if rate is None:
                continue
            h = max((_ts(read["at"]) - RELEASE.timestamp()) / 3600.0, 0.0)
            pts.append((math.log1p(h), logit_pct(rate)))
        if len(pts) < 3:
            continue
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        if sxx <= 1e-12:
            continue
        slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
        intercept = my - slope * mx
        resid = sum((p[1] - (intercept + slope * p[0])) ** 2 for p in pts) / n
        h_t = max((target - RELEASE.timestamp()) / 3600.0, 0.0)
        pred = intercept + slope * math.log1p(h_t)
        pred = min(10.0, max(-10.0, pred))
        sigma = math.sqrt(0.25 * days + resid)
        rows[trophy["id"]] = [round(100.0 / (1.0 + math.exp(-(pred + z * sigma))), 4) for z in Z]
    return rows


def model_rows(case):
    return rows_of(entry.forecast_case_safe(case))


# ---------------------------------------------------------------------------
# Synthetic launch simulator
# ---------------------------------------------------------------------------

# (t_days, players) anchors for the tracked-player base, log-interpolated.
PLAYER_ANCHORS = [(0.0, 638.0), (0.179, 3868.0), (0.5, 12000.0), (1.0, 30000.0),
                  (2.0, 62000.0), (3.5, 95000.0), (5.0, 125000.0), (7.0, 150000.0)]

# Per-group earner-curve ranges: (day-7 rate % low/high, tau days low/high,
# earner-vs-player growth exponent gamma low/high). Tau is divided by a
# per-seed speed factor (0.75 / 1.0 / 1.4) so the suite spans slow and fast
# launches and tuning cannot overfit to one saturation speed.
GROUP_PARAMS = {
    "story":       ((14.0, 58.0), (0.9, 3.4), (0.74, 0.92)),
    "side":        ((4.0, 12.0), (1.8, 3.8), (0.70, 0.90)),
    "collectibles": ((2.5, 8.0), (2.6, 5.0), (0.68, 0.88)),
    "stunts":      ((1.5, 6.0), (2.4, 4.6), (0.68, 0.88)),
    "completion":  ((0.8, 0.8), (5.5, 5.5), (0.85, 0.90)),
    "ultra":       ((0.8, 2.5), (4.0, 6.0), (0.70, 0.88)),
}


def players_at(t_days):
    if t_days <= PLAYER_ANCHORS[0][0]:
        return PLAYER_ANCHORS[0][1]
    for (t0, p0), (t1, p1) in zip(PLAYER_ANCHORS, PLAYER_ANCHORS[1:]):
        if t_days <= t1:
            f = (t_days - t0) / (t1 - t0)
            return math.exp(math.log(p0) + f * (math.log(p1) - math.log(p0)))
    return PLAYER_ANCHORS[-1][1]


class SimTrophy:
    def __init__(self, meta, rng, speed=1.0):
        self.meta = meta
        self.group = entry.classify_trophy(meta)
        (r7lo, r7hi), (taulo, tauhi), (glo, ghi) = GROUP_PARAMS[self.group]
        self.r7 = rng.uniform(r7lo, r7hi)
        self.tau = rng.uniform(taulo, tauhi) / speed
        self.gamma = rng.uniform(glo, ghi)
        if meta["id"] == 1:      # platinum: a sliver even after a week
            self.r7, self.tau, self.gamma = 0.12, 7.0 / speed, 0.9
        p7 = players_at(7.0)
        shape7 = 1.0 - math.exp(-((7.0 / self.tau) ** 1.25))
        self.a = (self.r7 / 100.0) * p7 ** (1.0 - self.gamma) / shape7

    def earners(self, t_days):
        if self.meta["id"] == 2:
            # The opening mission: essentially every tracked player has it.
            return players_at(t_days) * (1.0 - 0.006 * math.exp(-t_days / 0.4))
        shape = 1.0 - math.exp(-((max(t_days, 0.0) / self.tau) ** 1.25))
        return self.a * shape * players_at(t_days) ** self.gamma


def simulate(trophies_meta, seed, regime):
    """Returns (reads, truth_at) where reads are house-style page reads."""
    rng = random.Random(seed)
    speed = rng.choice([0.75, 1.0, 1.4])
    sims = [SimTrophy(m, rng, speed) for m in trophies_meta]
    frozen_rates = None
    frozen_players = 638.0
    if regime == "stale":
        practice = json.loads((KIT_INPUT / "p00.in").read_text())
        frozen_rates = practice["history"][-1]["rates"]
    # Refresh clock.
    refreshes = []
    t = 0.179 if regime == "stale" else 0.06
    while t < 8.0:
        refreshes.append(t)
        t += rng.uniform(2.0, 4.0) / 24.0

    def page_state(t_days):
        """(players, rates) the page shows at t_days: truth at last refresh."""
        if regime == "stale" and (not refreshes or t_days < refreshes[0]):
            return frozen_players, list(frozen_rates)
        last_ref = 0.05
        for r in refreshes:
            if r <= t_days:
                last_ref = r
            else:
                break
        players = players_at(last_ref)
        rates = []
        for s in sims:
            e = s.earners(last_ref)
            if regime == "stale":
                # The launch-night pool keeps earning a little on its own.
                e0 = (frozen_rates[s.meta["id"] - 1] or 0.0) / 100.0 * frozen_players
                e = max(e, e0 * (1.0 + 0.5 * (1.0 - math.exp(-last_ref / 1.0))))
            rates.append(round(min(100.0 * e / players, 100.0), 2))
        return players, rates

    reads = []
    t = -0.0069  # first house read: 2026-11-18T23:50Z
    while t < 8.0:
        players, rates = page_state(max(t, 0.0))
        when = RELEASE + timedelta(days=t)
        reads.append({
            "slot": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "at": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "players_tracked": int(round(players)),
            "trophies_earned": int(round(sum((r or 0.0) / 100.0 * players for r in rates))),
            "rates": rates,
        })
        t += 30.0 / 1440.0
    return reads


def cut_case(reads, trophies_meta, case_id, checkpoint, horizon_minutes):
    cp_ts = checkpoint.timestamp()
    history = [r for r in reads if _ts(r["at"]) < cp_ts]
    target = checkpoint + timedelta(minutes=horizon_minutes)
    answer = None
    for r in reads:
        if _ts(r["at"]) >= target.timestamp():
            answer = {t["id"]: r["rates"][i] for i, t in enumerate(trophies_meta)}
            break
    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": case_id,
        "at": checkpoint.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": horizon_minutes,
        "target": target.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "levels": LEVELS,
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19",
                 "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"},
        "trophies": trophies_meta,
        "history": history,
        "notes": "synthetic backtest case",
    }
    return case, answer


CELLS = [
    # (name, regime, checkpoint (days after release), horizon minutes)
    ("stale-practice-0300", "stale", 3.0 / 24.0, 90),
    ("stale-practice-0330", "stale", 3.5 / 24.0, 90),
    ("normal-early-90m", "normal", 0.15, 90),
    ("normal-day1-1d", "normal", 1.0, 1440),
    ("normal-day1-final", "normal", 1.0, 6510),
    ("stale-day1-final", "stale", 1.0, 6510),
    ("normal-day2-final", "normal", 2.0, 6510),
    ("stale-day2-final", "stale", 2.0, 6510),
]


def run_suite(seeds):
    trophies_meta = json.loads((KIT_INPUT / "p00.in").read_text())["trophies"]
    results = {}
    for name, regime, cp_days, horizon in CELLS:
        sums = {"model": 0.0, "persistence": 0.0, "curve": 0.0}
        cov_sum = 0.0
        used = 0
        for seed in seeds:
            reads = simulate(trophies_meta, seed * 1000 + zlib.crc32(name.encode()) % 997, regime)
            checkpoint = RELEASE + timedelta(days=cp_days)
            case, answer = cut_case(reads, trophies_meta, name, checkpoint, horizon)
            if answer is None or not case["history"]:
                continue
            mrows = model_rows(case)
            sums["model"] += pinball_loss(mrows, answer)
            sums["persistence"] += pinball_loss(persistence_rows(case), answer)
            sums["curve"] += pinball_loss(curve_rows(case), answer)
            cov_sum += coverage90(mrows, answer)
            used += 1
        if used:
            results[name] = {k: v / used for k, v in sums.items()}
            results[name]["coverage"] = cov_sum / used
    return results


def print_table(results):
    print(f"{'cell':<22} {'model':>8} {'persist':>8} {'curve':>8} {'cov90':>6}  verdict")
    tot = {"model": 0.0, "persistence": 0.0, "curve": 0.0}
    cov_tot = 0.0
    for name, r in results.items():
        best_base = min(r["persistence"], r["curve"])
        verdict = "WIN " if r["model"] < best_base else "lose"
        print(f"{name:<22} {r['model']:>8.4f} {r['persistence']:>8.4f} {r['curve']:>8.4f} "
              f"{r.get('coverage', float('nan')):>6.3f}  {verdict}")
        for k in tot:
            tot[k] += r[k]
        cov_tot += r.get("coverage", 0.0)
    n = max(len(results), 1)
    print(f"{'OVERALL (mean)':<22} {tot['model']/n:>8.4f} {tot['persistence']/n:>8.4f} "
          f"{tot['curve']/n:>8.4f} {cov_tot/n:>6.3f}")


def sweep(seeds):
    """Coarse grid over the main CONFIG knobs, restoring values afterwards."""
    base = dict(entry.CONFIG)
    grids = [
        ("damp", [0.45, 0.68, 0.9]),
        ("rw_sigma", [0.4, 0.5, 0.65]),
        ("sigma_scale", [0.85, 1.0, 1.2]),
        ("max_dlogit", [1.9, 2.5, 3.1]),
        ("flat_sigma_dn", [1.1, 1.3, 1.55]),
    ]
    for key, values in grids:
        for v in values:
            entry.CONFIG[key] = v
            res = run_suite(seeds)
            mean_model = sum(r["model"] for r in res.values()) / len(res)
            mean_base = sum(min(r["persistence"], r["curve"]) for r in res.values()) / len(res)
            print(f"{key}={v}: model {mean_model:.4f} vs best-baseline {mean_base:.4f}")
        entry.CONFIG[key] = base[key]
    entry.CONFIG.clear()
    entry.CONFIG.update(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    seeds = list(range(1, args.seeds + 1))
    t0 = time.time()
    if args.sweep:
        sweep(seeds)
    results = run_suite(seeds)
    print_table(results)
    print(f"(suite ran in {time.time() - t0:.1f}s over seeds {seeds})")


if __name__ == "__main__":
    main()
