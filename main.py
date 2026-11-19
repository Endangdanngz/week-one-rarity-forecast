"""Week-one rarity forecast for GTA VI trophies (make.fun gta6-trophy-rarity-forecast-01).

What this program does
----------------------
For every case file (<id>.in) in the input directory it writes a forecast file
(<id>.out) holding, for each of the case's trophies, five quantiles
(q05, q25, q50, q75, q95) of the trophy's Exophase earn rate (percent of tracked
players who hold it) at the case's target instant. Scoring is the mean pinball
loss on the logit scale, so all modelling happens in logit space and the
quantiles are read off a per-trophy logit distribution at the end.

Why the model looks the way it does
-----------------------------------
Exophase does not update a game's page continuously: its counts refresh every
few hours and every house read between two refreshes is byte-identical. A case
history is therefore a step function, and the first thing the model does is
collapse the reads into the distinct page states ("segments") separated by
refreshes. Everything else follows from the accounting identity

    rate = earners / players,

read in logit space:

* Per-trophy drift: the logit change per day between consecutive segments,
  recency-weighted, shrunk toward the median drift of the trophy's group
  (story / side / collectibles / stunts / completion / ultra-hard), then
  damped over the forecast horizon and capped, so a hot launch slope is never
  extrapolated linearly for four and a half days.
* Denominator growth: a log-log fit of players-tracked against hours since
  release projects how much the player base grows by the target. In the
  normal branch the observed drifts already contain the dilution that growth
  causes, so the projection is used only for spread sizing and for the
  special branch below.
* Frozen-page branch: if the whole history is a single state, no refresh has
  been observed yet and the dominant event inside the horizon is the first
  refresh itself: a small launch-night denominator can multiply several-fold
  while earners grow only by group-typical factors, so most rates fall. The
  branch prices that jump explicitly (earners factor vs players factor) with
  a wide, downward-skewed spread.
* Spread: a random-walk term in sqrt(horizon), plus the observed per-refresh
  jump variance times the expected number of refreshes in the horizon, plus
  inflation for thin data, stale pages and extreme rates. The two tails use
  different sigmas: the downside widens when dilution dominates, the upside
  when a trophy is rising fast.

Guards: rates are clamped to [0.01, 99.99] before any logit, logits are
clamped to [-10, 10], the median shift is capped, a 100% rate is never pinned
at 100 (it can only dilute), a 0% rate keeps an upward tail, and the
platinum's median is softly capped by the gold trophies' medians. Any failure
inside a case falls back to persistence with a wide spread for that case; the
program always writes a valid output for every input it can read.

Runs offline, standard library only, deterministic. Input directory is
$MAKEFUN_INPUT_DIR (else /input), output $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
RELEASE_FALLBACK = "2026-11-19"  # GTA VI's release date, if a case omits it.

# --------------------------------------------------------------------------
# Tunable constants (the "balanced" configuration).
#
# Variant notes (the parent may submit a later commit with these swapped):
#   conservative: damp 0.45, tau_damp_days 2.2, max_dlogit 1.9,
#                 rw_sigma 0.52, sigma_scale 1.05, flat sigma_dn 1.55 / up 1.15
#   aggressive:   damp 0.85, tau_damp_days 4.5, max_dlogit 3.1,
#                 rw_sigma 0.38, sigma_scale 0.72, flat sigma_dn 1.10 / up 0.80
# --------------------------------------------------------------------------
CONFIG = {
    # Drift extrapolation: median shift = drift * damp * H / (1 + H / tau),
    # H = horizon in days; the rational form saturates long horizons.
    "damp": 0.60,
    "tau_damp_days": 2.0,
    "max_dlogit": 2.5,          # cap on |median logit shift|, normal branch
    # Shrinkage of a trophy's drift toward its group's median drift.
    "shrink_k": 2.0,            # pseudo-count weight of the group median
    "drift_halflife_days": 1.25,  # recency weighting of segment transitions
    "slope_halflife_days": 0.55,  # recency weighting of the log-time slope fit
    # Spread, normal branch (logit units).
    "rw_sigma": 0.44,           # random-walk sigma per sqrt(day)
    "jump_w": 0.5,              # weight of observed per-refresh jump variance
    "jump_shrink": 6.0,         # pseudo-count shrinking jump var to pooled
    "thin_sigma": 0.30,         # added sigma when a trophy has no transitions
    "sigma_scale": 0.85,        # global multiplier on the normal-branch sigma
    "sigma_min": 0.15,
    "sigma_max": 2.4,
    "max_refreshes": 5.0,       # cap on expected refreshes inside horizon
    # Tail asymmetry (normal branch).
    "asym_down": 1.25,          # sigma_down multiplier when drift < 0
    "asym_up": 1.20,            # sigma_up multiplier when drift is strongly > 0
    # Frozen-page (single-state history) branch.
    "flat_fp_c0": 0.55,         # median ln(players factor) =
    "flat_fp_c1": 0.55,         #   c0 + c1*ln(1+horizon_hours)
    "flat_fp_c2": 0.45,         #        + c2*ln(1+frozen_hours), damped below
    "flat_fp_max": 5.52,        # cap on ln(players factor) (~ x250)
    "flat_base_ref": 2000.0,    # players base where growth damping starts
    "flat_anchor_lnfp": 1.80,   # ln(players factor) the earner factors anchor to
    "flat_ln_earners": {        # median ln(earners factor) at the anchor, by group
        "story": 1.10,
        "side": 0.80,
        "collectibles": 0.60,
        "stunts": 0.60,
        "completion": 0.25,
        "ultra": 0.35,
    },
    "flat_hi_rate_blend": 0.97,  # earners/players factor for a ~100% trophy
    "flat_sigma_dn": 1.30,
    "flat_sigma_up": 0.95,
    # Fallbacks.
    "tier_prior_rate": {"platinum": 0.05, "gold": 3.0, "silver": 1.5, "bronze": 5.0},
    "prior_sigma": 2.2,
}

_NORM = NormalDist()


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def clamp_rate(pct):
    """Clamp a percent rate the way the scorer does before taking logits."""
    return clamp(float(pct), 0.01, 99.99)


def logit_pct(pct):
    p = clamp_rate(pct) / 100.0
    return clamp(math.log(p / (1.0 - p)), -10.0, 10.0)


def inv_logit_pct(lg):
    lg = clamp(float(lg), -10.0, 10.0)
    return clamp(100.0 / (1.0 + math.exp(-lg)), 0.01, 99.99)


def median_of(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def parse_ts(text):
    """ISO-8601 timestamp -> epoch seconds (UTC), or None."""
    if not text:
        return None
    txt = str(text).strip()
    if txt.endswith(("Z", "z")):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def round4(x):
    return round(float(x), 4)


# --------------------------------------------------------------------------
# Trophy grouping (metadata only: name / description / tier / points / secret)
# --------------------------------------------------------------------------

def classify_trophy(trophy):
    name = str(trophy.get("name", "")).lower()
    desc = str(trophy.get("description", "")).lower()
    tier = str(trophy.get("tier", "")).lower()
    if tier == "platinum" or "100%" in desc:
        return "completion"
    if "chapter" in desc or "final mission." in desc or "gone fishing" in desc:
        return "story"
    if "stunt" in desc or "shooting range" in desc or "gold medal in all" in desc:
        return "ultra" if ("250" in desc or "all" in desc) else "stunts"
    if "race" in desc or "arcade" in desc or "fishing" in desc or "challenges" in desc \
            or "activity" in desc or "activities" in desc:
        return "stunts"
    if "collectible" in desc or "locations" in desc or "discover" in desc \
            or "meditate" in desc or "explore all" in desc or "purchase all" in desc \
            or "register" in desc or "every spot" in desc:
        return "collectibles"
    if desc.startswith("complete ") and "final mission for" not in desc \
            and "strands" not in desc and "strangers" not in name:
        return "story"
    return "side"


# --------------------------------------------------------------------------
# History handling: reads -> segments (distinct page states)
# --------------------------------------------------------------------------

class Segment:
    __slots__ = ("t", "players", "rates")

    def __init__(self, t, players, rates):
        self.t = t                # epoch seconds of the first read with this state
        self.players = players    # players tracked (float) or None
        self.rates = rates        # tuple of rates (float or None), trophy order


def build_segments(history, n_trophies):
    """Sort reads by time and merge consecutive identical page states."""
    reads = []
    for read in history or []:
        if not isinstance(read, dict):
            continue
        t = parse_ts(read.get("at")) or parse_ts(read.get("slot"))
        if t is None:
            continue
        raw_rates = read.get("rates") or []
        rates = []
        for i in range(n_trophies):
            v = raw_rates[i] if i < len(raw_rates) else None
            if v is None:
                rates.append(None)
            else:
                try:
                    rates.append(float(v))
                except (TypeError, ValueError):
                    rates.append(None)
        players = read.get("players_tracked")
        try:
            players = float(players) if players is not None else None
        except (TypeError, ValueError):
            players = None
        reads.append((t, players, tuple(rates)))
    reads.sort(key=lambda r: r[0])
    segments = []
    last_players = None
    for t, players, rates in reads:
        if players is None:
            players = last_players
        else:
            last_players = players
        if segments and segments[-1].players == players and segments[-1].rates == rates:
            continue  # same page state as the previous read: not a refresh
        segments.append(Segment(t, players, rates))
    return segments, reads


def carried_rates(segments, n_trophies):
    """carried[i][s] = trophy i's last non-null rate at or before segment s."""
    carried = [[None] * len(segments) for _ in range(n_trophies)]
    for i in range(n_trophies):
        last = None
        for s, seg in enumerate(segments):
            if seg.rates[i] is not None:
                last = seg.rates[i]
            carried[i][s] = last
    return carried


# --------------------------------------------------------------------------
# Players growth: log-log fit of players vs hours since release
# --------------------------------------------------------------------------

def fit_players(segments, release_ts):
    """Least squares of ln(players) on ln(1 + hours since release).

    Returns (slope, intercept) or None when fewer than two usable segments.
    """
    pts = []
    for seg in segments:
        if seg.players and seg.players > 0:
            h = max((seg.t - release_ts) / 3600.0, -0.9)
            pts.append((math.log1p(h), math.log(seg.players)))
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 1e-12:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    slope = sxy / sxx
    return slope, my - slope * mx


def project_players(fit, last_players, hours_target):
    """Projected players at the target hour; never below 80% of the last read."""
    if fit is None or not last_players:
        return last_players
    slope, intercept = fit
    ln_p = intercept + slope * math.log1p(max(hours_target, -0.9))
    proj = math.exp(clamp(ln_p, -5.0, 20.0))
    return max(proj, 0.8 * last_players)


# --------------------------------------------------------------------------
# Per-trophy drift across segments, with group shrinkage
# --------------------------------------------------------------------------

def trophy_drifts(segments, carried, groups, cfg):
    """Recency-weighted per-day logit drift per trophy + jump statistics.

    Returns (drift, jump_var, n_trans) lists indexed by trophy.
    """
    n = len(carried)
    n_seg = len(segments)
    drift = [0.0] * n
    n_trans = [0] * n
    jump_num = [0.0] * n   # recency-weighted sum of squared logit jumps
    jump_den = [0.0] * n
    if n_seg >= 2:
        t_last = segments[-1].t
        for i in range(n):
            num = 0.0
            den = 0.0
            for s in range(n_seg - 1):
                a = carried[i][s]
                b = carried[i][s + 1]
                if a is None or b is None:
                    continue
                dt_days = (segments[s + 1].t - segments[s].t) / 86400.0
                if dt_days <= 1e-6:
                    continue
                djump = logit_pct(b) - logit_pct(a)
                age_days = (t_last - 0.5 * (segments[s].t + segments[s + 1].t)) / 86400.0
                w = math.sqrt(max(dt_days, 0.02)) * 0.5 ** (age_days / cfg["drift_halflife_days"])
                num += w * (djump / dt_days)
                den += w
                jump_num[i] += w * djump * djump
                jump_den[i] += w
                n_trans[i] += 1
            if den > 0:
                drift[i] = num / den
    # Pooled per-refresh jump variance: median across trophies of their
    # recency-weighted mean squared jump, so neither the explosive launch
    # phase nor a few wild trophies set the scale for everyone.
    raw_jv = [(jump_num[i] / jump_den[i]) if jump_den[i] > 0 else None for i in range(n)]
    pooled_var = 1.5 * (median_of([v for v in raw_jv if v is not None]) or 0.04)
    jump_var = []
    for i in range(n):
        k = cfg["jump_shrink"]
        if raw_jv[i] is not None:
            jump_var.append((n_trans[i] * raw_jv[i] + k * pooled_var) / (n_trans[i] + k))
        else:
            jump_var.append(pooled_var)
    # Shrink each drift toward its group's median drift.
    by_group = {}
    for i in range(n):
        if n_trans[i] > 0:
            by_group.setdefault(groups[i], []).append(drift[i])
    group_med = {g: median_of(v) for g, v in by_group.items()}
    all_med = median_of([drift[i] for i in range(n) if n_trans[i] > 0])
    shrunk = []
    for i in range(n):
        g = group_med.get(groups[i], all_med)
        if g is None or n_trans[i] == 0:
            shrunk.append(0.0)
            continue
        w = float(n_trans[i])
        shrunk.append((w * drift[i] + cfg["shrink_k"] * g) / (w + cfg["shrink_k"]))
    return shrunk, jump_var, n_trans, pooled_var


def logtime_slopes(segments, carried, groups, release_ts, hours_target, cfg):
    """Per-trophy slope of logit rate against ln(1 + hours since release).

    A saturating earn curve is roughly linear on this axis, and its slope in
    real time decays like 1/(1 + hours), so extrapolating the slope to the
    target builds the deceleration in instead of projecting the recent
    per-day drift unchanged. Slopes are shrunk toward the group median.
    Also returns each fit's own line read at the target hour (or None): the
    line smooths the last read's rounding noise, so the median blends the
    anchored extrapolation with the line's reading.
    """
    n = len(carried)
    slopes = [0.0] * n
    used = [0] * n
    center = [None] * n   # (mx, my) of each trophy's weighted fit
    if len(segments) >= 2 and release_ts:
        t_last = segments[-1].t
        for i in range(n):
            pts = []
            for s, seg in enumerate(segments):
                if carried[i][s] is None:
                    continue
                h = max((seg.t - release_ts) / 3600.0, -0.9)
                age_days = (t_last - seg.t) / 86400.0
                w = 0.5 ** (age_days / cfg["slope_halflife_days"])
                pts.append((math.log1p(h), logit_pct(carried[i][s]), w))
            if len(pts) >= 2:
                sw = sum(p[2] for p in pts)
                mx = sum(p[2] * p[0] for p in pts) / sw
                my = sum(p[2] * p[1] for p in pts) / sw
                sxx = sum(p[2] * (p[0] - mx) ** 2 for p in pts)
                if sxx > 1e-9:
                    slopes[i] = sum(p[2] * (p[0] - mx) * (p[1] - my) for p in pts) / sxx
                    used[i] = len(pts)
                    center[i] = (mx, my)
    by_group = {}
    for i in range(n):
        if used[i] > 0:
            by_group.setdefault(groups[i], []).append(slopes[i])
    group_med = {g: median_of(v) for g, v in by_group.items()}
    all_med = median_of([slopes[i] for i in range(n) if used[i] > 0])
    out = []
    preds = []
    x_target = math.log1p(max(hours_target, -0.9))
    for i in range(n):
        g = group_med.get(groups[i], all_med)
        if g is None or used[i] == 0:
            out.append(0.0)
            preds.append(None)
        else:
            w = float(used[i])
            s = (w * slopes[i] + cfg["shrink_k"] * g) / (w + cfg["shrink_k"])
            out.append(s)
            mx, my = center[i]
            preds.append(clamp(my + s * (x_target - mx), -10.0, 10.0))
    return out, preds


# --------------------------------------------------------------------------
# Quantile row construction
# --------------------------------------------------------------------------

def quantile_row(median_logit, sigma_dn, sigma_up, zscores):
    qs = []
    for z in zscores:
        sigma = sigma_dn if z < 0 else sigma_up if z > 0 else 0.0
        qs.append(inv_logit_pct(median_logit + z * sigma))
    qs = [round4(q) for q in qs]
    for k in range(1, len(qs)):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return qs


# --------------------------------------------------------------------------
# Frozen-page branch: the whole history is one page state
# --------------------------------------------------------------------------

def flat_branch_row(rate, group, players0, horizon_hours, frozen_hours,
                    hours_since_release, cfg, zscores):
    """Forecast when no refresh has been observed yet.

    The first refresh inside the horizon multiplies the tracked-player base;
    earners grow by a group-typical factor. Work with earner counts so rates
    near 100% behave (their earners can only track the player base).
    """
    if rate is None or not players0 or players0 <= 0:
        return None
    scale = math.sqrt(1.0 + 0.8 * (horizon_hours / 24.0))
    if rate >= 99.995:
        # Printed as 100%: every tracked player holds it. It can only dilute,
        # and slowly (it is the opening mission), so keep the median just
        # under 100 with a fat upper tail: undershooting a rate that stays at
        # 100 is the expensive error on the logit scale, overshooting is not.
        return quantile_row(logit_pct(99.9), 0.85 * scale, 1.15 * scale, zscores)
    ln_fp = (cfg["flat_fp_c0"]
             + cfg["flat_fp_c1"] * math.log1p(max(horizon_hours, 0.0))
             + cfg["flat_fp_c2"] * math.log1p(max(frozen_hours, 0.0)))
    base_damp = min(1.0, (cfg["flat_base_ref"] / players0) ** 0.3) if players0 > cfg["flat_base_ref"] else 1.0
    age_damp = 1.0 / (1.0 + max(0.0, hours_since_release - 6.0) / 36.0)
    ln_fp = clamp(ln_fp * base_damp * age_damp, 0.0, cfg["flat_fp_max"])
    ln_fe = cfg["flat_ln_earners"].get(group, 0.7) * (ln_fp / cfg["flat_anchor_lnfp"])
    # A trophy almost everyone holds: its earners grow with the player base.
    w_hi = clamp((rate - 60.0) / 40.0, 0.0, 1.0)
    if w_hi > 0.0:
        ln_fe = (1.0 - w_hi) * ln_fe + w_hi * (ln_fp + math.log(cfg["flat_hi_rate_blend"]))
    earners0 = clamp_rate(rate) / 100.0 * players0
    earners1 = earners0 * math.exp(clamp(ln_fe, -2.0, 6.0))
    players1 = players0 * math.exp(ln_fp)
    median_rate = clamp(100.0 * earners1 / players1, 0.01, 99.99) if players1 > 0 else clamp_rate(rate)
    sigma_dn = cfg["flat_sigma_dn"] * scale
    sigma_up = cfg["flat_sigma_up"] * scale
    return quantile_row(logit_pct(median_rate), sigma_dn, sigma_up, zscores)


# --------------------------------------------------------------------------
# Fallbacks: tier prior, and persistence with a wide spread
# --------------------------------------------------------------------------

def tier_prior_row(trophy, horizon_days, cfg, zscores):
    tier = str(trophy.get("tier", "bronze")).lower()
    rate = cfg["tier_prior_rate"].get(tier, 2.0)
    sigma = cfg["prior_sigma"] * math.sqrt(1.0 + 0.35 * horizon_days)
    return quantile_row(logit_pct(rate), sigma, sigma, zscores)


def persistence_wide_row(rate, horizon_days, cfg, zscores):
    sigma = 0.5 * math.sqrt(max(horizon_days, 0.01)) * 2.2 + 0.4
    return quantile_row(logit_pct(rate if rate is not None else 0.0), sigma, sigma, zscores)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

def forecast_case(case):
    cfg = CONFIG
    trophies = case.get("trophies") or []
    n = len(trophies)
    levels = case.get("levels") or DEFAULT_LEVELS
    try:
        zscores = [_NORM.inv_cdf(float(t)) for t in levels]
    except Exception:
        zscores = [_NORM.inv_cdf(t) for t in DEFAULT_LEVELS]
    horizon_min = float(case.get("horizon_minutes") or 0.0)
    horizon_days = horizon_min / 1440.0
    horizon_hours = horizon_min / 60.0

    checkpoint = parse_ts(case.get("at"))
    target_ts = parse_ts(case.get("target"))
    if target_ts is None and checkpoint is not None:
        target_ts = checkpoint + horizon_min * 60.0
    release_ts = parse_ts(str((case.get("game") or {}).get("release_date", RELEASE_FALLBACK)) + "T00:00:00Z")
    if release_ts is None:
        release_ts = parse_ts(RELEASE_FALLBACK + "T00:00:00Z")

    segments, reads = build_segments(case.get("history"), n)
    if n == 0:
        return {"trophies": []}
    if not segments:
        return {"trophies": [{"id": t.get("id"), "q": tier_prior_row(t, horizon_days, cfg, zscores)}
                             for t in trophies]}

    groups = [classify_trophy(t) for t in trophies]
    carried = carried_rates(segments, n)
    last_rate = [carried[i][-1] for i in range(n)]
    last_seg = segments[-1]
    last_players = last_seg.players
    hours_cp = ((checkpoint - release_ts) / 3600.0) if (checkpoint is not None and release_ts) else 0.0
    hours_target = ((target_ts - release_ts) / 3600.0) if (target_ts is not None and release_ts) else hours_cp + horizon_hours

    # --- Frozen-page branch: a single distinct state in the whole history.
    if len(segments) == 1:
        first_t = reads[0][0] if reads else last_seg.t
        frozen_hours = ((checkpoint - first_t) / 3600.0) if checkpoint is not None else 0.0
        rows = []
        for i, trophy in enumerate(trophies):
            row = None
            if last_players:
                row = flat_branch_row(last_rate[i], groups[i], last_players, horizon_hours,
                                      max(frozen_hours, 0.0), hours_cp, cfg, zscores)
            if row is None:
                row = persistence_wide_row(last_rate[i], horizon_days, cfg, zscores) \
                    if last_rate[i] is not None else tier_prior_row(trophy, horizon_days, cfg, zscores)
            rows.append({"id": trophy.get("id"), "q": row})
        return {"trophies": rows}

    # --- Normal branch: drift + spread from the segment dynamics.
    drift, jump_var, n_trans, _pooled = trophy_drifts(segments, carried, groups, cfg)
    slopes, line_preds = logtime_slopes(segments, carried, groups, release_ts, hours_target, cfg)
    fit = fit_players(segments, release_ts)
    proj_players = project_players(fit, last_players, hours_target)

    span_days = max((segments[-1].t - segments[0].t) / 86400.0, 0.02)
    refresh_per_day = len(segments) / span_days
    exp_refreshes = min(refresh_per_day * horizon_days, cfg["max_refreshes"])
    stale_days = 0.0
    if checkpoint is not None:
        stale_days = clamp((checkpoint - last_seg.t) / 86400.0, 0.0, 3.0)
    drift_factor = cfg["damp"] * horizon_days / (1.0 + horizon_days / cfg["tau_damp_days"]) \
        if horizon_days > 0 else 0.0

    medians = [None] * n
    rows_q = [None] * n
    for i, trophy in enumerate(trophies):
        if last_rate[i] is None:
            rows_q[i] = tier_prior_row(trophy, horizon_days, cfg, zscores)
            continue
        base_logit = logit_pct(last_rate[i])
        weight = n_trans[i] / (n_trans[i] + 1.5)
        # Two views of the trend, blended by horizon: the recent per-day
        # drift (right for short horizons) and the log-time slope, whose
        # real-time effect decays as the curve ages (right for long ones).
        h_last = max((last_seg.t - release_ts) / 3600.0, -0.9) if release_ts else 0.0
        dln = math.log1p(max(hours_target, -0.9)) - math.log1p(h_last)
        shift_lt = slopes[i] * max(dln, 0.0)
        shift_drift = drift[i] * drift_factor
        w_lt = horizon_days / (horizon_days + 0.55) if horizon_days > 0 else 0.0
        shift = w_lt * shift_lt + (1.0 - w_lt) * shift_drift
        if shift > 0.0:
            # Headroom: a curve already near its ceiling cannot keep rising.
            shift *= clamp(1.0 - last_rate[i] / 100.0, 0.2, 1.0)
        shift = clamp(shift * weight, -cfg["max_dlogit"], cfg["max_dlogit"])
        med_logit = clamp(base_logit + shift, -10.0, 10.0)
        if line_preds[i] is not None and n_trans[i] >= 2:
            # Blend in the fit line's own reading at the target, kept near
            # the anchored median so a wild line cannot take over.
            line = clamp(line_preds[i], med_logit - 1.2, med_logit + 1.2)
            med_logit = clamp(0.55 * med_logit + 0.45 * line, -10.0, 10.0)
        medians[i] = med_logit
        var = (cfg["rw_sigma"] ** 2) * horizon_days \
            + cfg["jump_w"] * jump_var[i] * exp_refreshes
        sigma = math.sqrt(max(var, 0.0)) * cfg["sigma_scale"]
        sigma += cfg["thin_sigma"] / (1.0 + n_trans[i])
        sigma *= 1.0 + 0.12 * stale_days
        if last_rate[i] <= 0.05 or last_rate[i] >= 99.9:
            sigma *= 1.20
        elif last_rate[i] <= 0.5:
            sigma *= 1.10
        sigma = clamp(sigma, cfg["sigma_min"], cfg["sigma_max"])
        sigma_dn = sigma_up = sigma
        if drift[i] < -0.05:
            sigma_dn = sigma * cfg["asym_down"]
            sigma_up = sigma * 0.95
        elif drift[i] > 0.15:
            sigma_up = sigma * cfg["asym_up"]
            sigma_dn = sigma * 0.95
        rows_q[i] = (med_logit, clamp(sigma_dn, cfg["sigma_min"], cfg["sigma_max"]),
                     clamp(sigma_up, cfg["sigma_min"], cfg["sigma_max"]))

    # Soft cap: the platinum's median stays within 5% of the best gold median.
    gold_medians = [medians[i] for i in range(n)
                    if str(trophies[i].get("tier", "")).lower() == "gold" and medians[i] is not None]
    if gold_medians:
        cap_rate = inv_logit_pct(max(gold_medians)) * 1.05
        cap_logit = logit_pct(min(cap_rate, 99.99))
        for i in range(n):
            if str(trophies[i].get("tier", "")).lower() == "platinum" and medians[i] is not None \
                    and medians[i] > cap_logit:
                med_logit, s_dn, s_up = rows_q[i]
                rows_q[i] = (cap_logit, s_dn, s_up)
                medians[i] = cap_logit

    rows = []
    for i, trophy in enumerate(trophies):
        if isinstance(rows_q[i], list):
            q = rows_q[i]
        else:
            med_logit, s_dn, s_up = rows_q[i]
            q = quantile_row(med_logit, s_dn, s_up, zscores)
        rows.append({"id": trophy.get("id"), "q": q})
    return {"trophies": rows}


def forecast_case_safe(case):
    """Full model, degrading to persistence-wide, then to tier priors."""
    try:
        return forecast_case(case)
    except Exception:
        pass
    trophies = case.get("trophies") or []
    horizon_days = float(case.get("horizon_minutes") or 0.0) / 1440.0
    zscores = [_NORM.inv_cdf(t) for t in DEFAULT_LEVELS]
    try:
        segments, _reads = build_segments(case.get("history"), len(trophies))
        carried = carried_rates(segments, len(trophies)) if segments else None
        rows = []
        for i, trophy in enumerate(trophies):
            rate = carried[i][-1] if carried else None
            if rate is None:
                q = tier_prior_row(trophy, horizon_days, CONFIG, zscores)
            else:
                q = persistence_wide_row(rate, horizon_days, CONFIG, zscores)
            rows.append({"id": trophy.get("id"), "q": q})
        return {"trophies": rows}
    except Exception:
        return {"trophies": [{"id": t.get("id"),
                              "q": tier_prior_row(t, horizon_days, CONFIG, zscores)}
                             for t in trophies]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.is_dir():
        return
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        forecast = forecast_case_safe(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
