#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for each PS5 trophy, the share of tracked players who will hold it
one week after GTA VI's release, as five quantiles of the earn rate (%).

Model (hyperbolic age model):
  * Players join over time; a trophy is earned with a probability that rises
    with the *average age* of the player base, A(t) = (integral of N dt) / N.
  * The tracked-player count N(t) is extrapolated with a saturating logistic
    curve fitted over the distinct Exophase snapshots (never a power law).
  * Each trophy's rate follows a saturating curve in average age,
    r(A) = r_inf * A / (A + tau), with tau chosen on a grid.
  * The spread combines a random-walk term, a refresh-jump volatility term
    (Exophase refreshes in jumps, diluting rates), the per-trophy fit error
    and the uncertainty in tau.
  * When history is flat or stale (few distinct snapshots), a stale-page
    fallback is used instead: a delayed refresh means N is about to jump,
    which dilutes rates, so the median is pulled down.

Pure stdlib, deterministic, no wall clock, no network. Reads every *.in file
in $MAKEFUN_INPUT_DIR (default /input) and writes <id>.out to
$MAKEFUN_OUTPUT_DIR (default /output). Never crashes on a case: any failure
falls back to a valid forecast.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
TAU_GRID = (1.5, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072)
NMAX_MULTS = (1.3, 1.6, 2.0, 2.5, 3.5, 5, 8, 12, 20, 40, 100)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def parse_ts(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def logit_pct(x):
    c = clamp(x, 0.01, 99.99)
    return math.log(c / (100.0 - c))


def inv_logit_pct(l):
    return 100.0 / (1.0 + math.exp(-clamp(l, -700.0, 700.0)))


def round4(x):
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def safe_exp(x):
    return math.exp(clamp(x, -700.0, 700.0))


def quantiles(r_med, sigma):
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    if not math.isfinite(r_med):
        r_med = 1.0
    lg = logit_pct(r_med)
    qs = [round4(inv_logit_pct(lg + z * sigma)) for z in Z]
    for k in range(1, 5):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return qs


def last_rate(history, idx):
    """Last non-null earn rate for trophy idx over all reads (newest first)."""
    for read in reversed(history):
        rates = read.get("rates") or []
        if idx < len(rates) and rates[idx] is not None:
            return float(rates[idx])
    return 0.01


def fit_logistic(ts, ns):
    """Fit N(t) = Nmax / (1 + exp(-k (t - tm))) over distinct reads.

    Grid over Nmax (multiples of the last N); for each, linear regression of
    ln(Nmax/N - 1) on t gives k and tm. Returns (Nmax, k, tm, sse) for the
    smallest Nmax whose SSE is within 5% of the best, or None.
    """
    n_last = ns[-1]
    if len(ts) < 3 or n_last <= 0:
        return None
    results = []
    for mult in NMAX_MULTS:
        nmax = mult * n_last
        if any(n <= 0 or n >= nmax for n in ns):
            continue
        xs = ts
        ys = [math.log(nmax / n - 1.0) for n in ns]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den <= 1e-12:
            continue
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        a = my - b * mx
        k = -b
        if k <= 0:
            continue
        tm = a / k
        sse = 0.0
        for t, n_obs in zip(ts, ns):
            n_pred = nmax / (1.0 + safe_exp(-k * (t - tm)))
            if n_pred <= 0:
                sse = float("inf")
                break
            sse += (math.log(n_pred) - math.log(n_obs)) ** 2
        sse /= n
        results.append((mult, nmax, k, tm, sse))
    if not results:
        return None
    best_sse = min(r[4] for r in results)
    for mult, nmax, k, tm, sse in results:
        if sse <= 1.05 * best_sse:
            return (nmax, k, tm, sse)
    return None


def jump_volatility(distinct, ts, n_trophies):
    """Pooled std of logit changes between consecutive distinct reads,
    each normalised by sqrt(delta days). Floored at 0.35."""
    changes = []
    for i in range(1, len(distinct)):
        dt_days = (ts[i] - ts[i - 1]) / 24.0
        if dt_days <= 0:
            continue
        scale = math.sqrt(dt_days)
        r_prev = distinct[i - 1].get("rates") or []
        r_cur = distinct[i].get("rates") or []
        for j in range(n_trophies):
            if j >= len(r_prev) or j >= len(r_cur):
                continue
            a, b = r_prev[j], r_cur[j]
            if a is None or b is None:
                continue
            changes.append((logit_pct(b) - logit_pct(a)) / scale)
    if len(changes) < 2:
        return 0.35
    mean = sum(changes) / len(changes)
    var = sum((c - mean) ** 2 for c in changes) / len(changes)
    return max(0.35, math.sqrt(var))


def fit_trophy(rates, ages):
    """Fit r(A) = r_inf * A / (A + tau) for one trophy.

    rates/ages are aligned with the distinct reads. Returns
    (tau, r_inf, sse) or None when no tau has enough usable points.
    """
    best = None
    for tau in TAU_GRID:
        pts = []
        for a, r in zip(ages, rates):
            if r is None or a is None:
                continue
            f = a / (a + tau) if (a + tau) > 0 else 0.0
            if f >= 0.1:
                pts.append((a, float(r), f))
        if len(pts) < 3:
            continue
        r_inf = median([r / f for (_a, r, f) in pts])
        r_max = max(r for (_a, r, _f) in pts)
        r_inf = clamp(r_inf, r_max, 99.9)
        sse = 0.0
        for (_a, r, f) in pts:
            pred = clamp(r_inf * f, 0.01, 99.99)
            sse += (logit_pct(pred) - logit_pct(r)) ** 2
        sse = sse / len(pts) + 0.02 * (math.log(tau / 48.0)) ** 2
        if best is None or sse < best[2]:
            best = (tau, r_inf, sse)
    return best


def forecast_case(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes", 0) or 0)
    d_days = horizon_min / 1440.0
    horizon_h = horizon_min / 60.0

    game = case.get("game") or {}
    rel = game.get("release_date")
    if rel:
        t_rel = parse_ts(str(rel) + "T00:00:00Z")
    elif history:
        t_rel = parse_ts(history[0]["slot"])
    else:
        t_rel = parse_ts(case["at"])
    t_ck = parse_ts(case["at"])
    t_tg = parse_ts(case["target"])
    age_h = (t_ck - t_rel).total_seconds() / 3600.0

    if not history:
        return {"trophies": [
            {"id": tr["id"], "q": quantiles(1.0, 1.0)} for tr in trophies]}

    # Distinct snapshots: a read counts as new only if its snapshot differs
    # from the previous read's (players_tracked, rates).
    distinct = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or []))
        if key != prev_key:
            distinct.append(read)
            prev_key = key

    ts = [(parse_ts(r["slot"]) - t_rel).total_seconds() / 3600.0
          for r in distinct]
    ns = [float(r.get("players_tracked") or 0) for r in distinct]

    # Average player age A at each distinct read: trapezoid integral of N
    # over the read slots since the first read, divided by N.
    cum = [0.0]
    for i in range(1, len(distinct)):
        dt = ts[i] - ts[i - 1]
        cum.append(cum[-1] + 0.5 * (ns[i - 1] + ns[i]) * dt)
    ages = [cum[i] / ns[i] if ns[i] > 0 else 0.0 for i in range(len(distinct))]

    t_last = ts[-1]
    n_last = ns[-1]
    stale_h = (t_ck - parse_ts(distinct[-1]["slot"])).total_seconds() / 3600.0
    if stale_h < 0:
        stale_h = 0.0

    flat = len(distinct) < 4 or (ages and max(ages) <= 1.0)

    if flat:
        # Flat / stale history regime: no age signal to fit.
        g = 0.25 * max(0.0, 1.0 - age_h / 48.0)
        e = 0.05 * max(0.0, 1.0 - age_h / 48.0)
        rows = []
        for j, tr in enumerate(trophies):
            r_last = last_rate(history, j)
            if stale_h > 1:
                med = clamp(r_last * safe_exp((e - g) * (stale_h + horizon_h)),
                            0.01, 99.99)
                sig = math.sqrt((0.5 * math.sqrt(d_days)) ** 2
                                + (0.5 * math.sqrt(stale_h + horizon_h)) ** 2)
                sig = clamp(sig, 0.3, 3.0)
            else:
                med = r_last if r_last and r_last > 0 else 0.01
                med = clamp(med, 0.01, 99.99)
                sig = 0.5 * math.sqrt(d_days) + 0.2
            rows.append({"id": tr["id"], "q": quantiles(med, sig)})
        return {"trophies": rows}

    # --- Logistic extrapolation of N ---
    fit = fit_logistic(ts, ns)
    k_eff = fit[1] if fit else 0.0
    surge = 0.0
    if k_eff < 0.005 and stale_h > 1:
        surge = 0.25 * max(0.0, 1.0 - age_h / 48.0)

    def n_at(t):
        if fit is None:
            base = n_last
        else:
            nmax, k, tm, _sse = fit
            base = nmax / (1.0 + safe_exp(-k * (t - tm)))
        if t > t_last and surge > 0:
            if fit is None:
                anchor = n_last
            else:
                nmax, k, tm, _sse = fit
                anchor = nmax / (1.0 + safe_exp(-k * (t_last - tm)))
            return anchor * safe_exp(surge * (t - t_last))
        return base

    t_target = (t_tg - t_rel).total_seconds() / 3600.0
    n_target = clamp(n_at(t_target), 0.5 * n_last, 100.0 * n_last)

    # A_target: integral of N from t_last to target (1h trapezoid steps)
    # plus the integral up to the last read, divided by N_target.
    integ = cum[-1]
    t = t_last
    while t < t_target:
        step = min(1.0, t_target - t)
        integ += 0.5 * (n_at(t) + n_at(t + step)) * step
        t += step
    a_target = integ / n_target if n_target > 0 else 0.0

    jvol = jump_volatility(distinct, ts, n_tr)

    rows = []
    for j, tr in enumerate(trophies):
        rates = []
        for read in distinct:
            rs = read.get("rates") or []
            rates.append(rs[j] if j < len(rs) else None)
        r_last = last_rate(history, j)
        fitted = fit_trophy(rates, ages)
        if fitted is None:
            med = clamp(r_last if r_last > 0 else 0.01, 0.01, 99.99)
            sig = 0.5 * math.sqrt(d_days) + 0.2
            rows.append({"id": tr["id"], "q": quantiles(med, sig)})
            continue
        tau, r_inf, sse = fitted
        r_med = clamp(r_inf * a_target / (a_target + tau), 0.01, 99.99)
        r_half = clamp(r_inf * a_target / (a_target + tau / 2.0), 0.01, 99.99)
        r_dbl = clamp(r_inf * a_target / (a_target + tau * 2.0), 0.01, 99.99)
        sig_tau = abs(logit_pct(r_half) - logit_pct(r_dbl)) * 0.25
        sig = math.sqrt((0.5 * math.sqrt(d_days)) ** 2
                        + (jvol * math.sqrt(d_days)) ** 2
                        + (2.0 * math.sqrt(max(sse, 0.0))) ** 2
                        + sig_tau ** 2)
        sig = clamp(sig, 0.30, 3.0)
        if r_last >= 99.9 and r_med >= 99.5:
            r_med = 99.95
            sig = min(sig, 0.6)
        rows.append({"id": tr["id"], "q": quantiles(r_med, sig)})
    return {"trophies": rows}


def fallback_case(case):
    """Valid forecast when the main computation fails: last rate, sigma 1."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    rows = []
    for j, tr in enumerate(trophies):
        r = last_rate(history, j)
        med = r if r and r > 0 else 1.0
        rows.append({"id": tr["id"], "q": quantiles(med, 1.0)})
    return {"trophies": rows}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(in_dir.glob("*.in")):
        case_id = in_path.stem
        try:
            with open(in_path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
        except Exception:
            continue
        try:
            result = forecast_case(case)
        except Exception:
            try:
                result = fallback_case(case)
            except Exception:
                result = {"trophies": [
                    {"id": tr["id"], "q": quantiles(1.0, 1.0)}
                    for tr in case.get("trophies", [])]}
        out_path = out_dir / (case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
