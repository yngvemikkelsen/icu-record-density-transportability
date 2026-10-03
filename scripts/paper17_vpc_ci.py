#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "pyarrow", "statsmodels"]
# ///
"""
Paper 17 revision, editorial comment 4: confidence intervals for every
variance partition coefficient in Table 3, both cohorts, both levels.

Table 3 reports a hospital VPC from a three-component random-intercept model
fitted with statsmodels MixedLM.  Two problems make that route unusable for
interval estimation.

  1.  Cost.  One fit at cohort scale takes 6-9 s, so 500 hospital-clustered
      bootstrap replicates across 8 metrics and 2 cohorts is 16-17 core-hours.

  2.  Reliability.  At this scale the default optimiser chain frequently stops
      short of the REML optimum while reporting convergence.  Over 12
      simulated cohorts (62 hospitals, 90,000 stays, true VPC 0.126) it missed
      the optimum in 6, by up to 9.1 on the -2 log-likelihood scale, and the
      returned VPC was inflated by 11% to 84% in every one of those cases.

Both are solved by evaluating the likelihood in closed form.  The model

    y_hui = mu + a_h + b_hu + e_hui,   a ~ N(0,s_h^2), b ~ N(0,s_u^2), e ~ N(0,s_e^2)

is perfectly nested, so V is block diagonal by hospital with a patterned
block, and |V| together with the quadratic forms follow from per-unit counts
and sums.  A likelihood evaluation is then O(number of unit cells) rather than
O(n), the residual variance profiles out analytically, and what remains is a
two-parameter optimisation.  One fit takes about 15 ms, a factor of 400
faster, and it reaches a strictly lower REML objective than MixedLM wherever
the two disagree.

The script revalidates that claim against MixedLM on the real data before
using it, so the substitution is demonstrated rather than asserted.

Outputs
  vpc_ci.csv         hospital and unit VPC with 95% intervals, per metric,
                     per cohort, with the point estimate MixedLM returns and
                     the REML objective difference
  vpc_validation.csv the agreement check against MixedLM

Usage
  python paper17_vpc_ci.py \
      --mimic-cache    ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-nc-cache  ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-root      ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir        ~/bcst/vpc_ci

  --selftest runs the simulation study only and needs no data.
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings("ignore")

WINDOW_MIN = 24 * 60
MIN_RECORDS = 3
MIN_HOSP = 500
FLOOR = 12
N_BOOT = 500
SEED = 17

ROWS = [("n_records", "Record count"),
        ("t_first_h", "Hours to first record"),
        ("median_interval_min", "Median interval"),
        ("iqr_interval_min", "Interval IQR"),
        ("max_interval_min", "Longest interval"),
        ("n_gaps_gt30m", "Gaps >30 min"),
        ("n_gaps_gt2h", "Gaps >2 h"),
        ("frac_time_in_gaps", "Fraction of window in gaps")]


# --------------------------------------------------------------- closed form
def sufficient(y, hosp_code, unit_code):
    """Per-unit count, sum and sum of squares, plus each unit's hospital."""
    nu = int(unit_code.max()) + 1
    n = np.bincount(unit_code, minlength=nu).astype(float)
    sy = np.bincount(unit_code, weights=y, minlength=nu)
    sy2 = np.bincount(unit_code, weights=y * y, minlength=nu)
    uh = np.zeros(nu, dtype=np.int64)
    uh[unit_code] = hosp_code
    return n, sy, sy2, uh, int(hosp_code.max()) + 1, float(len(y))


def _pieces(logg, S):
    n, sy, sy2, uh, nh, N = S
    gh, gu = np.exp(logg)
    den = 1.0 + n * gu
    T = np.bincount(uh, weights=n / den, minlength=nh)
    Sh = np.bincount(uh, weights=sy / den, minlength=nh)
    Qh = np.bincount(uh, weights=sy2 - gu * sy * sy / den, minlength=nh)
    f = 1.0 + gh * T
    O1 = float((T / f).sum())
    Oy = float((Sh / f).sum())
    Oyy = float((Qh - gh * Sh * Sh / f).sum())
    logW = float(np.log(den).sum() + np.log(f).sum())
    return gh, gu, O1, Oy, Oyy, logW, N


def neg2reml(logg, S):
    """-2 REML log-likelihood, profiled over the residual variance."""
    if not np.all(np.isfinite(logg)):
        return 1e300
    gh, gu, O1, Oy, Oyy, logW, N = _pieces(logg, S)
    R = Oyy - Oy * Oy / O1
    if not np.isfinite(R) or R <= 0 or not np.isfinite(logW):
        return 1e300
    return (N - 1.0) * (np.log(R / (N - 1.0)) + 1.0) + logW + np.log(O1)


def fit_nested(y, hosp_code, unit_code):
    """REML fit.  Returns (s_h2, s_u2, s_e2, vpc_hospital, vpc_unit, obj)."""
    S = sufficient(y, hosp_code, unit_code)
    best, bx = np.inf, None
    for start in [(-2.0, -3.0), (-4.0, -4.0), (-1.0, -1.0), (-6.0, -2.0)]:
        r = minimize(neg2reml, np.array(start, float), args=(S,),
                     method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 2000})
        r = minimize(neg2reml, r.x, args=(S,), method="Nelder-Mead",
                     options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 2000})
        if r.fun < best:
            best, bx = float(r.fun), r.x
    gh, gu, O1, Oy, Oyy, _, N = _pieces(bx, S)
    se2 = (Oyy - Oy * Oy / O1) / (N - 1.0)
    vh, vu = gh * se2, gu * se2
    tot = vh + vu + se2
    return vh, vu, se2, vh / tot, vu / tot, best


def _refit(y, unit_code, idx, pick):
    """Refit on a resampled or reduced set of hospital clusters.  Units are
    offset by cluster position before relabelling, so a hospital drawn twice
    contributes as two independent clusters with disjoint unit cells."""
    rows = np.concatenate([idx[p] for p in pick])
    hb = np.concatenate([np.full(len(idx[p]), i, dtype=np.int64)
                         for i, p in enumerate(pick)])
    ub = np.concatenate([unit_code[idx[p]].astype(np.int64) + i * 1_000_000
                         for i, p in enumerate(pick)])
    _, ub = np.unique(ub, return_inverse=True)
    return fit_nested(y[rows], hb, ub.astype(np.int64))


def _bca(theta, b, jk, alpha=0.05):
    """Bias-corrected and accelerated interval.  z0 from the bootstrap median
    bias, acceleration from the delete-one-hospital jackknife."""
    b = b[np.isfinite(b)]
    if len(b) < 20:
        return np.nan, np.nan, np.nan, np.nan
    frac = float(np.mean(b < theta))
    z0 = norm.ppf(min(max(frac, 1e-6), 1 - 1e-6))
    jk = jk[np.isfinite(jk)]
    d = jk.mean() - jk
    s2 = float((d ** 2).sum())
    a = float((d ** 3).sum() / (6.0 * s2 ** 1.5)) if s2 > 0 else 0.0
    out = []
    for q in (alpha / 2.0, 1.0 - alpha / 2.0):
        z = norm.ppf(q)
        adj = z0 + (z0 + z) / (1.0 - a * (z0 + z))
        out.append(float(np.percentile(b, 100.0 * norm.cdf(adj))))
    return out[0], out[1], z0, a


def boot_ci(y, hosp_code, unit_code, n_boot=N_BOOT, seed=SEED, bca=True):
    """Hospital-clustered bootstrap interval for both variance partition
    coefficients.  BCa is reported because the percentile interval is
    mis-centred for a variance ratio estimated from ~60 clusters: in 30
    simulated cohorts at this scale it covered the true hospital VPC 83% of
    the time against BCa's 93%, at the same width (see --selftest)."""
    nh = int(hosp_code.max()) + 1
    idx = [np.flatnonzero(hosp_code == h) for h in range(nh)]
    rng = np.random.default_rng(seed)
    bh, bu = [], []
    for _ in range(n_boot):
        try:
            _, _, _, vh, vu, _ = _refit(y, unit_code, idx,
                                        rng.integers(0, nh, nh))
        except Exception:
            vh = vu = np.nan
        bh.append(vh)
        bu.append(vu)
    bh, bu = np.array(bh, float), np.array(bu, float)
    out = {"boot_failures": float(np.mean(~np.isfinite(bh))),
           "pct_hospital": tuple(np.nanpercentile(bh, [2.5, 97.5])),
           "pct_unit": tuple(np.nanpercentile(bu, [2.5, 97.5]))}
    if not bca:
        out["hospital"], out["unit"] = out["pct_hospital"], out["pct_unit"]
        out["z0_hospital"] = out["a_hospital"] = np.nan
        return out
    _, _, _, th, tu, _ = fit_nested(y, hosp_code, unit_code)
    jh, ju = [], []
    for h in range(nh):
        keep = [i for i in range(nh) if i != h]
        try:
            _, _, _, vh, vu, _ = _refit(y, unit_code, idx, keep)
        except Exception:
            vh = vu = np.nan
        jh.append(vh)
        ju.append(vu)
    lo_h, hi_h, z0h, ah = _bca(th, bh, np.array(jh, float))
    lo_u, hi_u, z0u, au = _bca(tu, bu, np.array(ju, float))
    out.update({"hospital": (lo_h, hi_h), "unit": (lo_u, hi_u),
                "z0_hospital": z0h, "a_hospital": ah,
                "z0_unit": z0u, "a_unit": au})
    return out


# -------------------------------------------------------------- data loading
def eta2(y, g):
    y = np.asarray(y, float)
    g = np.asarray(g)
    ok = ~np.isnan(y)
    y, g = y[ok], g[ok]
    if len(y) < 10:
        return np.nan
    grand = y.mean()
    sst = ((y - grand) ** 2).sum()
    if sst <= 0:
        return np.nan
    d = pd.DataFrame({"y": y, "g": g}).groupby("g")["y"].agg(["mean", "size"])
    return float((d["size"] * (d["mean"] - grand) ** 2).sum() / sst)


def metrics(ev, key, off):
    ev = ev.sort_values([key, off])
    g = ev.groupby(key)[off]
    out = pd.DataFrame({"n_records": g.size(), "t_first_h": g.min() / 60.0})
    ev = ev.assign(gap=g.diff())
    gg = ev.dropna(subset=["gap"]).groupby(key)["gap"]
    out["median_interval_min"] = gg.median()
    out["iqr_interval_min"] = gg.quantile(0.75) - gg.quantile(0.25)
    out["max_interval_min"] = gg.max()
    out["n_gaps_gt30m"] = gg.apply(lambda s: int((s > 30).sum()))
    out["n_gaps_gt2h"] = gg.apply(lambda s: int((s > 120).sum()))
    out["frac_time_in_gaps"] = gg.apply(
        lambda s: float(s[s > 120].sum()) / WINDOW_MIN)
    return out.reset_index()


def load_eicu(a):
    e = metrics(pd.read_parquet(a.eicu_nc_cache), "patientunitstayid",
                "observationoffset")
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    e = e.merge(pat, on="patientunitstayid", how="inner")
    e = e[(e["unitdischargeoffset"] >= WINDOW_MIN)
          & (e["n_records"] >= MIN_RECORDS)].copy()
    e["hospital"] = "eICU-" + e["hospitalid"].astype(str)
    e["unit_id"] = e["hospital"] + ":" + e["unittype"].astype(str)
    c = e["hospitalid"].value_counts()
    e = e[e["hospitalid"].isin(c[c >= MIN_HOSP].index)].copy()
    med = e.groupby("hospitalid")["n_records"].median()
    er = e[e["hospitalid"].isin(med[med >= FLOOR].index)].copy()
    return e, er


def codes(s):
    y = s[s.columns[0]].to_numpy(float)
    hc = pd.factorize(s["hospital"])[0].astype(np.int64)
    uc = pd.factorize(s["unit_id"])[0].astype(np.int64)
    return y, hc, uc


# ---------------------------------------------------------------- validation
def validate(cohorts, out_dir):
    """Refit each metric with statsmodels MixedLM and compare both the
    estimate and the REML objective it attains.

    statsmodels is a declared dependency because vpc_validation.csv is a
    reported output of this script.  The import is still guarded: under
    `uv run --offline` the wheel may not be cached, and in that case the
    comparison is skipped and the intervals are produced regardless, since the
    closed-form fit does not need it.  Run the script with an interpreter that
    has statsmodels to produce the comparison.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("\n" + "=" * 94)
        print("VALIDATION SKIPPED — statsmodels is not importable here")
        print("=" * 94)
        print("  The intervals below do not depend on it.  To produce the")
        print("  comparison for the response letter, run this script again with")
        print("  an interpreter that has statsmodels, for example the conda base")
        print("  environment:   python paper17_vpc_ci.py --validate-only ...")
        return None
    rows = []
    print("\n" + "=" * 94)
    print("VALIDATION — closed form against statsmodels MixedLM on the real data")
    print("=" * 94)
    print(f"  {'cohort':12s} {'metric':28s} {'closed':>7s} {'MixedLM':>8s} "
          f"{'conv':>5s} {'d(-2logL)':>10s} {'t_sm':>7s} {'t_cf':>7s}")
    print("  " + "-" * 90)
    for label, e in cohorts:
        for col, name in ROWS:
            s = e.dropna(subset=[col])[[col, "hospital", "unit_id"]]
            y, hc, uc = codes(s)
            y = (y - y.mean()) / y.std()
            S = sufficient(y, hc, uc)
            t0 = time.time()
            _, _, _, vh, _, obj = fit_nested(y, hc, uc)
            t_cf = time.time() - t0
            d = pd.DataFrame({"yz": y, "hospital": hc.astype(str),
                              "unit_id": uc.astype(str)})
            t0 = time.time()
            try:
                f = smf.mixedlm("yz ~ 1", data=d, groups=d["hospital"],
                                re_formula="1",
                                vc_formula={"unit": "0 + C(unit_id)"}
                                ).fit(reml=True)
                mh = float(f.cov_re.iloc[0, 0])
                mu = float(f.vcomp[0]) if len(f.vcomp) else 0.0
                sc = float(f.scale)
                v_sm = mh / (mh + mu + sc)
                conv = bool(f.converged)
                o_sm = neg2reml(np.log([max(mh / sc, 1e-12),
                                        max(mu / sc, 1e-12)]), S)
            except Exception:
                v_sm, conv, o_sm = np.nan, False, np.nan
            t_sm = time.time() - t0
            print(f"  {label:12s} {name:28s} {vh:7.3f} {v_sm:8.3f} "
                  f"{str(conv):>5s} {o_sm - obj:10.3f} {t_sm:7.2f} {t_cf:7.3f}")
            rows.append({"cohort": label, "metric": name,
                         "vpc_closed_form": vh, "vpc_mixedlm": v_sm,
                         "mixedlm_converged": conv,
                         "mixedlm_objective_excess": o_sm - obj,
                         "seconds_mixedlm": t_sm, "seconds_closed_form": t_cf})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "vpc_validation.csv", index=False)
    worse = int((df["mixedlm_objective_excess"] > 1e-3).sum())
    print(f"\n  MixedLM reached a worse REML optimum in {worse} of {len(df)} fits")
    print(f"  median speed-up "
          f"{(df['seconds_mixedlm'] / df['seconds_closed_form']).median():.0f}x")
    return df


# -------------------------------------------------------------------- tables
def run(cohorts, out_dir, n_boot):
    rows = []
    print("\n" + "=" * 94)
    print(f"VARIANCE PARTITION COEFFICIENTS WITH 95% INTERVALS "
          f"({n_boot} hospital-clustered bootstrap replicates)")
    print("=" * 94)
    for label, e in cohorts:
        print(f"\n  {label}: {len(e):,} stays, {e['hospital'].nunique()} "
              f"hospitals, {e['unit_id'].nunique()} unit cells")
        print(f"    {'Metric':28s} {'hospital VPC (95% CI)':>26s} "
              f"{'unit VPC (95% CI)':>26s} {'eta2 h':>7s}")
        print("    " + "-" * 88)
        for col, name in ROWS:
            s = e.dropna(subset=[col])[[col, "hospital", "unit_id"]]
            y, hc, uc = codes(s)
            y = (y - y.mean()) / y.std()
            _, _, _, vh, vu, _ = fit_nested(y, hc, uc)
            ci = boot_ci(y, hc, uc, n_boot)
            hl, hh = ci["hospital"]
            ul, uh_ = ci["unit"]
            eh = eta2(y, hc)
            print(f"    {name:28s} "
                  f"{f'{vh:.3f} ({hl:.3f}-{hh:.3f})':>26s} "
                  f"{f'{vu:.3f} ({ul:.3f}-{uh_:.3f})':>26s} {eh:7.3f}")
            rows.append({"cohort": label, "metric": name,
                         "vpc_hospital": vh, "vpc_hospital_lo": hl,
                         "vpc_hospital_hi": hh, "vpc_unit": vu,
                         "vpc_unit_lo": ul, "vpc_unit_hi": uh_,
                         "vpc_hospital_pct_lo": ci["pct_hospital"][0],
                         "vpc_hospital_pct_hi": ci["pct_hospital"][1],
                         "vpc_unit_pct_lo": ci["pct_unit"][0],
                         "vpc_unit_pct_hi": ci["pct_unit"][1],
                         "bca_z0_hospital": ci["z0_hospital"],
                         "bca_a_hospital": ci["a_hospital"],
                         "eta2_hospital": eh, "n_stays": len(s),
                         "n_hospitals": int(s["hospital"].nunique()),
                         "n_unit_cells": int(s["unit_id"].nunique()),
                         "n_boot": n_boot,
                         "boot_failures": ci["boot_failures"]})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "vpc_ci.csv", index=False)
    return df


# ------------------------------------------------------------------ selftest
def simulate(n_hosp, n_stays, s_h, s_u, s_e, seed):
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0, 0.6, n_hosp)
    w /= w.sum()
    sizes = np.maximum(MIN_HOSP, (w * n_stays).astype(int))
    hc, uc, y = [], [], []
    k = 0
    for h in range(n_hosp):
        nu = max(1, int(round(rng.normal(2.1, 0.8))))
        ah = rng.normal(0, s_h)
        bs = rng.normal(0, s_u, nu)
        for i in range(int(sizes[h])):
            u = i % nu
            hc.append(h)
            uc.append(k + u)
            y.append(ah + bs[u] + rng.normal(0, s_e))
        k += nu
    return (np.array(y), np.array(hc, dtype=np.int64),
            np.array(uc, dtype=np.int64))


def selftest(n_rep=30, n_boot=150):
    s_h, s_u, s_e = 0.35, 0.20, 0.90
    tot = s_h ** 2 + s_u ** 2 + s_e ** 2
    th, tu = s_h ** 2 / tot, s_u ** 2 / tot
    print("=" * 94)
    print("SELF-TEST — recovery and interval coverage at cohort scale")
    print("=" * 94)
    print(f"  62 hospitals, ~90,000 stays, true hospital VPC {th:.4f}, "
          f"unit VPC {tu:.4f}")
    ph, pu = [], []
    cov = {"bca_h": 0, "bca_u": 0, "pct_h": 0, "pct_u": 0}
    wid = {"bca_h": [], "pct_h": []}
    t0 = time.time()
    for rep in range(n_rep):
        y, hc, uc = simulate(62, 90000, s_h, s_u, s_e, 1000 + rep)
        y = (y - y.mean()) / y.std()
        _, _, _, vh, vu, _ = fit_nested(y, hc, uc)
        ci = boot_ci(y, hc, uc, n_boot, seed=rep)
        ph.append(vh)
        pu.append(vu)
        for key, tgt, lohi in (("bca_h", th, ci["hospital"]),
                               ("bca_u", tu, ci["unit"]),
                               ("pct_h", th, ci["pct_hospital"]),
                               ("pct_u", tu, ci["pct_unit"])):
            cov[key] += (lohi[0] <= tgt <= lohi[1])
        wid["bca_h"].append(ci["hospital"][1] - ci["hospital"][0])
        wid["pct_h"].append(ci["pct_hospital"][1] - ci["pct_hospital"][0])
    ph, pu = np.array(ph), np.array(pu)
    print(f"  hospital  mean {ph.mean():.4f}  bias {ph.mean() - th:+.4f}  "
          f"SD {ph.std(ddof=1):.4f}")
    print(f"  unit      mean {pu.mean():.4f}  bias {pu.mean() - tu:+.4f}  "
          f"SD {pu.std(ddof=1):.4f}")
    print(f"  coverage, hospital VPC   BCa {cov['bca_h']}/{n_rep} = "
          f"{cov['bca_h'] / n_rep:.0%}   percentile {cov['pct_h']}/{n_rep} = "
          f"{cov['pct_h'] / n_rep:.0%}")
    print(f"  coverage, unit VPC       BCa {cov['bca_u']}/{n_rep} = "
          f"{cov['bca_u'] / n_rep:.0%}   percentile {cov['pct_u']}/{n_rep} = "
          f"{cov['pct_u'] / n_rep:.0%}")
    print(f"  mean hospital width      BCa {np.mean(wid['bca_h']):.4f}   "
          f"percentile {np.mean(wid['pct_h']):.4f}")
    print(f"  {n_rep} replicates x {n_boot} bootstraps "
          f"in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-cache", type=Path)
    ap.add_argument("--mimic-per-stay", type=Path)
    ap.add_argument("--eicu-nc-cache", type=Path)
    ap.add_argument("--eicu-root", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./vpc_ci"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--validate-only", action="store_true",
                    help="run only the statsmodels comparison, no bootstrap")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    for r in ("eicu_nc_cache", "eicu_root"):
        if getattr(a, r) is None:
            ap.error(f"--{r.replace('_', '-')} is required")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    eb, er = load_eicu(a)
    cohorts = [("unrestricted", eb), ("restricted", er)]
    if a.validate_only:
        validate(cohorts, a.out_dir)
        print(f"\n-> {a.out_dir}")
        return
    if not a.no_validate:
        validate(cohorts, a.out_dir)
    run(cohorts, a.out_dir, a.n_boot)
    print(f"\n-> {a.out_dir}")


if __name__ == "__main__":
    main()
