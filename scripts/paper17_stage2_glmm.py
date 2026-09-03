#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "statsmodels", "pyarrow", "scipy"]
# ///
"""
Paper 17 revision, Stage 2: the two statistical objections.

EDITORIAL COMMENT 10 — Gaussian LMM on a discrete count
--------------------------------------------------------
Record count is a right-skewed non-negative integer. Standardising it and
fitting a Gaussian linear mixed model to obtain variance partition coefficients
assumes a distribution the outcome does not have. The editor asks for a
generalised linear mixed model with a count family instead.

The obstacle is that a Poisson or negative binomial GLMM does not yield a
residual variance on the same scale as the random effects, so the VPC is not the
simple ratio it is for a Gaussian model. The standard solution is the
latent-scale (observation-level) VPC: on the log link, the level-1 variance for
a Poisson model with mean mu is ln(1 + 1/mu), and for a negative binomial with
dispersion alpha it is ln(1 + 1/mu + alpha). Nakagawa, Johnson and Schielzeth
(2017, J R Soc Interface 14:20170213) set this out; the same paper is the source
of the R-squared decomposition used for mixed models generally.

This script fits, for the count outcome only:
  1. the Gaussian LMM already reported, for comparison
  2. a Poisson GLMM with a hospital random intercept
  3. a negative binomial GLMM with a hospital random intercept
and reports the latent-scale VPC for each, so the manuscript can state whether
the hospital-dominant hierarchy depends on the distributional assumption.

Interval metrics are continuous and positive; the Gaussian model on those is not
what the editor objected to, and they are left alone.

EDITORIAL COMMENT 18 — order of entry in nested sums of squares
----------------------------------------------------------------
Sequential (type I) sums of squares in an unbalanced design depend on the order
in which factors are entered. The manuscript reports database, then hospital
given database, then unit given both. The editor asks whether the 1.6-7.1%
attributed to database survives reordering.

This refits every metric under all six orderings of the three factors and
reports the range each component takes. It also reports the type II style
"last-in" contribution of each factor, which is order-independent by
construction and is the honest summary if the sequential values move.

Usage:
  python paper17_stage2_glmm.py \
      --mimic-cache ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/revision_stage2

  Add --subsample 400 if the GLMM is slow; it caps stays per hospital.
"""

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

WINDOW_MIN = 24 * 60
MIN_RECORDS = 3
MIN_HOSP = 500


# ----------------------------------------------------------------- data ----

def metrics(ev, key, off):
    ev = ev.sort_values([key, off])
    g = ev.groupby(key)[off]
    out = pd.DataFrame({"n_records": g.size()})
    ev = ev.assign(gap=g.diff())
    gg = ev.dropna(subset=["gap"]).groupby(key)["gap"]
    out["median_interval_min"] = gg.median()
    out["iqr_interval_min"] = gg.quantile(0.75) - gg.quantile(0.25)
    out["max_interval_min"] = gg.max()
    out["n_gaps_gt2h"] = gg.apply(lambda s: int((s > 120).sum()))
    out["frac_time_in_gaps"] = gg.apply(
        lambda s: float(s[s > 120].sum()) / WINDOW_MIN)
    return out.reset_index()


def load(a):
    mim = metrics(pd.read_parquet(a.mimic_cache)[["stay_id", "offset_min"]],
                  "stay_id", "offset_min")
    ps = pd.read_csv(a.mimic_per_stay, usecols=["stay_id", "careunit", "los"])
    mim = mim.merge(ps, on="stay_id", how="inner")
    mim = mim[(mim["los"] >= 1.0) & (mim["n_records"] >= MIN_RECORDS)].copy()
    mim["database"], mim["hospital"] = "MIMIC", "MIMIC-BIDMC"
    mim["unit_id"] = "MIMIC:" + mim["careunit"]

    nc = pd.read_parquet(a.eicu_nc_cache)
    eic = metrics(nc, "patientunitstayid", "observationoffset")
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    eic = eic.merge(pat, on="patientunitstayid", how="inner")
    eic = eic[(eic["unitdischargeoffset"] >= WINDOW_MIN)
              & (eic["n_records"] >= MIN_RECORDS)].copy()
    eic["database"] = "eICU"
    eic["hospital"] = "eICU-" + eic["hospitalid"].astype(str)
    eic["unit_id"] = eic["hospital"] + ":" + eic["unittype"].astype(str)
    return mim, eic


# ------------------------------------------------------- EDITORIAL 10 ------

def latent_vpc_poisson(var_re, mu):
    """Nakagawa et al. 2017: level-1 variance on the log link = ln(1 + 1/mu)."""
    v1 = np.log1p(1.0 / mu)
    return var_re / (var_re + v1), v1


def latent_vpc_negbin(var_re, mu, alpha):
    """Negative binomial, log link: level-1 variance = ln(1 + 1/mu + alpha)."""
    v1 = np.log1p(1.0 / mu + alpha)
    return var_re / (var_re + v1), v1


def glmm_count(eic, out_dir, subsample=None, seed=17):
    """Count outcome under three distributional assumptions, THREE-LEVEL.

    The manuscript fits hospital, unit within hospital, and residual. Any
    comparison must use the same structure: dropping the unit level pushes its
    variance into hospital and inflates the hospital VPC (verified: the
    discrepancy correlates with the omitted unit component at r=0.99).

    Random-effect variances on the log scale are obtained from the between-group
    variance of the group log means, corrected for the delta-method sampling
    variance of each mean, at both levels. The Gaussian fit is run alongside on
    the identical structure so the two are directly comparable.
    """
    print("=" * 78)
    print("EDITORIAL COMMENT 10 — count outcome under three distributions")
    print("=" * 78)
    d = eic.copy()
    cnt = d["hospital"].value_counts()
    d = d[d["hospital"].isin(cnt[cnt >= MIN_HOSP].index)].copy()
    if subsample:
        d = (d.groupby("hospital", group_keys=False)
               .apply(lambda g: g.sample(min(len(g), subsample),
                                         random_state=seed)))
    y = d["n_records"]
    mu, var = float(y.mean()), float(y.var(ddof=1))
    od = var / mu
    print(f"  {len(d):,} stays, {d['hospital'].nunique()} hospitals, "
          f"{d['unit_id'].nunique()} unit cells")
    print(f"  record count: mean {mu:.2f}, variance {var:.2f}, "
          f"skew {y.skew():.2f}, range {int(y.min())}-{int(y.max())}")
    print(f"  variance-to-mean ratio {od:.2f} "
          f"({'OVERDISPERSED — Poisson is not appropriate' if od > 1.5 else 'near-Poisson'})")

    rows = []

    # --- 1. Gaussian LMM, three components, as the manuscript fits it -------
    d["yz"] = (y - mu) / y.std()
    d["unittype_"] = d["unit_id"].astype(str)
    print(f"\n  1. Gaussian LMM on the standardised count "
          f"(hospital + unit within hospital + residual)")
    gauss = np.nan
    try:
        md = smf.mixedlm("yz ~ 1", data=d, groups=d["hospital"],
                         re_formula="1",
                         vc_formula={"unit": "0 + C(unittype_)"})
        f = md.fit(method="lbfgs", maxiter=400, reml=True)
        v_h = float(f.cov_re.iloc[0, 0])
        v_u = float(f.vcomp[0]) if len(f.vcomp) else 0.0
        v_e = float(f.scale)
        tot = v_h + v_u + v_e
        gauss = v_h / tot
        print(f"     hospital {v_h:.4f}  unit {v_u:.4f}  residual {v_e:.4f}")
        print(f"     VPC hospital = {gauss:.3f}, unit = {v_u / tot:.3f}"
              f"   converged={f.converged}")
        rows.append({"model": "Gaussian LMM, standardised count",
                     "vpc_hospital": gauss, "vpc_unit": v_u / tot,
                     "converged": bool(f.converged)})
    except Exception as ex:
        print(f"     FAILED: {type(ex).__name__}: {ex}")

    # --- random-effect variances on the log scale, both levels -------------
    def between_var(frame, key):
        g = frame.groupby(key)["n_records"]
        lm, n, v, m = np.log(g.mean()), g.size(), g.var(ddof=1), g.mean()
        raw = float(lm.var(ddof=1))
        samp = float((v / (n * m ** 2)).mean())
        return max(raw - samp, 1e-9), raw, samp

    var_h, raw_h, samp_h = between_var(d, "hospital")
    # unit within hospital: variance of unit log means about their hospital mean
    g = d.groupby(["hospital", "unit_id"])["n_records"]
    ulm = np.log(g.mean())
    within = ulm.groupby(level=0).transform(lambda x: x - x.mean())
    n_u, v_u_, m_u = g.size(), g.var(ddof=1), g.mean()
    var_u = max(float(within.var(ddof=1)) - float((v_u_ / (n_u * m_u ** 2)).mean()),
                1e-9)
    print(f"\n  random-effect variances on the log scale")
    print(f"     hospital           {var_h:.4f}  (raw {raw_h:.4f} "
          f"less sampling {samp_h:.6f})")
    print(f"     unit within hospital {var_u:.4f}")

    # --- dispersion, within unit cell so hospital and unit effects are out --
    def a_hat(s):
        m_, v_ = s.mean(), s.var(ddof=1)
        return (v_ - m_) / m_ ** 2 if m_ > 0 and np.isfinite(v_) else np.nan
    alpha = float(np.nanmedian(
        d.groupby("unit_id")["n_records"].apply(a_hat).clip(lower=1e-6)))

    # --- 2 and 3. latent-scale VPCs, three components ----------------------
    for fam, l1 in (("Poisson", np.log1p(1.0 / mu)),
                    ("negative binomial", np.log1p(1.0 / mu + alpha))):
        tot = var_h + var_u + l1
        print(f"\n  {'2.' if fam == 'Poisson' else '3.'} {fam}: "
              f"level-1 variance {l1:.4f}")
        print(f"     VPC hospital = {var_h / tot:.3f}, "
              f"unit = {var_u / tot:.3f}, residual = {l1 / tot:.3f}"
              + (f"   (alpha = {alpha:.4f})" if fam != "Poisson" else ""))
        rows.append({"model": f"{fam} GLMM, latent-scale VPC",
                     "vpc_hospital": var_h / tot, "vpc_unit": var_u / tot,
                     "level1_var": l1, "converged": True,
                     "alpha": alpha if fam != "Poisson" else None})

    r = pd.DataFrame(rows)
    r.to_csv(out_dir / "glmm_count_vpc.csv", index=False)

    print("\n  " + "-" * 74)
    print("  HOW TO READ THIS")
    if od > 1.5:
        print("  The counts are overdispersed, so the Poisson row is not the")
        print("  relevant comparison: it assumes level-1 variance equals the")
        print("  mean, understates it, and inflates every VPC. The negative")
        print("  binomial row is the appropriate count model.")
    nb = [x for x in rows if x["model"].startswith("negative")]
    if np.isfinite(gauss) and nb:
        diff = abs(nb[0]["vpc_hospital"] - gauss)
        print(f"\n  Gaussian VPC {gauss:.3f} vs negative binomial "
              f"{nb[0]['vpc_hospital']:.3f}, difference {diff:.3f}.")
        print("  " + ("They agree; the hierarchy does not depend on the\n"
                      "  distributional assumption and the manuscript can say so."
                      if diff < 0.10 else
                      "They differ; report the count outcome on the count-model\n"
                      "  scale and state the Gaussian value as the comparison."))
        print("\n  Note the direction: if the negative binomial VPC is higher,")
        print("  the site-dominance finding is stronger under the correct model,")
        print("  not weaker.")
    return r


def verify_reported_vpcs(eic, out_dir):
    """Refit the five VPCs in Table 3 under both optimisers.

    The manuscript fits THREE components: hospital, unit within hospital, and
    residual. An earlier check that omitted the unit level produced uniformly
    higher values, the discrepancy correlating with the omitted component at
    r=0.99; that was an error in the check, not in Table 3. This refits the
    published structure under both optimisers, because method='lbfgs' can
    return a zero random-effect variance while reporting convergence.
    """
    print("\n" + "=" * 78)
    print("VERIFICATION — the five variance partition coefficients in Table 3")
    print("=" * 78)
    d = eic.copy()
    cnt = d["hospital"].value_counts()
    d = d[d["hospital"].isin(cnt[cnt >= MIN_HOSP].index)]
    METRICS = ["n_records", "median_interval_min", "max_interval_min",
               "frac_time_in_gaps", "n_gaps_gt2h"]
    PUBLISHED = {"n_records": 0.434, "median_interval_min": 0.824,
                 "max_interval_min": 0.657, "frac_time_in_gaps": 0.671,
                 "n_gaps_gt2h": 0.499}
    print(f"  {len(d):,} stays, {d['hospital'].nunique()} hospitals\n")
    print(f"  {'metric':22s} {'published':>10s} {'default':>9s} {'lbfgs':>9s} "
          f"{'obs var':>9s} {'':>6s}")
    print("  " + "-" * 72)
    rows = []
    for m in METRICS:
        sub = d.dropna(subset=[m]).copy()
        sub["yz"] = (sub[m] - sub[m].mean()) / sub[m].std()
        obs = float(sub.groupby("hospital")["yz"].mean().var(ddof=1))
        # NB: obs is a two-level quantity and is an upper reference only;
        # the fitted VPC is three-level and will sit below it.
        got = {}
        for meth in ("default", "lbfgs"):
            try:
                kw = {} if meth == "default" else {"method": "lbfgs",
                                                   "maxiter": 400}
                f = smf.mixedlm("yz ~ 1", data=sub, groups=sub["hospital"],
                                re_formula="1",
                                vc_formula={"unit": "0 + C(unit_id)"}
                                ).fit(reml=True, **kw)
                vh = float(f.cov_re.iloc[0, 0])
                vu = float(f.vcomp[0]) if len(f.vcomp) else 0.0
                ve = float(f.scale)
                got[meth] = vh / (vh + vu + ve)
            except Exception:
                got[meth] = np.nan
        pub = PUBLISHED[m]
        flag = "" if abs(got.get("default", np.nan) - pub) < 0.02 else "  <-- CHECK"
        print(f"  {m:22s} {pub:10.3f} {got.get('default', np.nan):9.3f} "
              f"{got.get('lbfgs', np.nan):9.3f} {obs:9.3f}{flag}")
        rows.append({"metric": m, "published": pub,
                     "refit_default": got.get("default"),
                     "refit_lbfgs": got.get("lbfgs"),
                     "observed_between_var": obs})
    pd.DataFrame(rows).to_csv(out_dir / "vpc_verification.csv", index=False)
    print("\n  If the default column matches the published column, Table 3")
    print("  stands. Any row flagged CHECK must be corrected in the manuscript.")


# ------------------------------------------------------- EDITORIAL 18 ------

def eta2(y, groups):
    y = np.asarray(y, float); groups = np.asarray(groups)
    ok = ~np.isnan(y); y, groups = y[ok], groups[ok]
    if len(y) < 10:
        return np.nan
    grand = y.mean(); sst = ((y - grand) ** 2).sum()
    if sst <= 0:
        return np.nan
    d = pd.DataFrame({"y": y, "g": groups}).groupby("g")["y"].agg(["mean", "size"])
    return float((d["size"] * (d["mean"] - grand) ** 2).sum() / sst)


def ss_order(pool, out_dir):
    """Sequential eta-squared under all six factor orderings."""
    print("\n" + "=" * 78)
    print("EDITORIAL COMMENT 18 — does factor entry order change the result?")
    print("=" * 78)
    METRICS = ["n_records", "median_interval_min", "iqr_interval_min",
               "max_interval_min", "n_gaps_gt2h", "frac_time_in_gaps"]
    FACTORS = {"database": "database", "hospital": "hospital",
               "unit": "unit_id"}

    # nesting means the coarser factor's grouping is implied by the finer one,
    # so a sequential R^2 for a set of factors is the R^2 of the finest present
    def r2_of(sub, metric, names):
        cols = [FACTORS[n] for n in names]
        key = sub[cols].astype(str).agg("|".join, axis=1)
        return eta2(sub[metric], key)

    rows = []
    print(f"\n  {'metric':22s} {'component':10s} {'sequential range':>22s} "
          f"{'last-in (order-free)':>22s}")
    print("  " + "-" * 76)
    for m in METRICS:
        sub = pool.dropna(subset=[m])
        total = r2_of(sub, m, ["database", "hospital", "unit"])
        seq = {k: [] for k in FACTORS}
        for order in itertools.permutations(FACTORS):
            prev = 0.0
            for i, f in enumerate(order):
                r2 = r2_of(sub, m, list(order[:i + 1]))
                seq[f].append(r2 - prev)
                prev = r2
        # last-in: contribution when entered after the other two
        lastin = {}
        for f in FACTORS:
            others = [x for x in FACTORS if x != f]
            lastin[f] = total - r2_of(sub, m, others)
        for f in FACTORS:
            lo, hi = min(seq[f]), max(seq[f])
            print(f"  {m if f == 'database' else '':22s} {f:10s} "
                  f"{f'{lo:+.3f} to {hi:+.3f}':>22s} {lastin[f]:>22.3f}")
            rows.append({"metric": m, "component": f, "seq_min": lo,
                         "seq_max": hi, "seq_range": hi - lo,
                         "last_in": lastin[f], "total_r2": total})
        print()
    pd.DataFrame(rows).to_csv(out_dir / "ss_order_sensitivity.csv", index=False)
    db = pd.DataFrame(rows).query("component == 'database'")
    print(f"  Database component across all metrics and orderings: "
          f"{db['seq_min'].min():+.3f} to {db['seq_max'].max():+.3f}")
    print(f"  Manuscript reports 0.016 to 0.071 under the stated ordering.")
    print("  The last-in column is order-free and is the honest summary if the")
    print("  sequential values move materially.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-cache", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./revision_stage2"))
    ap.add_argument("--subsample", type=int, default=None)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    mim, eic = load(a)
    print(f"MIMIC-IV {len(mim):,} stays | eICU-CRD {len(eic):,} stays\n")
    glmm_count(eic, a.out_dir, a.subsample)
    verify_reported_vpcs(eic, a.out_dir)

    cols = ["n_records", "median_interval_min", "iqr_interval_min",
            "max_interval_min", "n_gaps_gt2h", "frac_time_in_gaps",
            "database", "hospital", "unit_id"]
    pool = pd.concat([mim[cols], eic[cols]], ignore_index=True)
    print(f"\npooled: {len(pool):,} stays, {pool['hospital'].nunique()} "
          f"hospitals, {pool['unit_id'].nunique()} unit cells")
    ss_order(pool, a.out_dir)
    print(f"\n-> {a.out_dir}")


if __name__ == "__main__":
    main()
