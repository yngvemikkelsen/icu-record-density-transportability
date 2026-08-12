#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "statsmodels", "pyarrow"]
# ///
"""
Paper 17: the two sensitivity analyses promised in Methods 2.2 and section 4.5.

A. ELIGIBILITY SENSITIVITY (the >=3-record criterion)
   Every analysis so far required at least three heart-rate records in the first
   24 h.  For interval statistics that is unavoidable — two records give one
   interval — but the count endpoint does not need it, and requiring it
   conditions inclusion on the record-generating process under study.  If sites
   that chart sparsely are also the sites whose stays fall below the threshold,
   the exclusion attenuates exactly the between-site variation being estimated.

   This section rebuilds the count endpoint over ALL eligible stays, assigning
   zero to stays with no heart-rate record in the window, and reports:
     - the share excluded by the >=3 rule, per cohort and per hospital
     - whether that share correlates with the hospital's charting rate
     - the hospital variance component for the count, with and without the rule

B. MIXED-EFFECTS VARIANCE PARTITION
   Eta-squared from nested sums of squares is transparent but sensitive to
   unequal group sizes, and eICU hospital sizes are highly unequal.  A
   random-intercept model with hospital and unit-within-hospital gives variance
   partition coefficients that do not depend on group size in the same way.  If
   the hierarchy is unchanged, the principal result is robust to the estimator;
   if it is not, the eta-squared version cannot stand alone.

   Metrics are standardised before fitting, purely for convergence.  VPC is
   reported as each component over the total.

Usage:
  python paper17_sensitivity.py \
      --mimic-cache ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/sensitivity

Mixed models are slow; use --subsample N to fit on a random subset of stays per
hospital if the full fit does not complete in reasonable time.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

WINDOW_MIN = 24 * 60
MIN_HOSP_STAYS = 500

VC_METRICS = ["n_records", "median_interval_min", "max_interval_min",
              "frac_time_in_gaps", "n_gaps_gt2h"]


def eta2(y, groups):
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    ok = ~np.isnan(y)
    y, groups = y[ok], groups[ok]
    if len(y) < 10:
        return np.nan
    grand = y.mean()
    sst = ((y - grand) ** 2).sum()
    if sst <= 0:
        return np.nan
    d = pd.DataFrame({"y": y, "g": groups}).groupby("g")["y"].agg(["mean", "size"])
    return float((d["size"] * (d["mean"] - grand) ** 2).sum() / sst)


def metrics_from_offsets(ev, key, offset_col):
    ev = ev.sort_values([key, offset_col])
    g = ev.groupby(key)[offset_col]
    out = pd.DataFrame({"n_records": g.size()})
    ev = ev.assign(gap_min=g.diff())
    gg = ev.dropna(subset=["gap_min"]).groupby(key)["gap_min"]
    out["median_interval_min"] = gg.median()
    out["max_interval_min"] = gg.max()
    out["n_gaps_gt2h"] = gg.apply(lambda s: int((s > 120).sum()))
    out["frac_time_in_gaps"] = gg.apply(
        lambda s: float(s[s > 120].sum()) / WINDOW_MIN)
    return out.reset_index()


# ---------------------------------------------------------------------------
# A. eligibility sensitivity
# ---------------------------------------------------------------------------

def eligibility(mimic_cache, mimic_per_stay, eicu_cache, eicu_root, out_dir):
    print("=" * 78)
    print("A. ELIGIBILITY SENSITIVITY: the >=3-record criterion")
    print("=" * 78)

    # --- MIMIC: denominator is every stay completing the window ---
    ps = pd.read_csv(mimic_per_stay, usecols=["stay_id", "careunit", "los"])
    ps = ps[ps["los"] >= 1.0]
    cnt = (pd.read_parquet(mimic_cache).groupby("stay_id").size()
             .rename("n_records").reset_index())
    m = ps.merge(cnt, on="stay_id", how="left")
    m["n_records"] = m["n_records"].fillna(0).astype(int)

    # --- eICU: denominator is every unit stay completing the window ---
    pat = pd.read_csv(eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    pat = pat[pat["unitdischargeoffset"] >= WINDOW_MIN]
    ec = (pd.read_parquet(eicu_cache).groupby("patientunitstayid").size()
            .rename("n_records").reset_index())
    e = pat.merge(ec, on="patientunitstayid", how="left")
    e["n_records"] = e["n_records"].fillna(0).astype(int)

    rows = []
    for name, d in (("MIMIC", m), ("eICU", e)):
        n = len(d)
        for k in (0, 1, 2):
            share = float((d["n_records"] == k).mean())
            rows.append({"cohort": name, "n_records": k, "n": int((d["n_records"] == k).sum()),
                         "share": share})
        excl = float((d["n_records"] < 3).mean())
        print(f"\n  {name}: {n:,} stays completing the 24 h window")
        print(f"    with 0 records: {int((d['n_records'] == 0).sum()):,} "
              f"({(d['n_records'] == 0).mean():.4f})")
        print(f"    with 1-2 records: {int(d['n_records'].between(1, 2).sum()):,} "
              f"({d['n_records'].between(1, 2).mean():.4f})")
        print(f"    excluded by >=3 rule: {excl:.4f}")
    pd.DataFrame(rows).to_csv(out_dir / "eligibility_shares.csv", index=False)

    # --- does exclusion track a hospital's charting rate? ---
    counts = e["hospitalid"].value_counts()
    big = counts[counts >= MIN_HOSP_STAYS].index
    eb = e[e["hospitalid"].isin(big)]
    per_h = eb.groupby("hospitalid").agg(
        n_stays=("n_records", "size"),
        excluded_share=("n_records", lambda s: float((s < 3).mean())),
        median_count=("n_records", "median"),
        median_count_eligible=("n_records", lambda s: float(s[s >= 3].median())))
    r = per_h["excluded_share"].corr(per_h["median_count_eligible"],
                                     method="spearman")
    print(f"\n  eICU hospitals with >= {MIN_HOSP_STAYS} stays: {len(per_h)}")
    print(f"    excluded share: median {per_h['excluded_share'].median():.4f}, "
          f"p10 {per_h['excluded_share'].quantile(.1):.4f}, "
          f"p90 {per_h['excluded_share'].quantile(.9):.4f}")
    print(f"    Spearman(excluded share, median count among eligible) = {r:+.3f}")
    print("    A strong negative value means sparsely-charting hospitals lose "
          "more stays to the rule, so the rule attenuates the between-hospital "
          "variance it is used to estimate.")
    per_h.to_csv(out_dir / "eligibility_by_hospital.csv")

    # --- the count endpoint with and without the rule ---
    print("\n  HOSPITAL VARIANCE COMPONENT FOR THE COUNT ENDPOINT")
    a = eta2(eb["n_records"], eb["hospitalid"])
    b = eta2(eb.loc[eb["n_records"] >= 3, "n_records"],
             eb.loc[eb["n_records"] >= 3, "hospitalid"])
    print(f"    all stays (0/1/2 admitted):  eta^2 = {a:.3f}  n={len(eb):,}")
    print(f"    >=3 records only:            eta^2 = {b:.3f}  "
          f"n={int((eb['n_records'] >= 3).sum()):,}")
    print(f"    difference: {a - b:+.3f}")
    pd.DataFrame([{"set": "all stays", "eta2_hospital": a, "n": len(eb)},
                  {"set": ">=3 records", "eta2_hospital": b,
                   "n": int((eb["n_records"] >= 3).sum())}]).to_csv(
        out_dir / "eligibility_count_eta2.csv", index=False)

    mu = eta2(m["n_records"], m["careunit"])
    mu3 = eta2(m.loc[m["n_records"] >= 3, "n_records"],
               m.loc[m["n_records"] >= 3, "careunit"])
    print(f"\n    MIMIC between-unit eta^2, all stays: {mu:.4f}; "
          f">=3 records: {mu3:.4f}")
    return eb


# ---------------------------------------------------------------------------
# B. mixed-effects variance partition
# ---------------------------------------------------------------------------

def mixed_vpc(eic, out_dir, subsample=None, seed=17):
    print("\n" + "=" * 78)
    print("B. MIXED-EFFECTS VARIANCE PARTITION (random intercepts: hospital, "
          "unit within hospital)")
    print("=" * 78)
    d = eic.copy()
    if subsample:
        rng = np.random.default_rng(seed)
        d = (d.groupby("hospitalid", group_keys=False)
               .apply(lambda g: g.sample(min(len(g), subsample),
                                         random_state=seed)))
        print(f"  subsampled to <= {subsample} stays per hospital: {len(d):,}")
    d["unittype"] = d["unittype"].astype(str)

    print(f"\n  {'metric':22s} {'VPC hospital':>13s} {'VPC unit|hosp':>14s} "
          f"{'VPC residual':>13s} {'eta2 hosp':>10s}")
    print("-" * 78)
    rows = []
    for m in VC_METRICS:
        sub = d.dropna(subset=[m]).copy()
        if len(sub) < 500 or sub[m].std() == 0:
            print(f"  {m:22s} {'insufficient variation':>40s}")
            continue
        sub["y"] = (sub[m] - sub[m].mean()) / sub[m].std()
        try:
            md = smf.mixedlm("y ~ 1", data=sub, groups=sub["hospitalid"],
                             re_formula="1",
                             vc_formula={"unit": "0 + C(unittype)"})
            fit = md.fit(method="lbfgs", maxiter=200, reml=True)
            v_h = float(fit.cov_re.iloc[0, 0])
            v_u = float(fit.vcomp[0]) if len(fit.vcomp) else 0.0
            v_e = float(fit.scale)
            tot = v_h + v_u + v_e
            e2 = eta2(sub[m], sub["hospitalid"])
            print(f"  {m:22s} {v_h / tot:13.3f} {v_u / tot:14.3f} "
                  f"{v_e / tot:13.3f} {e2:10.3f}")
            rows.append({"metric": m, "vpc_hospital": v_h / tot,
                         "vpc_unit_within_hospital": v_u / tot,
                         "vpc_residual": v_e / tot, "eta2_hospital": e2,
                         "converged": bool(fit.converged), "n": len(sub)})
        except Exception as ex:
            print(f"  {m:22s} FAILED ({type(ex).__name__}: {str(ex)[:40]})")
            rows.append({"metric": m, "vpc_hospital": np.nan,
                         "vpc_unit_within_hospital": np.nan,
                         "vpc_residual": np.nan,
                         "eta2_hospital": eta2(sub[m], sub["hospitalid"]),
                         "converged": False, "n": len(sub)})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "mixed_vpc.csv", index=False)
    print("\n  If VPC hospital greatly exceeds VPC unit-within-hospital, the "
          "hierarchy found by eta-squared is reproduced under an estimator "
          "that does not weight by group size in the same way.")
    print("  Check the 'converged' column before using any row.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-cache", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-cache", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./sensitivity"))
    ap.add_argument("--subsample", type=int, default=None,
                    help="max stays per hospital for the mixed model")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    eb = eligibility(args.mimic_cache, args.mimic_per_stay, args.eicu_cache,
                     args.eicu_root, args.out_dir)

    # mixed model needs the timing metrics, so rebuild on eligible stays
    ev = pd.read_parquet(args.eicu_cache)
    met = metrics_from_offsets(ev, "patientunitstayid", "observationoffset")
    pat = pd.read_csv(args.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    eic = met.merge(pat, on="patientunitstayid", how="inner")
    eic = eic[(eic["unitdischargeoffset"] >= WINDOW_MIN)
              & (eic["n_records"] >= 3)]
    counts = eic["hospitalid"].value_counts()
    eic = eic[eic["hospitalid"].isin(counts[counts >= MIN_HOSP_STAYS].index)]
    print(f"\nmixed-model cohort: {len(eic):,} stays, "
          f"{eic['hospitalid'].nunique()} hospitals")

    mixed_vpc(eic, args.out_dir, args.subsample)
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
