#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "statsmodels", "pyarrow"]
# ///
"""
Paper 17 revision, Stage 3: who is excluded, and is the site effect real?

EDITORIAL COMMENT 2 — characteristics of the excluded stays
------------------------------------------------------------
The three-record minimum removes 0.12% of MIMIC-IV stays and 10.22% of
eICU-CRD stays, and up to 35% at individual hospitals. The editor asks what
those stays look like. This describes them on every characteristic available
without leaving the analytic frame: length of stay, unit type, mortality where
recorded, and, in eICU-CRD, age and sex.

EDITORIAL COMMENT 5 and REVIEWER N COMMENT 1 — practice or pipeline?
---------------------------------------------------------------------
This is the strongest criticism in the review. Hospital-level variance may be
clinical practice, or it may be incomplete data contribution, and the two have
opposite implications. The paper currently cannot separate them.

A partial separation is available. Nurse charting of vital signs in an ICU has a
floor set by practice: guidelines and unit protocols put routine observation at
hourly or more often for the first 24 hours, so a hospital whose median stay
records a handful of heart-rate values over 24 hours is far more likely to be
contributing partially than to be charting that sparsely. This script therefore
recomputes the hospital variance component after excluding hospitals below a
series of plausibility thresholds, and reports whether the component survives.

It also asks whether the excluded hospitals differ in ways a data-contribution
explanation predicts but a practice explanation does not: whether sparseness is
uniform across their stays or concentrated, whether it holds across all three
vital signs simultaneously, and whether their monitor stream is intact while
their nurse stream is thin. A hospital that charts sparsely as a matter of
practice should be thin in both streams; one contributing partially need not be.

Usage:
  python paper17_stage3_exclusions.py \
      --mimic-cache ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-vp-cache ~/bcst/unit_profile_eicu/vitalperiodic_offsets.parquet \
      --cross-dir ~/bcst/cross_vitals \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --mimic-root ~/physionet.org/files/mimiciv/3.1 \
      --out-dir ~/bcst/revision_stage3
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WINDOW_MIN = 24 * 60
MIN_RECORDS = 3
MIN_HOSP = 500


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


def metrics(ev, key, off):
    ev = ev.sort_values([key, off])
    g = ev.groupby(key)[off]
    out = pd.DataFrame({"n_records": g.size()})
    ev = ev.assign(gap=g.diff())
    gg = ev.dropna(subset=["gap"]).groupby(key)["gap"]
    out["median_interval_min"] = gg.median()
    out["max_interval_min"] = gg.max()
    out["frac_time_in_gaps"] = gg.apply(
        lambda s: float(s[s > 120].sum()) / WINDOW_MIN)
    return out.reset_index()


# --------------------------------------------------------- EDITORIAL 2 -----

def excluded_profile(a, out_dir):
    print("=" * 78)
    print("EDITORIAL COMMENT 2 — what do the excluded stays look like?")
    print("=" * 78)
    rows = []

    # ---- MIMIC ----
    cnt = (pd.read_parquet(a.mimic_cache).groupby("stay_id").size()
             .rename("n").reset_index())
    ps = pd.read_csv(a.mimic_per_stay)
    m = ps.merge(cnt, on="stay_id", how="left")
    m["n"] = m["n"].fillna(0).astype(int)
    m = m[m["los"] >= 1.0].copy()
    m["excluded"] = m["n"] < MIN_RECORDS
    print(f"\n  MIMIC-IV: {len(m):,} stays completing the window, "
          f"{int(m['excluded'].sum()):,} excluded "
          f"({m['excluded'].mean():.4f})")
    for col, label in (("los", "ICU length of stay, d"),
                       ("age_z", "age, standardised"),
                       ("n_comorbid_z", "comorbidity count, standardised")):
        if col not in m.columns:
            continue
        a_, b_ = m.loc[~m["excluded"], col], m.loc[m["excluded"], col]
        print(f"    {label:34s} retained {a_.median():8.2f}   "
              f"excluded {b_.median():8.2f}")
        rows.append({"cohort": "MIMIC-IV", "variable": label,
                     "retained_median": float(a_.median()),
                     "excluded_median": float(b_.median()),
                     "n_excluded": int(m["excluded"].sum())})
    for col in ("careunit", "gender"):
        if col not in m.columns:
            continue
        t = (m.groupby(col)["excluded"].mean().sort_values(ascending=False)
               .head(5))
        print(f"    exclusion rate by {col}: "
              + ", ".join(f"{k} {v:.4f}" for k, v in t.items()))
    if "hospital_expire_flag" in m.columns:
        print(f"    in-hospital mortality  retained "
              f"{m.loc[~m['excluded'], 'hospital_expire_flag'].mean():.4f}   "
              f"excluded {m.loc[m['excluded'], 'hospital_expire_flag'].mean():.4f}")

    # ---- eICU ----
    nc = (pd.read_parquet(a.eicu_nc_cache).groupby("patientunitstayid").size()
            .rename("n").reset_index())
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset", "unitdischargestatus",
                               "age", "gender"])
    e = pat.merge(nc, on="patientunitstayid", how="left")
    e["n"] = e["n"].fillna(0).astype(int)
    e = e[e["unitdischargeoffset"] >= WINDOW_MIN].copy()
    e["excluded"] = e["n"] < MIN_RECORDS
    e["age_num"] = pd.to_numeric(e["age"].replace("> 89", "90"),
                                 errors="coerce")
    e["los_d"] = e["unitdischargeoffset"] / 1440.0
    e["died"] = (e["unitdischargestatus"] == "Expired").astype(float)
    print(f"\n  eICU-CRD: {len(e):,} stays completing the window, "
          f"{int(e['excluded'].sum()):,} excluded "
          f"({e['excluded'].mean():.4f})")
    for col, label in (("los_d", "ICU length of stay, d"),
                       ("age_num", "age, years"),
                       ("died", "ICU mortality")):
        a_, b_ = e.loc[~e["excluded"], col], e.loc[e["excluded"], col]
        f = "mean" if col == "died" else "median"
        va = a_.mean() if f == "mean" else a_.median()
        vb = b_.mean() if f == "mean" else b_.median()
        print(f"    {label:34s} retained {va:8.2f}   excluded {vb:8.2f}")
        rows.append({"cohort": "eICU-CRD", "variable": label,
                     "retained_median": float(va), "excluded_median": float(vb),
                     "n_excluded": int(e["excluded"].sum())})
    t = e.groupby("unittype")["excluded"].mean().sort_values(ascending=False)
    print(f"    exclusion rate by unit type: "
          + ", ".join(f"{k} {v:.3f}" for k, v in t.head(4).items()))
    print(f"    exclusion rate by sex: "
          + ", ".join(f"{k} {v:.4f}"
                      for k, v in e.groupby("gender")["excluded"].mean()
                      .head(3).items()))

    # concentration: is exclusion spread across hospitals or concentrated?
    hx = e.groupby("hospitalid")["excluded"].agg(["mean", "size"])
    hx = hx[hx["size"] >= 100].sort_values("mean", ascending=False)
    top = hx.head(10)
    tot_excl = int(e["excluded"].sum())
    share_top = float((top["mean"] * top["size"]).sum() / tot_excl)
    print(f"\n    exclusion is concentrated: the 10 hospitals with the highest")
    print(f"    exclusion rate supply {share_top:.3f} of all excluded stays")
    print(f"    their rates: "
          + ", ".join(f"{v:.2f}" for v in top['mean'].head(10)))
    pd.DataFrame(rows).to_csv(out_dir / "excluded_stay_profile.csv",
                              index=False)
    return e


# --------------------------------------- EDITORIAL 5 / REVIEWER N 1 --------

def plausibility(e, a, out_dir):
    print("\n" + "=" * 78)
    print("EDITORIAL 5 / REVIEWER N 1 — practice or partial data contribution?")
    print("=" * 78)

    nc = pd.read_parquet(a.eicu_nc_cache)
    d = metrics(nc, "patientunitstayid", "observationoffset")
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    d = d.merge(pat, on="patientunitstayid", how="inner")
    d = d[(d["unitdischargeoffset"] >= WINDOW_MIN)
          & (d["n_records"] >= MIN_RECORDS)].copy()
    d["unit_id"] = ("eICU-" + d["hospitalid"].astype(str) + ":"
                    + d["unittype"].astype(str))
    cnt = d["hospitalid"].value_counts()
    d = d[d["hospitalid"].isin(cnt[cnt >= MIN_HOSP].index)]

    hosp_med = d.groupby("hospitalid")["n_records"].median()
    print(f"\n  {d['hospitalid'].nunique()} hospitals, per-hospital median "
          f"record count over 24 h:")
    print(f"    min {hosp_med.min():.0f}, p10 {hosp_med.quantile(.1):.0f}, "
          f"median {hosp_med.median():.0f}, p90 {hosp_med.quantile(.9):.0f}, "
          f"max {hosp_med.max():.0f}")

    print(f"\n  Hospital variance component after excluding hospitals below a")
    print(f"  plausibility floor on the median 24-hour record count.")
    print(f"  Four-hourly observation over 24 h is 6 records; two-hourly is 12;")
    print(f"  hourly is 24. A median below these is progressively less")
    print(f"  consistent with any documented ICU observation standard.\n")
    print(f"    {'floor':>6s} {'hospitals':>10s} {'stays':>10s} "
          f"{'eta2 count':>11s} {'eta2 interval':>14s} {'eta2 gapfrac':>13s}")
    print("    " + "-" * 68)
    rows = []
    for floor in (0, 3, 6, 12, 18, 24):
        keep = hosp_med[hosp_med >= floor].index
        s = d[d["hospitalid"].isin(keep)]
        if s["hospitalid"].nunique() < 10:
            continue
        r = {"floor": floor, "n_hospitals": int(s["hospitalid"].nunique()),
             "n_stays": len(s)}
        for m, lab in (("n_records", "eta2_count"),
                       ("median_interval_min", "eta2_interval"),
                       ("frac_time_in_gaps", "eta2_gapfrac")):
            r[lab] = eta2(s[m], s["hospitalid"])
        print(f"    {floor:>6d} {r['n_hospitals']:>10d} {r['n_stays']:>10,} "
              f"{r['eta2_count']:>11.3f} {r['eta2_interval']:>14.3f} "
              f"{r['eta2_gapfrac']:>13.3f}")
        rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir / "plausibility_thresholds.csv",
                              index=False)
    print("\n  If the components hold as the floor rises, the site effect is")
    print("  not carried by the implausibly sparse hospitals, and the")
    print("  data-contribution explanation cannot account for it alone.")

    # --- does sparseness hold across streams and variables? ---
    print("\n" + "-" * 78)
    print("  Do the sparse hospitals look like practice or like partial data?")
    print("-" * 78)
    vp = (pd.read_parquet(a.eicu_vp_cache).groupby("patientunitstayid").size()
            .rename("n_vp").reset_index())
    j = d[["patientunitstayid", "hospitalid", "n_records"]].merge(
        vp, on="patientunitstayid", how="left")
    j["n_vp"] = j["n_vp"].fillna(0)
    h = j.groupby("hospitalid").agg(nurse=("n_records", "median"),
                                    monitor=("n_vp", "median"),
                                    n=("n_records", "size"))
    h = h[h["n"] >= MIN_HOSP]
    sparse = h[h["nurse"] < 12]
    rest = h[h["nurse"] >= 12]
    print(f"\n  hospitals with nurse median < 12 records: {len(sparse)}")
    print(f"    their nurse median   {sparse['nurse'].median():.1f}  "
          f"vs {rest['nurse'].median():.1f} elsewhere")
    print(f"    their monitor median {sparse['monitor'].median():.0f}  "
          f"vs {rest['monitor'].median():.0f} elsewhere")
    print("    A hospital charting sparsely as a matter of practice would be")
    print("    thin in the nurse stream only; the monitor stream is machine-")
    print("    generated and should be unaffected either way. An intact")
    print("    monitor stream with a thin nurse stream is therefore consistent")
    print("    with both explanations and does not separate them. What would")
    print("    separate them is the cross-variable pattern below.")

    # cross-variable: are the same hospitals sparse for SpO2 and RR?
    cd = a.cross_dir
    if (cd / "eicu_nc.parquet").exists():
        cn = pd.read_parquet(cd / "eicu_nc.parquet")
        piv = {}
        for var in sorted(cn["variable"].unique()):
            c = (cn[cn["variable"] == var]
                 .groupby("patientunitstayid").size().rename(var).reset_index())
            piv[var] = c
        allv = d[["patientunitstayid", "hospitalid"]].copy()
        for var, c in piv.items():
            allv = allv.merge(c, on="patientunitstayid", how="left")
        hv = allv.groupby("hospitalid").median(numeric_only=True)
        hv = hv.join(h[["nurse"]], how="inner").dropna()
        print(f"\n  per-hospital median counts, heart rate vs the other two "
              f"variables ({len(hv)} hospitals)")
        for var in piv:
            if var in hv.columns:
                print(f"    corr(HR median, {var} median) = "
                      f"{hv['nurse'].corr(hv[var]):+.3f}")
        print("    A correlation near 1 means the same hospitals are sparse for")
        print("    every variable, which is what partial contribution of the")
        print("    whole nurse stream predicts. Variable-specific sparseness")
        print("    would instead suggest differing charting practice.")
        hv.to_csv(out_dir / "cross_variable_hospital_medians.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-cache", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--eicu-vp-cache", required=True, type=Path)
    ap.add_argument("--cross-dir", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--mimic-root", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./revision_stage3"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    e = excluded_profile(a, a.out_dir)
    plausibility(e, a, a.out_dir)
    print(f"\n-> {a.out_dir}")


if __name__ == "__main__":
    main()
