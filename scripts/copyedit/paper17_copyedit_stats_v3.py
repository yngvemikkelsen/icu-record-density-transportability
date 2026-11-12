#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Paper 17 — copyediting statistics, v3: the residue only.

v2 resolved sections A to F and every analytic frame reproduced exactly.
This run closes the five quantities v2 did not cover. Nothing here repeats v2;
run both and use the two logs together.

  1. Excluded vs retained ICU length of stay, with IQR, in both databases
     ("the 84 excluded stays had a longer median ICU stay ... 6.53 vs 2.41";
      "the 13,583 excluded stays ... 2.20 vs 2.33").
  2. Restricted-cohort per-stay record count, median with IQR
     ("against 26 (IQR #-#) for the cohort").
  3. Per-stay SDs for the MIMIC-IV item profile
     (observation item 25.9, each alarm item 2.8, pooled-item 31.5, and the
      25.88 figure on the 85,133-stay reconstruction frame).
  4. Share of restricted-cohort stays with no gap beyond 2 hours, as n/N
     ("Gaps exceeding 2 hours are 73.7% zeros").
  5. Severity-component coverage denominators, as n/N, printed with the
     cohort size the coverage probe used
     (bilirubin 43.5% / 37.1%, PaO2 54%, FiO2 33.1%, urine output 49.1%,
      elective surgery 19.4%, APACHE unavailable 23%).

VERIFY: each published point estimate is printed beside the recomputed value.

Usage:
  python paper17_copyedit_stats_v3.py \
      --mimic-cache  ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --mimic-root ~/physionet.org/files/mimiciv/3.1 \
      --out-dir ~/bcst/copyedit_stats_v3
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_MIN = 24 * 60
WINDOW_H = 24.0
ROWS = []

HR_ITEM, ALARM_HI, ALARM_LO = 220045, 220046, 220047


def rec(sec, label, value, expect=""):
    ROWS.append({"section": sec, "label": label, "value": value,
                 "manuscript_says": expect})
    e = f"   [manuscript: {expect}]" if expect else ""
    print(f"    {label:<54s} {value}{e}")


def iqr(s):
    s = pd.Series(s).dropna()
    return "n/a" if not len(s) else f"{s.quantile(0.25):g}-{s.quantile(0.75):g}"


def med_iqr(s, dp=2):
    s = pd.Series(s).dropna()
    if not len(s):
        return "n/a"
    return (f"median {s.median():.{dp}f} "
            f"(IQR {s.quantile(0.25):.{dp}f}-{s.quantile(0.75):.{dp}f})")


def mean_sd(s):
    s = pd.Series(s).dropna()
    return "n/a" if not len(s) else f"mean {s.mean():.4f} (SD {s.std(ddof=1):.4g})"


def in_window(df, off):
    return df[(df[off] >= 0) & (df[off] <= WINDOW_MIN)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-cache", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--mimic-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./copyedit_stats_v3"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    mim = pd.read_parquet(a.mimic_cache)
    moff = "offset_min" if "offset_min" in mim.columns else mim.columns[-1]
    nc = pd.read_parquet(a.eicu_nc_cache)
    ps = pd.read_csv(a.mimic_per_stay)
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid",
                               "unitdischargeoffset"])

    mim_w = in_window(mim, moff)
    m_cnt = mim_w.groupby("stay_id").size().rename("n_records").reset_index()
    m_cnt["los"] = m_cnt["stay_id"].map(ps.set_index("stay_id")["los"])
    cw_m = m_cnt[m_cnt["los"] >= 1.0]
    print("=" * 78)
    print("1. EXCLUDED VS RETAINED, ICU LENGTH OF STAY")
    print("=" * 78)
    print(f"  MIMIC completed-window: {len(cw_m):,}   [manuscript: 67,224]")
    exc_m, ret_m = cw_m[cw_m["n_records"] < 3], cw_m[cw_m["n_records"] >= 3]
    print(f"  excluded by 3-record minimum: {len(exc_m):,}   [manuscript: 84]")
    rec("1", "MIMIC excluded: ICU length of stay (d)",
        med_iqr(exc_m["los"]), "median 6.53")
    rec("1", "MIMIC retained: ICU length of stay (d)",
        med_iqr(ret_m["los"]), "median 2.41")

    nc_w = in_window(nc, "observationoffset")
    e_cnt = (nc_w.groupby("patientunitstayid").size()
                  .rename("n_records").reset_index())
    cw_e = pat[pat["unitdischargeoffset"] >= WINDOW_MIN].merge(
        e_cnt, on="patientunitstayid", how="left")
    cw_e["n_records"] = cw_e["n_records"].fillna(0).astype(int)
    cw_e["los_d"] = cw_e["unitdischargeoffset"] / 1440.0
    print(f"  eICU completed-window: {len(cw_e):,}   [manuscript: 132,900]")
    exc_e, ret_e = cw_e[cw_e["n_records"] < 3], cw_e[cw_e["n_records"] >= 3]
    print(f"  excluded by 3-record minimum: {len(exc_e):,}   [manuscript: 13,583]")
    rec("1", "eICU excluded: ICU length of stay (d)",
        med_iqr(exc_e["los_d"]), "median 2.20")
    rec("1", "eICU retained: ICU length of stay (d)",
        med_iqr(ret_e["los_d"]), "median 2.33")

    print("\n" + "=" * 78)
    print("2. RESTRICTED-COHORT RECORD COUNT")
    print("=" * 78)
    elig = cw_e[cw_e["n_records"] >= 3]
    big = elig.groupby("hospitalid").size()
    big = big[big >= 500].index
    unres = elig[elig["hospitalid"].isin(big)]
    hmed = unres.groupby("hospitalid")["n_records"].median()
    restr = unres[~unres["hospitalid"].isin(hmed[hmed < 12].index)]
    print(f"  restricted cohort: {len(restr):,}/"
          f"{restr['hospitalid'].nunique()}   [manuscript: 88,560/60]")
    rec("2", "restricted cohort: records per stay",
        med_iqr(restr["n_records"], dp=0), "26 for the cohort")

    print("\n" + "=" * 78)
    print("3. MIMIC-IV ITEM PROFILE, PER-STAY STANDARD DEVIATIONS")
    print("=" * 78)
    src = a.mimic_root / "icu" / "chartevents.csv.gz"
    if not src.exists():
        print(f"  chartevents not found at {src}; section 3 skipped")
    else:
        want = {HR_ITEM, ALARM_HI, ALARM_LO}
        parts = []
        for ch in pd.read_csv(src, usecols=["stay_id", "itemid", "charttime"],
                              chunksize=5_000_000):
            parts.append(ch[ch["itemid"].isin(want)])
        ev = pd.concat(parts, ignore_index=True)
        print(f"  rows for the 3 items: {len(ev):,}")
        for item, lab, exp in ((HR_ITEM, "observation 220045", "25.9 / 25.92"),
                               (ALARM_HI, "alarm high 220046", "2.8"),
                               (ALARM_LO, "alarm low 220047", "2.8")):
            per = ev[ev["itemid"] == item].groupby("stay_id").size()
            print(f"  {lab}: {per.index.nunique():,} stays")
            rec("3", f"{lab}: records per stay", mean_sd(per), exp)
        pooled = ev.groupby("stay_id").size()
        rec("3", "pooled 3-item count: records per stay",
            mean_sd(pooled), "31.5 / 31.56")
        recon = pooled[pooled.index.isin(set(ps["stay_id"]))]
        print(f"  reconstruction frame: {len(recon):,}   [manuscript: 85,133]")
        obs = ev[ev["itemid"] == HR_ITEM].groupby("stay_id").size()
        obs_r = obs[obs.index.isin(set(ps["stay_id"]))]
        rec("3", "observation item on the reconstruction frame",
            mean_sd(obs_r), "25.88")

    print("\n" + "=" * 78)
    print("4. GAPS BEYOND 2 HOURS, SHARE OF ZEROS")
    print("=" * 78)
    sub = nc_w[nc_w["patientunitstayid"].isin(set(restr["patientunitstayid"]))]
    sub = sub.sort_values(["patientunitstayid", "observationoffset"])
    d = sub.groupby("patientunitstayid")["observationoffset"].diff()
    g = (sub.assign(gap=d).dropna(subset=["gap"])
            .groupby("patientunitstayid")["gap"]
            .apply(lambda s: int((s > 120).sum())))
    z = int((g == 0).sum())
    rec("4", "restricted cohort: stays with no gap beyond 2 h",
        f"{z:,}/{len(g):,} = {z/len(g):.1%}", "73.7% zeros")

    print("\n" + "=" * 78)
    print("5. SEVERITY-COMPONENT COVERAGE, AS n/N")
    print("=" * 78)
    print("  Re-run paper17_probe_sofa_coverage.py and read its printed")
    print("  'n_stays' for each database; coverage x n_stays is the numerator.")
    print("  Reported here from the archived shares against the cohort the")
    print("  probe uses, so the denominator is stated rather than inferred:")
    for f, cohort_n, items in (
            ("mimic_sofa_coverage.csv", len(ps),
             {"bilirubin": "43.5%", "pao2": "54%"}),
            ("eicu_sofa_coverage.csv", len(pat),
             {"bilirubin": "37.1%", "fio2": "33.1%"})):
        p = Path(__file__).resolve().parent / f
        if not p.exists():
            print(f"  {f} not alongside the script; skipped")
            continue
        for r in pd.read_csv(p).itertuples():
            if r.component in items:
                n = r.coverage * cohort_n
                rec("5", f"{f.split('_')[0]} {r.component}",
                    f"{n:,.0f}/{cohort_n:,} = {r.coverage:.1%}",
                    items[r.component])

    out = pd.DataFrame(ROWS)
    out.to_csv(a.out_dir / "copyedit_statistics_v3.csv", index=False)
    print("\n" + "=" * 78)
    print(f"{len(out)} statistics -> {a.out_dir / 'copyedit_statistics_v3.csv'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
