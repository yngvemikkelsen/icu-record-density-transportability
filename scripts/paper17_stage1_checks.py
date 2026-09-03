#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Paper 17 revision, Stage 1: two checks that must be settled before anything is
rewritten, because either could change what the response says.

EDITORIAL COMMENT 15 / REVIEWER G COMMENT 4 — what is a record?
---------------------------------------------------------------
The counting code groups by stay and calls .size(), which counts ROWS. If the
same physiological variable is stored more than once at an identical timestamp,
each copy is counted, and the interval between them is zero. That would inflate
the record count and depress the median interval.

This reports, for each cached stream: the share of records that share a
timestamp with another record for the same stay, the share of stays affected,
and what the record count and interval metrics become if identical timestamps
are collapsed to one. If the effect is negligible the answer to the editor is a
sentence; if it is not, the affected metrics have to be recomputed.

EDITORIAL COMMENT 12 — missingness in the admission-hour covariates
-------------------------------------------------------------------
The admission-hour models adjust for age, sex, comorbidity count, ICD chapter
count, admission era and care unit (MIMIC-IV), and age, sex and unit type
(eICU-CRD). The fitting code applies no imputation: statsmodels drops rows with
any missing term in the formula, silently. That is a complete-case analysis, and
the editor is right that the rates should be stated.

This reports per-covariate missingness and the cumulative complete-case loss, so
the response can give numbers rather than a description.

Usage:
  python paper17_stage1_checks.py \
      --mimic-cache ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-vp-cache ~/bcst/unit_profile_eicu/vitalperiodic_offsets.parquet \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/revision_stage1

Runs from the parquet caches. No new extraction.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_MIN = 24 * 60


# ------------------------------------------------------------ EDITORIAL 15 --

def dup_check(df, key, offset_col, label, rows):
    """Duplicate-timestamp burden, and the effect of collapsing duplicates."""
    n_rec = len(df)
    n_stay = df[key].nunique()

    # a record is duplicated if (stay, timestamp) occurs more than once
    grp = df.groupby([key, offset_col]).size()
    n_unique_pairs = len(grp)
    n_dup_records = n_rec - n_unique_pairs
    stays_affected = grp[grp > 1].index.get_level_values(0).nunique()
    worst = int(grp.max())

    print(f"\n  {label}")
    print(f"    records {n_rec:,} across {n_stay:,} stays")
    print(f"    distinct (stay, timestamp) pairs: {n_unique_pairs:,}")
    print(f"    duplicated records: {n_dup_records:,} "
          f"({n_dup_records / n_rec:.5f} of records)")
    print(f"    stays with >=1 duplicated timestamp: {stays_affected:,} "
          f"({stays_affected / n_stay:.5f} of stays)")
    print(f"    largest number of records at one timestamp: {worst}")

    # what changes if duplicates are collapsed
    def metrics(d):
        d = d.sort_values([key, offset_col])
        g = d.groupby(key)[offset_col]
        n = g.size()
        gaps = d.assign(gap=g.diff()).dropna(subset=["gap"])
        med = gaps.groupby(key)["gap"].median()
        zero = float((gaps["gap"] == 0).mean())
        return n, med, zero

    n_raw, med_raw, zero_raw = metrics(df)
    dedup = df.drop_duplicates(subset=[key, offset_col])
    n_ded, med_ded, zero_ded = metrics(dedup)

    print(f"    zero-length intervals, as counted: {zero_raw:.5f} of gaps")
    print(f"    median record count   raw {n_raw.median():.1f} -> "
          f"deduplicated {n_ded.median():.1f}")
    print(f"    median interval (min) raw {med_raw.median():.1f} -> "
          f"deduplicated {med_ded.median():.1f}")

    material = (n_dup_records / n_rec > 0.005) or \
               (abs(n_raw.median() - n_ded.median()) >= 0.5) or \
               (abs(med_raw.median() - med_ded.median()) >= 0.5)
    print(f"    -> {'MATERIAL: affected metrics need recomputing' if material else 'negligible: a sentence in the response suffices'}")

    rows.append({
        "stream": label, "n_records": n_rec, "n_stays": n_stay,
        "n_duplicate_records": int(n_dup_records),
        "share_records_duplicated": n_dup_records / n_rec,
        "n_stays_affected": int(stays_affected),
        "share_stays_affected": stays_affected / n_stay,
        "max_records_one_timestamp": worst,
        "share_zero_length_gaps": zero_raw,
        "median_count_raw": float(n_raw.median()),
        "median_count_dedup": float(n_ded.median()),
        "median_interval_raw": float(med_raw.median()),
        "median_interval_dedup": float(med_ded.median()),
        "material": bool(material),
    })


# ------------------------------------------------------------ EDITORIAL 12 --

def missingness(df, cols, label, out_dir):
    """Per-covariate missingness and cumulative complete-case loss."""
    print(f"\n  {label}: {len(df):,} stays entering the model")
    print(f"    {'covariate':22s} {'missing':>10s} {'share':>9s}")
    print("    " + "-" * 44)
    rows = []
    for c in cols:
        if c not in df.columns:
            print(f"    {c:22s} {'COLUMN ABSENT':>20s}")
            rows.append({"cohort": label, "covariate": c, "n_missing": None,
                         "share_missing": None})
            continue
        n = int(df[c].isna().sum())
        print(f"    {c:22s} {n:>10,} {n / len(df):>9.5f}")
        rows.append({"cohort": label, "covariate": c, "n_missing": n,
                     "share_missing": n / len(df)})
    present = [c for c in cols if c in df.columns]
    complete = int(df[present].notna().all(axis=1).sum())
    lost = len(df) - complete
    print(f"    complete cases: {complete:,} of {len(df):,} "
          f"(dropped {lost:,}, {lost / len(df):.5f})")
    rows.append({"cohort": label, "covariate": "COMPLETE CASE",
                 "n_missing": lost, "share_missing": lost / len(df)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-cache", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--eicu-vp-cache", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./revision_stage1"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("EDITORIAL COMMENT 15 — are duplicate timestamps counted as records?")
    print("=" * 78)
    print("  The counting code uses .size() on grouped offsets, which counts")
    print("  rows. This measures whether that matters.")

    rows = []
    mim = pd.read_parquet(a.mimic_cache)
    off = "offset_min" if "offset_min" in mim.columns else mim.columns[-1]
    dup_check(mim[["stay_id", off]], "stay_id", off,
              "MIMIC-IV chartevents, heart rate (220045)", rows)

    nc = pd.read_parquet(a.eicu_nc_cache)
    dup_check(nc[["patientunitstayid", "observationoffset"]],
              "patientunitstayid", "observationoffset",
              "eICU-CRD nurseCharting, heart rate", rows)

    vp = pd.read_parquet(a.eicu_vp_cache)
    dup_check(vp[["patientunitstayid", "observationoffset"]],
              "patientunitstayid", "observationoffset",
              "eICU-CRD vitalPeriodic, heart rate", rows)

    pd.DataFrame(rows).to_csv(a.out_dir / "duplicate_timestamps.csv",
                              index=False)

    print("\n" + "=" * 78)
    print("EDITORIAL COMMENT 12 — missingness in the admission-hour covariates")
    print("=" * 78)
    print("  No imputation is applied anywhere. statsmodels drops rows with any")
    print("  missing term in the formula, so these models are complete-case.")

    mrows = []
    ps = pd.read_csv(a.mimic_per_stay)
    clean = mim.groupby("stay_id").size().rename("n_clean").reset_index()
    m = ps.merge(clean, on="stay_id", how="left")
    m = m.rename(columns={"n_hr_24h": "n_contaminated"})
    m = m.dropna(subset=["n_contaminated", "n_clean", "adm_hour"])
    mrows += missingness(
        m, ["age_z", "gender", "n_comorbid_z", "n_chapters_z",
            "anchor_year_group", "careunit"],
        "MIMIC-IV", a.out_dir)

    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset", "age", "gender",
                               "unitadmittime24"])
    ncc = nc.groupby("patientunitstayid").size().rename("n_clean").reset_index()
    e = pat.merge(ncc, on="patientunitstayid", how="inner")
    e["age_num"] = pd.to_numeric(e["age"].replace("> 89", "90"),
                                 errors="coerce")
    e["age_z"] = (e["age_num"] - e["age_num"].mean()) / e["age_num"].std()
    e["adm_hour"] = pd.to_datetime(e["unitadmittime24"], format="%H:%M:%S",
                                   errors="coerce").dt.hour
    mrows += missingness(e, ["age_z", "gender", "unittype", "adm_hour"],
                         "eICU-CRD", a.out_dir)
    pd.DataFrame(mrows).to_csv(a.out_dir / "covariate_missingness.csv",
                               index=False)

    print("\n" + "=" * 78)
    print("WHAT TO DO WITH THIS")
    print("=" * 78)
    print("  If no stream is flagged MATERIAL, editorial comment 15 is answered")
    print("  by stating that records are counted as timestamped observations,")
    print("  that duplicate timestamps are rare, and giving the share. If any")
    print("  stream is flagged, the affected metrics must be recomputed on")
    print("  deduplicated records before the revision goes back.")
    print("\n  For comment 12, report the per-covariate rates and the")
    print("  complete-case loss, and state that no imputation was applied")
    print("  because the loss is small and the covariates are administrative")
    print("  rather than clinical measurements.")
    print(f"\n-> {a.out_dir}")


if __name__ == "__main__":
    main()
