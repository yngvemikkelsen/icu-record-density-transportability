#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Paper 17 — copyediting statistics, v5. The last three gaps.

Resolved and NOT repeated here:
  eICU excluded/retained length of stay  2.20 (IQR 1.56-3.87) / 2.33 (1.59-4.08)
  MIMIC retained length of stay          2.41 (IQR 1.57-4.48)
  Gaps beyond 2 hours, zeros             65,301/88,560 = 73.7%
  Observation item per-stay SD           25.91 (SD 9.58), already in the v1 run
                                         off hr_timestamps.parquet

Three defects in v3 are fixed:

  1. The MIMIC excluded set came out as 12, not 84, because the frame was built
     from the heart-rate cache and so contained only stays with at least one
     record. The published split is 72 with no record plus 12 with one or two,
     of 67,224; 67,152 + 72 = 67,224 confirms the 72 were missing. The frame is
     now built from icustays, with cache absence counted as zero.

  2. '26 for the cohort' is the median of per-hospital median counts, not the
     median over stays, which returned 31. Both are reported on both cohorts.

  3. The chartevents scan was not restricted to the first 24 hours, so the
     observation item returned 92.68 records per stay against the published
     25.92. The window filter from paper17_diagnose_exposure.py is restored:
     merge icustays intime, take (charttime - intime) in [0, 24) hours. This
     needs charttime, so the scan is slower than v4 attempted; progress is
     printed every chunk.

Usage:
  python paper17_copyedit_stats_v5.py \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --mimic-root ~/physionet.org/files/mimiciv/3.1 \
      --sofa-dir ~/bcst/sofa_probe \
      --out-dir ~/bcst/copyedit_stats_v5
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_MIN = 24 * 60
WINDOW_H = 24.0
CHUNKSIZE = 5_000_000
ROWS = []
HR_ITEM, ALARM_HI, ALARM_LO = 220045, 220046, 220047


def rec(sec, label, value, expect=""):
    ROWS.append({"section": sec, "label": label, "value": value,
                 "manuscript_says": expect})
    e = f"   [manuscript: {expect}]" if expect else ""
    print(f"    {label:<50s} {value}{e}", flush=True)


def med_iqr(s, dp=0):
    s = pd.Series(s).dropna()
    if not len(s):
        return "n/a"
    return (f"median {s.median():.{dp}f} "
            f"(IQR {s.quantile(0.25):.{dp}f}-{s.quantile(0.75):.{dp}f})")


def mean_sd(s):
    s = pd.Series(s).dropna()
    return "n/a" if not len(s) else f"mean {s.mean():.4f} (SD {s.std(ddof=1):.4g})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--mimic-root", required=True, type=Path)
    ap.add_argument("--mimic-cache", type=Path)
    ap.add_argument("--sofa-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./copyedit_stats_v5"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------- 1. MIMIC excluded 84 ----------
    print("=" * 78)
    print("1. MIMIC EXCLUDED STAYS, THE FULL 84")
    print("=" * 78)
    icu = pd.read_csv(a.mimic_root / "icu" / "icustays.csv.gz",
                      usecols=["subject_id", "hadm_id", "stay_id",
                               "intime", "outtime", "los"],
                      parse_dates=["intime", "outtime"])
    icu = icu.sort_values(["hadm_id", "intime"])
    icu["rk"] = icu.groupby("hadm_id")["intime"].rank(method="first")
    coh = icu[(icu["rk"] == 1) & icu["intime"].notna() & icu["outtime"].notna()]
    print(f"  MIMIC cohort: {len(coh):,}   [manuscript: 85,228]", flush=True)
    cw = coh[coh["los"] >= 1.0]
    print(f"  completed window: {len(cw):,}   [manuscript: 67,224]", flush=True)

    if a.mimic_cache and a.mimic_cache.exists():
        mc = pd.read_parquet(a.mimic_cache)
        moff = "offset_min" if "offset_min" in mc.columns else mc.columns[-1]
        mc = mc[(mc[moff] >= 0) & (mc[moff] <= WINDOW_MIN)]
        cnt = mc.groupby("stay_id").size()
    else:
        raise SystemExit("--mimic-cache is required for section 1")
    cw = cw.assign(n_records=cw["stay_id"].map(cnt).fillna(0).astype(int))
    n0 = int((cw["n_records"] == 0).sum())
    n12 = int(((cw["n_records"] >= 1) & (cw["n_records"] <= 2)).sum())
    print(f"  no record {n0:,} | one or two {n12:,} | total "
          f"{n0+n12:,}   [manuscript: 72 | 12 | 84]", flush=True)
    exc = cw[cw["n_records"] < 3]
    ret = cw[cw["n_records"] >= 3]
    rec("1", "MIMIC excluded (84): ICU length of stay (d)",
        med_iqr(exc["los"], dp=2), "median 6.53")
    rec("1", "MIMIC retained: ICU length of stay (d)",
        med_iqr(ret["los"], dp=2), "median 2.41")

    # ---------------------------------------- 2. '26 for the cohort' -------
    print("\n" + "=" * 78)
    print("2. WHAT IS '26 FOR THE COHORT'?")
    print("=" * 78)
    nc = pd.read_parquet(a.eicu_nc_cache)
    nc = nc[(nc["observationoffset"] >= 0)
            & (nc["observationoffset"] <= WINDOW_MIN)]
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid",
                               "unitdischargeoffset"])
    e_cnt = (nc.groupby("patientunitstayid").size()
               .rename("n_records").reset_index())
    cwe = pat[pat["unitdischargeoffset"] >= WINDOW_MIN].merge(
        e_cnt, on="patientunitstayid", how="left")
    cwe["n_records"] = cwe["n_records"].fillna(0).astype(int)
    elig = cwe[cwe["n_records"] >= 3]
    big = elig.groupby("hospitalid").size()
    big = big[big >= 500].index
    unres = elig[elig["hospitalid"].isin(big)]
    hmu = unres.groupby("hospitalid")["n_records"].median()
    restr = unres[~unres["hospitalid"].isin(hmu[hmu < 12].index)]
    hmr = restr.groupby("hospitalid")["n_records"].median()
    print(f"  unrestricted {len(unres):,}/{unres['hospitalid'].nunique()} | "
          f"restricted {len(restr):,}/{restr['hospitalid'].nunique()}"
          f"   [96,518/68 | 88,560/60]", flush=True)
    rec("2", "median of per-hospital medians, unrestricted", med_iqr(hmu), "26")
    rec("2", "median of per-hospital medians, restricted", med_iqr(hmr), "26")
    rec("2", "median over stays, unrestricted",
        med_iqr(unres["n_records"]), "26")
    rec("2", "median over stays, restricted",
        med_iqr(restr["n_records"]), "26")

    # ------------------------------- 3. item profile, windowed --------------
    print("\n" + "=" * 78)
    print("3. MIMIC-IV ITEM PROFILE, FIRST 24 HOURS")
    print("=" * 78)
    src = a.mimic_root / "icu" / "chartevents.csv.gz"
    want = {HR_ITEM, ALARM_HI, ALARM_LO}
    stays = icu[["stay_id", "intime"]]
    kept, scanned, t0 = [], 0, time.time()
    for i, ch in enumerate(pd.read_csv(
            src, usecols=["stay_id", "itemid", "charttime"],
            parse_dates=["charttime"], chunksize=CHUNKSIZE), start=1):
        scanned += len(ch)
        ch = ch[ch["itemid"].isin(want)]
        if not ch.empty:
            ch = ch.merge(stays, on="stay_id", how="inner")
            h = (ch["charttime"] - ch["intime"]).dt.total_seconds() / 3600.0
            ch = ch[(h >= 0) & (h < WINDOW_H)]
            if not ch.empty:
                kept.append(ch[["stay_id", "itemid"]])
        print(f"    chunk {i:>3d}  scanned {scanned:>12,}  "
              f"{time.time()-t0:6.0f}s", flush=True)
    ev = pd.concat(kept, ignore_index=True)
    print(f"  in-window rows for the three items: {len(ev):,}", flush=True)
    for item, lab, exp in ((HR_ITEM, "observation 220045", "25.92"),
                           (ALARM_HI, "alarm high 220046", "2.8"),
                           (ALARM_LO, "alarm low 220047", "2.8")):
        per = ev[ev["itemid"] == item].groupby("stay_id").size()
        print(f"  {lab}: {len(per):,} stays"
              f"   [per_item_volume.csv: 94,361 / 93,018 / 93,012]", flush=True)
        rec("3", f"{lab}: records per stay", mean_sd(per), exp)
    pooled = ev.groupby("stay_id").size()
    rec("3", "pooled three-item count", mean_sd(pooled), "31.56")
    ps = pd.read_csv(a.mimic_per_stay, usecols=["stay_id"])
    keep = set(ps["stay_id"])
    obs = ev[ev["itemid"] == HR_ITEM].groupby("stay_id").size()
    pr = pooled[pooled.index.isin(keep)]
    orr = obs[obs.index.isin(keep)]
    print(f"  reconstruction frame: {len(pr):,}   [manuscript: 85,133]",
          flush=True)
    rec("3", "pooled count, reconstruction frame", mean_sd(pr), "31.48")
    rec("3", "observation item, reconstruction frame", mean_sd(orr), "25.88")

    # ------------------------------- 4. coverage denominators ---------------
    print("\n" + "=" * 78)
    print("4. SEVERITY-COMPONENT COVERAGE")
    print("=" * 78)
    if a.sofa_dir and a.sofa_dir.exists():
        for f in ("mimic_sofa_coverage.csv", "eicu_sofa_coverage.csv"):
            p = a.sofa_dir / f
            if p.exists():
                print(f"\n  {f}", flush=True)
                print(pd.read_csv(p).to_string(index=False), flush=True)
        print("\n  Numerator = coverage x the denominator the probe used;", flush=True)
        print("  re-run paper17_probe_sofa_coverage.py to see it printed.", flush=True)
    else:
        print(f"  --sofa-dir not found; default is ~/bcst/sofa_probe", flush=True)

    out = pd.DataFrame(ROWS)
    out.to_csv(a.out_dir / "copyedit_statistics_v5.csv", index=False)
    print("\n" + "=" * 78)
    print(f"{len(out)} statistics -> {a.out_dir / 'copyedit_statistics_v5.csv'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
