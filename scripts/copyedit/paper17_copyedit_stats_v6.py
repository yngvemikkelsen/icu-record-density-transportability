#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Paper 17 — copyediting statistics, v6. Two loose ends. Seconds to run.

Everything else is settled. v5 reproduced the MIMIC 72|12|84 split, the
excluded/retained length of stay in both databases, and the whole item profile
(25.9229, 2.8192, 2.8209, 31.4826, 25.8869, all against their published values,
with 2,970,732 in-window records matching the manuscript exactly).

A. '26 for the cohort' is still unidentified. v5 tested four frames and got
   32, 33, 30 and 31. This tries the wider ones, including the frames that
   admit the sparse hospitals the restriction removes, and prints every
   candidate so the one that returns 26 identifies itself.

B. Three coverage figures are not in the deposited results and are not in
   sofa_probe either: urine output 49.1%, elective surgery 19.4%, and the
   APACHE physiology table unavailable for 23%. They come from the severity
   build. This reads whatever is in --severity-dir and --sofa-dir rather than
   reimplementing the definitions, which is how a figure gets silently
   redefined.

   Already settled by exact-denominator arithmetic, no rerun needed:
     bilirubin  MIMIC 41,126/94,458 (43.5%)   eICU 74,584/200,859 (37.1%)
     PaO2       MIMIC 50,986/94,458 (54.0%)
     FiO2                                     eICU 66,532/200,859 (33.1%)
   MIMIC coverage is on the 94,458-row icustays table and eICU on the
   200,859-row patient table; all ten components resolve exactly on those two.

Usage:
  python paper17_copyedit_stats_v6.py \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --severity-dir ~/bcst/severity_v3 \
      --sofa-dir ~/bcst/sofa_probe \
      --out-dir ~/bcst/copyedit_stats_v6
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_MIN = 24 * 60
ROWS = []


def rec(sec, label, value, expect=""):
    ROWS.append({"section": sec, "label": label, "value": value,
                 "manuscript_says": expect})
    e = f"   [manuscript: {expect}]" if expect else ""
    print(f"    {label:<56s} {value}{e}", flush=True)


def med_iqr(s, dp=0):
    s = pd.Series(s).dropna()
    if not len(s):
        return "n/a"
    return (f"median {s.median():.{dp}f} "
            f"(IQR {s.quantile(0.25):.{dp}f}-{s.quantile(0.75):.{dp}f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--severity-dir", type=Path)
    ap.add_argument("--sofa-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./copyedit_stats_v6"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    nc = pd.read_parquet(a.eicu_nc_cache)
    nc = nc[(nc["observationoffset"] >= 0)
            & (nc["observationoffset"] <= WINDOW_MIN)]
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid",
                               "unitdischargeoffset"])
    cnt = (nc.groupby("patientunitstayid").size()
             .rename("n_records").reset_index())

    print("=" * 78)
    print("A. WHICH FRAME GIVES '26 FOR THE COHORT'?")
    print("=" * 78)
    allp = pat.merge(cnt, on="patientunitstayid", how="left")
    allp["n_records"] = allp["n_records"].fillna(0).astype(int)
    cw = allp[allp["unitdischargeoffset"] >= WINDOW_MIN]
    elig = cw[cw["n_records"] >= 3]
    big500 = elig.groupby("hospitalid").size()
    big500 = big500[big500 >= 500].index
    unres = elig[elig["hospitalid"].isin(big500)]
    hmu = unres.groupby("hospitalid")["n_records"].median()
    restr = unres[~unres["hospitalid"].isin(hmu[hmu < 12].index)]
    cw500 = cw.groupby("hospitalid").size()
    cw500 = cw500[cw500 >= 500].index

    frames = [
        ("eligible-stay set, all 193 hospitals: per-hospital medians",
         elig.groupby("hospitalid")["n_records"].median()),
        ("eligible-stay set, all 193 hospitals: over stays",
         elig["n_records"]),
        ("completed-window set, zeros retained: over stays",
         cw["n_records"]),
        ("completed-window set, zeros retained: per-hospital medians",
         cw.groupby("hospitalid")["n_records"].median()),
        ("eligibility-sensitivity set, 80 hospitals >=500 completed-window",
         cw[cw["hospitalid"].isin(cw500)].groupby("hospitalid")["n_records"].median()),
        ("whole patient table, zeros retained: over stays",
         allp["n_records"]),
        ("whole patient table: per-hospital medians",
         allp.groupby("hospitalid")["n_records"].median()),
        ("unrestricted cohort: per-hospital medians", hmu),
        ("restricted cohort: per-hospital medians",
         restr.groupby("hospitalid")["n_records"].median()),
    ]
    for lab, s in frames:
        rec("A", lab, med_iqr(s), "26")

    print("\n" + "=" * 78)
    print("B. REMAINING COVERAGE FIGURES")
    print("=" * 78)
    print("  Looking for urine output 49.1%, elective surgery 19.4%,")
    print("  APACHE physiology unavailable 23%.", flush=True)
    for d, nm in ((a.severity_dir, "severity"), (a.sofa_dir, "sofa")):
        if not d:
            continue
        if not d.exists():
            print(f"\n  {nm} dir not found: {d}", flush=True)
            continue
        print(f"\n  {nm} dir: {d}", flush=True)
        for p in sorted(d.glob("*.csv")):
            try:
                df = pd.read_csv(p, nrows=40)
            except Exception as e:
                print(f"    {p.name}: unreadable ({e})", flush=True)
                continue
            cols = [c for c in df.columns
                    if any(k in c.lower() for k in
                           ("cover", "share", "avail", "frac", "pct", "prop"))]
            print(f"    {p.name}  cols={list(df.columns)[:8]}", flush=True)
            if cols:
                print(df[[c for c in df.columns
                          if c in cols or df[c].dtype == object][:4]]
                      .to_string(index=False), flush=True)

    print("\n  Denominators, if a share needs converting:")
    print("    MIMIC coverage is on icustays, N = 94,458")
    print("    eICU coverage is on the patient table, N = 200,859")

    out = pd.DataFrame(ROWS)
    out.to_csv(a.out_dir / "copyedit_statistics_v6.csv", index=False)
    print("\n" + "=" * 78)
    print(f"{len(out)} statistics -> {a.out_dir / 'copyedit_statistics_v6.csv'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
