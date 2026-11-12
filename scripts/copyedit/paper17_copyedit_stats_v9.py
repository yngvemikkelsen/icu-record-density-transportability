#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Paper 17 — copyediting statistics, v9. One numerator left.

Settled by v8, once the denominator was identified. apacheApsVar holds 171,177
distinct stays and the severity build has exactly 171,177 rows, so the severity
coverage figures are on the APACHE frame, not the 200,859 patient table:

    urine output        84,062/171,177 = 49.1%   matches the manuscript
    elective surgery    33,256/171,177 = 19.4%   matches the manuscript

Outstanding: "The APACHE physiology table was unavailable for 23% of eICU-CRD
stays." Against the whole patient table that is 29,682/200,859 = 14.8%. The
sentence continues by comparing bottom-decile enrichment, so the share is
presumably taken within an analytic cohort. This prints the share of stays
lacking apacheApsVar on each candidate frame, and the one returning 23%
identifies itself, as the 0.5% and the 49.1% did.

Usage:
  python paper17_copyedit_stats_v9.py \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/copyedit_stats_v9
"""

import argparse
from pathlib import Path

import pandas as pd

WINDOW_MIN = 24 * 60
ROWS = []


def rec(label, value, expect="23%"):
    ROWS.append({"label": label, "value": value, "manuscript_says": expect})
    print(f"    {label:<52s} {value}   [manuscript: {expect}]", flush=True)


def find(root: Path, stem: str):
    t = stem.lower()
    for p in root.iterdir():
        if p.name.lower().startswith(t + ".csv"):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./copyedit_stats_v9"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    pat = pd.read_csv(find(a.eicu_root, "patient"),
                      usecols=["patientunitstayid", "hospitalid",
                               "unitdischargeoffset"])
    aps = pd.read_csv(find(a.eicu_root, "apacheApsVar"),
                      usecols=["patientunitstayid"])
    have = set(aps["patientunitstayid"].unique())
    print("=" * 78)
    print(f"  patient {len(pat):,} | apacheApsVar stays {len(have):,}")
    print("=" * 78)

    nc = pd.read_parquet(a.eicu_nc_cache)
    nc = nc[(nc["observationoffset"] >= 0)
            & (nc["observationoffset"] <= WINDOW_MIN)]
    cnt = (nc.groupby("patientunitstayid").size()
             .rename("n_records").reset_index())
    allp = pat.merge(cnt, on="patientunitstayid", how="left")
    allp["n_records"] = allp["n_records"].fillna(0).astype(int)
    cw = allp[allp["unitdischargeoffset"] >= WINDOW_MIN]
    elig = cw[cw["n_records"] >= 3]
    big = elig.groupby("hospitalid").size()
    big = big[big >= 500].index
    unres = elig[elig["hospitalid"].isin(big)]
    hm = unres.groupby("hospitalid")["n_records"].median()
    restr = unres[~unres["hospitalid"].isin(hm[hm < 12].index)]
    mon = cw  # completed-window, monitor-stream eligibility handled below

    print("\nSHARE OF STAYS LACKING apacheApsVar, BY FRAME", flush=True)
    for lab, d in (("whole patient table (200,859)", allp),
                   ("completed-window set (132,900)", cw),
                   ("eligible-stay set (119,317)", elig),
                   ("unrestricted cohort (96,518)", unres),
                   ("restricted primary cohort (88,560)", restr)):
        ids = d["patientunitstayid"]
        miss = int((~ids.isin(have)).sum())
        rec(lab, f"{miss:,}/{len(d):,} = {miss/len(d):.1%}")

    out = pd.DataFrame(ROWS)
    out.to_csv(a.out_dir / "copyedit_statistics_v9.csv", index=False)
    print("\n" + "=" * 78)
    print(f"{len(out)} rows -> {a.out_dir / 'copyedit_statistics_v9.csv'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
