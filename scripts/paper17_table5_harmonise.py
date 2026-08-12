#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Paper 17: put the Table 5 variance components on identical footing.

THE PROBLEM
-----------
The heart-rate hospital eta-squared reported in Table 3 (0.473) is estimated
across every eICU hospital contributing eligible stays (119,317 stays).  The
replication values for oxygen saturation (0.421) and respiratory rate (0.424)
were computed on hospitals contributing at least 500 nurse-stream stays, because
that filter is applied inside the cross-variable script.  Table 5 sets them
beside one another, so the comparison is not like for like.

This script recomputes hospital and unit-within-hospital eta-squared for all
three variables under BOTH cohort definitions, so Table 5 can be stated on one
footing and the sensitivity of the estimate to the threshold is visible rather
than buried.

Nothing else is recomputed.  Table 3 continues to report the all-hospital
estimate with its bootstrap interval; this script only supplies the matched
columns for the replication table.

Usage:
  python paper17_table5_harmonise.py \
      --hr-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --cross-dir ~/bcst/cross_vitals \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/table5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
    return out.reset_index()


def components(d, label, cohort):
    row = {"variable": label, "cohort": cohort, "n_stays": len(d),
           "n_hospitals": int(d["hospitalid"].nunique())}
    for m in ("n_records", "median_interval_min"):
        eh = eta2(d[m], d["hospitalid"])
        eu = eta2(d[m], d["unit_id"]) - eh
        row[f"hosp_{m}"] = eh
        row[f"unit_{m}"] = eu
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr-cache", required=True, type=Path)
    ap.add_argument("--cross-dir", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./table5"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pat = pd.read_csv(args.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])

    frames = {}
    frames["heart_rate"] = metrics(
        pd.read_parquet(args.hr_cache), "patientunitstayid", "observationoffset")
    cross = pd.read_parquet(args.cross_dir / "eicu_nc.parquet")
    for var in sorted(cross["variable"].unique()):
        frames[var] = metrics(cross[cross["variable"] == var]
                              [["patientunitstayid", "offset_min"]],
                              "patientunitstayid", "offset_min")

    rows = []
    print("=" * 78)
    print("HOSPITAL AND UNIT-WITHIN-HOSPITAL eta^2, BOTH COHORT DEFINITIONS")
    print("=" * 78)
    for var, m in frames.items():
        d = m.merge(pat, on="patientunitstayid", how="inner")
        d = d[(d["unitdischargeoffset"] >= WINDOW_MIN)
              & (d["n_records"] >= MIN_RECORDS)].copy()
        d["unit_id"] = ("eICU-" + d["hospitalid"].astype(str) + ":"
                        + d["unittype"].astype(str))
        rows.append(components(d, var, "all eligible hospitals"))
        cnt = d["hospitalid"].value_counts()
        big = d[d["hospitalid"].isin(cnt[cnt >= MIN_HOSP].index)]
        rows.append(components(big, var, f">={MIN_HOSP} stays"))

    out = pd.DataFrame(rows)
    print(f"\n{'variable':12s} {'cohort':24s} {'stays':>8s} {'hosp':>5s} "
          f"{'H count':>8s} {'U count':>8s} {'H interval':>11s} "
          f"{'U interval':>11s}")
    print("-" * 78)
    for _, r in out.iterrows():
        print(f"{r['variable']:12s} {r['cohort']:24s} {r['n_stays']:8,d} "
              f"{r['n_hospitals']:5d} {r['hosp_n_records']:8.3f} "
              f"{r['unit_n_records']:8.3f} "
              f"{r['hosp_median_interval_min']:11.3f} "
              f"{r['unit_median_interval_min']:11.3f}")
    out.to_csv(args.out_dir / "table5_components.csv", index=False)

    print("\nUse ONE cohort definition down each column of Table 5. The "
          "all-eligible rows match the basis of the Table 3 heart-rate "
          "estimate; the restricted rows match the basis on which the "
          "per-hospital summaries and the mixed model were computed.")
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
