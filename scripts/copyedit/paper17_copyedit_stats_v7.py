#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Paper 17 — copyediting statistics, v7. The last three numerators. Final run.

Everything else is settled. This computes only the three coverage figures that
are in neither the deposited results nor sofa_probe:

  urine output                     49.1% of eICU-CRD stays
  elective-surgery status          19.4%
  APACHE physiology unavailable    23%

All three are on the eICU patient table, N = 200,859, which is the denominator
every other eICU coverage figure resolves on exactly.

urine_24h is taken from the severity build rather than recomputed, so the
definition is the one that was published; the other two come straight from
patient.csv.gz and apacheApsVar.csv.gz.

Usage:
  python paper17_copyedit_stats_v7.py \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --severity-dir ~/bcst/severity_v3 \
      --out-dir ~/bcst/copyedit_stats_v7
"""

import argparse
from pathlib import Path

import pandas as pd

ROWS = []


def rec(label, value, expect=""):
    ROWS.append({"label": label, "value": value, "manuscript_says": expect})
    e = f"   [manuscript: {expect}]" if expect else ""
    print(f"    {label:<48s} {value}{e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--severity-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./copyedit_stats_v7"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "electivesurgery"])
    N = len(pat)
    print("=" * 78)
    print(f"eICU patient table: {N:,}   [denominator for all eICU coverage: "
          f"200,859]")
    print("=" * 78)

    # ---- elective surgery --------------------------------------------------
    es = pat["electivesurgery"]
    n_any = int(es.notna().sum())
    rec("elective surgery, value recorded",
        f"{n_any:,}/{N:,} = {n_any/N:.1%}", "19.4%")
    for v in sorted(x for x in es.dropna().unique()):
        k = int((es == v).sum())
        rec(f"  electivesurgery == {v!r}", f"{k:,}/{N:,} = {k/N:.1%}")

    # ---- urine output, from the severity build -----------------------------
    sev = a.severity_dir / "eicu_severity_v3.csv"
    if sev.exists():
        s = pd.read_csv(sev, usecols=["patientunitstayid", "urine_24h"])
        n_u = int(s["urine_24h"].notna().sum())
        print(f"\n  severity rows: {len(s):,}", flush=True)
        rec("urine output recorded (severity build)",
            f"{n_u:,}/{N:,} = {n_u/N:.1%}", "49.1%")
    else:
        print(f"\n  not found: {sev}", flush=True)

    # ---- APACHE physiology availability ------------------------------------
    aps = a.eicu_root / "apacheApsVar.csv.gz"
    if not aps.exists():
        aps = a.eicu_root / "apacheapsvar.csv.gz"
    if aps.exists():
        ap_ids = pd.read_csv(aps, usecols=["patientunitstayid"])
        have = ap_ids["patientunitstayid"].nunique()
        miss = N - have
        print(f"\n  apacheApsVar rows: {len(ap_ids):,}  distinct stays: "
              f"{have:,}", flush=True)
        rec("APACHE physiology unavailable",
            f"{miss:,}/{N:,} = {miss/N:.1%}", "23%")
    else:
        print(f"\n  apacheApsVar not found under {a.eicu_root}", flush=True)

    out = pd.DataFrame(ROWS)
    out.to_csv(a.out_dir / "copyedit_statistics_v7.csv", index=False)
    print("\n" + "=" * 78)
    print(f"{len(out)} statistics -> {a.out_dir / 'copyedit_statistics_v7.csv'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
