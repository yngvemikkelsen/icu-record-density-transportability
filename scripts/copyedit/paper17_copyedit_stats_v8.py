#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Paper 17 — copyediting statistics, v8. The last three numerators.

v7 crashed: electivesurgery is not a column of the eICU patient table. It is an
APACHE-derived field and lives in apachePredVar, which is consistent with the
sentence it appears in, alongside the APACHE physiology availability figure.

v8 reads it from apachePredVar, resolves every table name case-insensitively,
and on a missing column prints the columns that are present rather than
raising, so a name mismatch costs a line of output instead of a rerun.

  urine output                     49.1%   severity build, published definition
  elective-surgery status          19.4%   apachePredVar.electivesurgery
  APACHE physiology unavailable    23%     stays absent from apacheApsVar

Denominator throughout is the eICU patient table, 200,859, which is what every
other eICU coverage figure resolves on exactly.

Usage:
  python paper17_copyedit_stats_v8.py \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --severity-dir ~/bcst/severity_v3 \
      --out-dir ~/bcst/copyedit_stats_v8
"""

import argparse
from pathlib import Path

import pandas as pd

ROWS = []


def rec(label, value, expect=""):
    ROWS.append({"label": label, "value": value, "manuscript_says": expect})
    e = f"   [manuscript: {expect}]" if expect else ""
    print(f"    {label:<46s} {value}{e}", flush=True)


def find(root: Path, stem: str):
    """eICU ships mixed-case names; match on the lowercased stem."""
    target = stem.lower()
    for p in root.iterdir():
        n = p.name.lower()
        if n.startswith(target + ".csv"):
            return p
    return None


def load(root, stem, cols=None):
    p = find(root, stem)
    if p is None:
        print(f"  table not found: {stem}.csv[.gz] under {root}", flush=True)
        return None
    if cols:
        head = pd.read_csv(p, nrows=0)
        missing = [c for c in cols if c not in head.columns]
        if missing:
            print(f"  {p.name}: missing {missing}", flush=True)
            print(f"    columns present: {list(head.columns)}", flush=True)
            cols = [c for c in cols if c in head.columns]
            if not cols:
                return None
    print(f"  reading {p.name}", flush=True)
    return pd.read_csv(p, usecols=cols) if cols else pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--severity-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./copyedit_stats_v8"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    pat = load(a.eicu_root, "patient", ["patientunitstayid"])
    if pat is None:
        raise SystemExit("patient table is required")
    N = len(pat)
    print(f"  eICU patient table: {N:,}   [denominator: 200,859]")
    print("=" * 78)

    # ---- elective surgery, from apachePredVar ------------------------------
    print("\nELECTIVE SURGERY", flush=True)
    apv = load(a.eicu_root, "apachePredVar", ["patientunitstayid",
                                              "electivesurgery"])
    if apv is not None and "electivesurgery" in apv.columns:
        es = apv["electivesurgery"]
        n_any = int(es.notna().sum())
        rec("value recorded", f"{n_any:,}/{N:,} = {n_any/N:.1%}", "19.4%")
        for v in sorted(es.dropna().unique()):
            k = int((es == v).sum())
            rec(f"  electivesurgery == {v!r}", f"{k:,}/{N:,} = {k/N:.1%}")

    # ---- urine output ------------------------------------------------------
    print("\nURINE OUTPUT", flush=True)
    sev = a.severity_dir / "eicu_severity_v3.csv"
    if sev.exists():
        head = pd.read_csv(sev, nrows=0)
        if "urine_24h" in head.columns:
            s = pd.read_csv(sev, usecols=["patientunitstayid", "urine_24h"])
            n_u = int(s["urine_24h"].notna().sum())
            print(f"  severity rows: {len(s):,}", flush=True)
            rec("urine output recorded",
                f"{n_u:,}/{N:,} = {n_u/N:.1%}", "49.1%")
        else:
            print(f"  urine_24h absent; columns: {list(head.columns)}",
                  flush=True)
    else:
        print(f"  not found: {sev}", flush=True)

    # ---- APACHE physiology availability ------------------------------------
    print("\nAPACHE PHYSIOLOGY", flush=True)
    aps = load(a.eicu_root, "apacheApsVar", ["patientunitstayid"])
    if aps is not None:
        have = aps["patientunitstayid"].nunique()
        miss = N - have
        print(f"  apacheApsVar rows {len(aps):,}, distinct stays {have:,}",
              flush=True)
        rec("APACHE physiology unavailable",
            f"{miss:,}/{N:,} = {miss/N:.1%}", "23%")

    out = pd.DataFrame(ROWS)
    out.to_csv(a.out_dir / "copyedit_statistics_v8.csv", index=False)
    print("\n" + "=" * 78)
    print(f"{len(out)} statistics -> {a.out_dir / 'copyedit_statistics_v8.csv'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
