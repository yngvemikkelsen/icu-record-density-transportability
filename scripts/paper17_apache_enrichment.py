#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Paper 17 — the APACHE availability sentence, verified.

The Discussion states:

    "The APACHE physiology table was unavailable for 23% of eICU-CRD stays;
     those missing-APACHE stays were approximately sixfold enriched in the
     exposure bottom decile (30.8% vs 5.1%). Restricting analyses to stays
     with APACHE data would therefore induce exposure-related selection."

Both figures trace to a docstring in paper17_probe_sofa_coverage.py describing
the pre-probe state of the analysis, not to any deposited computation. The same
docstring gives urine as 42% and electivesurgery as 17%, which are the values on
the 200,859 patient table; the manuscript carries the probe's corrected 49.1%
and 19.4% on the 171,177 APACHE frame. So for those two the manuscript was
updated and for APACHE availability it was not.

This script measures both quantities. The availability share is simply stays
with no apacheApsVar row. The enrichment is the share of each group falling in
the exposure's bottom decile, where the exposure is the count of nurse-charted
heart-rate records in the first 24 hours.

The decile cut depends on the frame and on whether stays with no nurse-charted
record are retained, so every combination is printed and the one reproducing
30.8% vs 5.1% identifies itself. If none does, the sentence has no reproducible
basis and should be restated on a frame that can be named.

Usage:
  python paper17_apache_enrichment.py \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/apache_enrichment
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_MIN = 24 * 60
ROWS = []


def find(root: Path, stem: str):
    t = stem.lower()
    for p in root.iterdir():
        if p.name.lower().startswith(t + ".csv"):
            return p
    raise SystemExit(f"table not found: {stem} under {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./apache_enrichment"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    pat = pd.read_csv(find(a.eicu_root, "patient"),
                      usecols=["patientunitstayid", "hospitalid",
                               "unitdischargeoffset"])
    aps = pd.read_csv(find(a.eicu_root, "apacheApsVar"),
                      usecols=["patientunitstayid"])
    have = set(aps["patientunitstayid"].unique())
    print("=" * 78)
    print(f"  patient {len(pat):,} | apacheApsVar distinct stays {len(have):,}")
    print("=" * 78, flush=True)

    nc = pd.read_parquet(a.eicu_nc_cache)
    nc = nc[(nc["observationoffset"] >= 0)
            & (nc["observationoffset"] <= WINDOW_MIN)]
    cnt = (nc.groupby("patientunitstayid").size()
             .rename("n_records").reset_index())
    d = pat.merge(cnt, on="patientunitstayid", how="left")
    d["n_records"] = d["n_records"].fillna(0).astype(int)
    d["has_aps"] = d["patientunitstayid"].isin(have)

    cw = d[d["unitdischargeoffset"] >= WINDOW_MIN]
    elig = cw[cw["n_records"] >= 3]
    big = elig.groupby("hospitalid").size()
    big = big[big >= 500].index
    unres = elig[elig["hospitalid"].isin(big)]
    hm = unres.groupby("hospitalid")["n_records"].median()
    restr = unres[~unres["hospitalid"].isin(hm[hm < 12].index)]

    FRAMES = [("whole patient table", d),
              ("completed-window set", cw),
              ("eligible-stay set", elig),
              ("unrestricted cohort", unres),
              ("restricted primary cohort", restr)]

    print("\n1. APACHE PHYSIOLOGY UNAVAILABLE, BY FRAME   [manuscript: 23%]")
    print("-" * 78, flush=True)
    for lab, f in FRAMES:
        m = int((~f["has_aps"]).sum())
        print(f"    {lab:<34s} {m:>7,}/{len(f):>8,} = {m/len(f):6.1%}",
              flush=True)
        ROWS.append({"question": "APACHE unavailable", "frame": lab,
                     "value": f"{m}/{len(f)}", "pct": round(100*m/len(f), 2),
                     "manuscript": "23%"})

    print("\n2. BOTTOM-DECILE ENRICHMENT AMONG MISSING-APACHE STAYS")
    print("   [manuscript: 30.8% vs 5.1%, approximately sixfold]")
    print("-" * 78, flush=True)
    print(f"    {'frame':<30s} {'cut':>5s} {'missing':>9s} {'present':>9s} "
          f"{'ratio':>7s}", flush=True)
    for lab, f in FRAMES:
        if len(f) < 100:
            continue
        cut = f["n_records"].quantile(0.10)
        low = f["n_records"] <= cut
        miss, pres = ~f["has_aps"], f["has_aps"]
        if miss.sum() == 0 or pres.sum() == 0:
            continue
        a_ = float(low[miss].mean())
        b_ = float(low[pres].mean())
        r = a_ / b_ if b_ else np.nan
        print(f"    {lab:<30s} {cut:>5.0f} {a_:>8.1%} {b_:>8.1%} "
              f"{r:>7.2f}", flush=True)
        ROWS.append({"question": "bottom-decile enrichment", "frame": lab,
                     "value": f"{a_:.3%} vs {b_:.3%}", "pct": round(r, 2),
                     "manuscript": "30.8% vs 5.1%"})

    print("\n   same, with the decile cut taken on the eligible-stay set and")
    print("   applied to each frame (a fixed cutoff rather than a per-frame one)",
          flush=True)
    fixed = elig["n_records"].quantile(0.10)
    print(f"    fixed cutoff = {fixed:.0f} records", flush=True)
    for lab, f in FRAMES:
        low = f["n_records"] <= fixed
        miss, pres = ~f["has_aps"], f["has_aps"]
        if miss.sum() == 0 or pres.sum() == 0:
            continue
        a_ = float(low[miss].mean()); b_ = float(low[pres].mean())
        r = a_ / b_ if b_ else np.nan
        print(f"    {lab:<30s}       {a_:>8.1%} {b_:>8.1%} {r:>7.2f}",
              flush=True)
        ROWS.append({"question": "bottom-decile enrichment, fixed cut",
                     "frame": lab, "value": f"{a_:.3%} vs {b_:.3%}",
                     "pct": round(r, 2), "manuscript": "30.8% vs 5.1%"})

    out = pd.DataFrame(ROWS)
    out.to_csv(a.out_dir / "apache_enrichment.csv", index=False)
    print("\n" + "=" * 78)
    print(f"{len(out)} rows -> {a.out_dir / 'apache_enrichment.csv'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
