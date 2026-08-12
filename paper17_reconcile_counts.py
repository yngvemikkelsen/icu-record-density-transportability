#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy"]
# ///
"""
Reconcile the competing definitions of "HR records in the first 24 h".

Three counts exist or will exist:
  A  per_stay_multi_outcomes.csv : n_hr_24h          <- the manuscript exposure
  B  physiology_24h.csv          : hr_n              <- built alongside the acuity covariates
  C  mimic_physiology_v2.csv     : n_hr_24h_rebuilt  <- built by paper17_build_physiology_v2.py

A and B disagree (verified on the uploaded files: 0.9% exact agreement,
hr_n <= n_hr_24h in 100% of stays, median gap 6, bottom-decile cutoffs 22 vs 17,
Jaccard 0.76).  Table 1, the percentile-matched exposure and the acuity
covariates therefore do not come from one extraction.  One definition has to be
declared authoritative and used throughout.

This script quantifies the disagreement; it does not resolve it.  Resolution
requires looking at the two extraction scripts and deciding which window and
which value filter is intended.

Usage:
  python paper17_reconcile_counts.py \
      --per-stay   ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --physio     ~/bcst/mimic_oasis_adjustment_results/physiology_24h.csv \
      [--physio-v2 ~/bcst/physiology_v2/mimic_physiology_v2.csv] \
      --out-dir    ~/bcst/reconcile
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def pairwise(df, a, b, out_lines):
    s = df.dropna(subset=[a, b])
    if s.empty:
        out_lines.append(f"{a} vs {b}: no overlapping rows")
        return None
    diff = s[a] - s[b]
    ratio = s[b] / s[a].replace(0, np.nan)
    lo_a = s[a] <= s[a].quantile(0.10)
    lo_b = s[b] <= s[b].quantile(0.10)
    jac = float((lo_a & lo_b).sum() / max((lo_a | lo_b).sum(), 1))
    rec = {
        "count_a": a, "count_b": b, "n": int(len(s)),
        "exact_agreement": float((s[a] == s[b]).mean()),
        "pearson": float(s[a].corr(s[b])),
        "mean_a": float(s[a].mean()), "mean_b": float(s[b].mean()),
        "median_diff": float(diff.median()),
        "iqr_diff": f"{diff.quantile(.25):.0f}-{diff.quantile(.75):.0f}",
        "share_b_le_a": float((s[b] <= s[a]).mean()),
        "median_ratio_b_over_a": float(ratio.median()),
        "p10_cutoff_a": float(s[a].quantile(0.10)),
        "p10_cutoff_b": float(s[b].quantile(0.10)),
        "bottom_decile_jaccard": jac,
    }
    out_lines.append(
        f"\n{a} vs {b}   (n={rec['n']:,})\n"
        f"  exact agreement      {rec['exact_agreement']:.4f}   pearson {rec['pearson']:.4f}\n"
        f"  mean                 {rec['mean_a']:.2f} vs {rec['mean_b']:.2f}\n"
        f"  {a} - {b}            median {rec['median_diff']:.0f}  IQR {rec['iqr_diff']}\n"
        f"  share {b} <= {a}     {rec['share_b_le_a']:.4f}   median ratio {rec['median_ratio_b_over_a']:.3f}\n"
        f"  bottom-10% cutoff    {rec['p10_cutoff_a']:.0f} vs {rec['p10_cutoff_b']:.0f}\n"
        f"  bottom-decile set overlap (Jaccard)  {jac:.3f}"
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stay", required=True, type=Path)
    ap.add_argument("--physio", required=True, type=Path)
    ap.add_argument("--physio-v2", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./reconcile"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.per_stay, usecols=["stay_id", "n_hr_24h"])
    p = pd.read_csv(args.physio, usecols=["stay_id", "hr_n"])
    d = d.merge(p, on="stay_id", how="left")
    cols = ["n_hr_24h", "hr_n"]

    if args.physio_v2 and args.physio_v2.exists():
        v2 = pd.read_csv(args.physio_v2, usecols=["stay_id", "n_hr_24h_rebuilt"])
        d = d.merge(v2, on="stay_id", how="left")
        cols.append("n_hr_24h_rebuilt")
    else:
        print("(no --physio-v2 supplied or file absent; comparing A vs B only)")

    lines = [f"stays: {len(d):,}"]
    for c in cols:
        lines.append(f"  {c:20s} present {d[c].notna().sum():,}  "
                     f"missing {d[c].isna().sum():,}")

    recs = [r for a, b in itertools.combinations(cols, 2)
            if (r := pairwise(d, a, b, lines)) is not None]

    report = "\n".join(lines)
    print(report)
    (args.out_dir / "count_reconciliation.txt").write_text(report + "\n")
    pd.DataFrame(recs).to_csv(args.out_dir / "count_reconciliation.csv",
                              index=False)

    print(f"\n-> {args.out_dir}/count_reconciliation.{{txt,csv}}")
    print("\nDECISION REQUIRED: declare one count authoritative and rebuild "
          "Table 1, the percentile cutoffs and the residualisation on it. "
          "A near-constant one-sided gap points to a window boundary or a "
          "null/range filter, not a different itemid set.")


if __name__ == "__main__":
    main()
