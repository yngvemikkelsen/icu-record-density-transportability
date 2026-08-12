#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Paper 17: correct chance comparator for nurse-versus-monitor stream agreement.

THE PROBLEM
-----------
Section 3.9 established that a percentile rule applied to integer record counts
does not flag exactly that percentile, because the rule includes every stay tied
at the cutoff value. The same rule was used for the stream-agreement analysis in
section 3.2, where observed positive agreement was compared against a fixed
chance rate of 0.100.

That comparator is wrong unless both streams flag exactly 10%. Under
independence, the expected share of monitor-flagged stays that are also
nurse-flagged is the NURSE stream's actual marginal, not the nominal percentile.
The nurse stream has counts near 28 and therefore many ties at the cutoff; the
monitor stream has counts near 280 and few. The two marginals will differ.

This script recomputes, for each of the three variables, the actual marginals of
both rules and the correct chance comparator, and reports the observed agreement
against it. It changes no other analysis.

Usage:
  python paper17_chance_comparator.py \
      --hr-nc ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --hr-vp ~/bcst/unit_profile_eicu/vitalperiodic_offsets.parquet \
      --cross-dir ~/bcst/cross_vitals \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/chance_comparator
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_MIN = 24 * 60
MIN_RECORDS = 3
PCTS = [5, 10, 20]


def counts_from(df, key):
    return df.groupby(key).size().rename("n").reset_index()


def analyse(vp, nc, label, pat, rows):
    """vp, nc: DataFrames with patientunitstayid and n."""
    d = (vp.rename(columns={"n": "n_vp"})
           .merge(nc.rename(columns={"n": "n_nc"}), on="patientunitstayid")
           .merge(pat, on="patientunitstayid", how="inner"))
    d = d[(d["unitdischargeoffset"] >= WINDOW_MIN)
          & (d["n_vp"] >= MIN_RECORDS) & (d["n_nc"] >= MIN_RECORDS)]
    n = len(d)
    print(f"\n{label}: {n:,} stays present in both streams")
    print(f"  Spearman {d['n_vp'].rank().corr(d['n_nc'].rank()):+.4f}")
    print(f"  {'pct':>4s} {'vp cutoff':>10s} {'vp marg':>8s} "
          f"{'nc cutoff':>10s} {'nc marg':>8s} {'observed':>9s} "
          f"{'chance':>8s} {'ratio':>7s}")
    print("  " + "-" * 74)
    for pct in PCTS:
        cv = d["n_vp"].quantile(pct / 100)
        cn = d["n_nc"].quantile(pct / 100)
        lo_vp = d["n_vp"] <= cv
        lo_nc = d["n_nc"] <= cn
        m_vp, m_nc = float(lo_vp.mean()), float(lo_nc.mean())
        inter = int((lo_vp & lo_nc).sum())
        obs = inter / max(int(lo_vp.sum()), 1)
        # under independence, P(nurse-flagged | monitor-flagged) = nurse marginal
        chance = m_nc
        jac = inter / max(int((lo_vp | lo_nc).sum()), 1)
        print(f"  {pct:>4d} {cv:10.0f} {m_vp:8.3f} {cn:10.0f} {m_nc:8.3f} "
              f"{obs:9.3f} {chance:8.3f} {obs / chance:7.2f}")
        rows.append({"variable": label, "percentile": pct, "n_stays": n,
                     "vp_cutoff": float(cv), "vp_marginal": m_vp,
                     "nc_cutoff": float(cn), "nc_marginal": m_nc,
                     "n_intersection": inter, "observed_agreement": obs,
                     "chance_agreement": chance,
                     "ratio_observed_to_chance": obs / chance,
                     "jaccard": jac,
                     "spearman": float(d["n_vp"].rank().corr(d["n_nc"].rank()))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr-nc", required=True, type=Path)
    ap.add_argument("--hr-vp", required=True, type=Path)
    ap.add_argument("--cross-dir", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./chance_comparator"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pat = pd.read_csv(args.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "unitdischargeoffset"])

    print("=" * 78)
    print("STREAM AGREEMENT AGAINST THE CORRECT CHANCE COMPARATOR")
    print("=" * 78)
    print("chance = the nurse stream's actual marginal, i.e. P(nurse-flagged) "
          "under independence")

    rows = []
    analyse(counts_from(pd.read_parquet(args.hr_vp), "patientunitstayid"),
            counts_from(pd.read_parquet(args.hr_nc), "patientunitstayid"),
            "heart_rate", pat, rows)

    cv_nc = pd.read_parquet(args.cross_dir / "eicu_nc.parquet")
    cv_vp = pd.read_parquet(args.cross_dir / "eicu_vp.parquet")
    for var in sorted(cv_nc["variable"].unique()):
        analyse(
            counts_from(cv_vp[cv_vp["variable"] == var], "patientunitstayid"),
            counts_from(cv_nc[cv_nc["variable"] == var], "patientunitstayid"),
            var, pat, rows)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / "chance_comparator.csv", index=False)

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    print("ratio > 1 means the two streams agree more than independence would "
          "give; ratio near 1 means the designations are effectively unrelated; "
          "ratio < 1 means they agree LESS than chance.")
    print("The manuscript currently compares against a fixed 0.100. Replace "
          "with the nurse marginal reported here, and with the ratio.")
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
