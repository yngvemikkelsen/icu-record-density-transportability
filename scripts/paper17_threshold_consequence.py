#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Paper 17: what a global low-density threshold classifies.

RATIONALE
---------
The decomposition shows hospital explaining 0.473 of the variance in record
count and 0.761 in median charting interval, against 0.003-0.032 for unit within
hospital.  That is a variance statement.  This script turns it into a
consequence for the proposed use of record density as a process measure.

If a hospital's whole documentation distribution is shifted downward, then a
threshold defined on the pooled distribution will select that hospital's
patients regardless of anything about the patients.  The question is how much
of the "low density" designation is carried by site rather than by stay:

    global bottom decile      threshold on the pooled eICU distribution
    hospital-specific decile  threshold within each hospital

Both flag 10% of stays overall.  If they agree, the site effect does not reach
the classification.  If they disagree, a global threshold classifies the
hospital as much as the patient.

Reported: concordance and Jaccard between the two designations; the share of
stays whose classification changes; the per-hospital share flagged by the global
threshold and its spread; and how many hospitals contribute essentially none or
essentially all of their stays to the global bottom decile.

The same is computed within MIMIC across care units, where the decomposition
predicts near-agreement, as an internal contrast.

Usage:
  python paper17_threshold_consequence.py \
      --eicu-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --mimic-cache ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --out-dir ~/bcst/threshold_consequence
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_MIN = 24 * 60
MIN_RECORDS = 3
MIN_GROUP = 500
PCTS = [5, 10, 20]


def counts(path, key):
    return (pd.read_parquet(path).groupby(key).size()
              .rename("n_records").reset_index())


def agreement(df, group_col, value_col, pct, label):
    """Global vs within-group threshold at the same nominal percentile."""
    d = df.copy()
    g_cut = d[value_col].quantile(pct / 100)
    d["low_global"] = (d[value_col] <= g_cut).astype(int)
    d["low_local"] = (d.groupby(group_col)[value_col]
                        .transform(lambda s: s <= s.quantile(pct / 100))
                        .astype(int))
    both = int((d["low_global"] & d["low_local"]).sum())
    union = int((d["low_global"] | d["low_local"]).sum())
    jac = both / union if union else np.nan
    changed = float((d["low_global"] != d["low_local"]).mean())
    conc = both / max(int(d["low_global"].sum()), 1)

    per_g = d.groupby(group_col)["low_global"].mean()
    out = {
        "level": label, "percentile": pct, "n_stays": len(d),
        "global_cutoff": float(g_cut),
        "n_flagged_global": int(d["low_global"].sum()),
        "n_flagged_local": int(d["low_local"].sum()),
        "concordant": conc, "jaccard": jac, "share_reclassified": changed,
        "group_share_p10": float(per_g.quantile(.10)),
        "group_share_median": float(per_g.median()),
        "group_share_p90": float(per_g.quantile(.90)),
        "group_share_min": float(per_g.min()),
        "group_share_max": float(per_g.max()),
        "n_groups": int(per_g.size),
        "groups_under_1pct": int((per_g < 0.01).sum()),
        "groups_over_50pct": int((per_g > 0.50).sum()),
    }
    return out, per_g, d


def report(rows, per_g_10, label, out_dir, stem):
    print(f"\n{'pct':>4s} {'cutoff':>8s} {'concordant':>11s} {'Jaccard':>8s} "
          f"{'reclassified':>13s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['percentile']:>4d} {r['global_cutoff']:8.0f} "
              f"{r['concordant']:11.3f} {r['jaccard']:8.3f} "
              f"{r['share_reclassified']:13.3f}")
    r10 = [r for r in rows if r["percentile"] == 10][0]
    print(f"\n  share of each {label} flagged by the GLOBAL bottom decile "
          f"({r10['n_groups']} groups)")
    print(f"    min {r10['group_share_min']:.3f}   p10 "
          f"{r10['group_share_p10']:.3f}   median "
          f"{r10['group_share_median']:.3f}   p90 "
          f"{r10['group_share_p90']:.3f}   max {r10['group_share_max']:.3f}")
    print(f"    groups contributing <1% of their stays: "
          f"{r10['groups_under_1pct']}")
    print(f"    groups contributing >50% of their stays: "
          f"{r10['groups_over_50pct']}")
    pd.DataFrame(rows).to_csv(out_dir / f"{stem}_agreement.csv", index=False)
    per_g_10.rename("share_flagged_global").to_csv(
        out_dir / f"{stem}_group_shares.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eicu-cache", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--mimic-cache", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("./threshold_consequence"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- eICU: hospitals ----------------
    e = counts(args.eicu_cache, "patientunitstayid")
    pat = pd.read_csv(args.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    e = e.merge(pat, on="patientunitstayid", how="inner")
    e = e[(e["unitdischargeoffset"] >= WINDOW_MIN)
          & (e["n_records"] >= MIN_RECORDS)]
    keep = e["hospitalid"].value_counts()
    e = e[e["hospitalid"].isin(keep[keep >= MIN_GROUP].index)]
    print("=" * 78)
    print(f"eICU: {len(e):,} stays, {e['hospitalid'].nunique()} hospitals "
          f"(>= {MIN_GROUP} stays)")
    print("GLOBAL vs HOSPITAL-SPECIFIC LOW-DENSITY THRESHOLD")
    print("=" * 78)
    rows, per_h = [], None
    for pct in PCTS:
        r, pg, d = agreement(e, "hospitalid", "n_records", pct, "hospital")
        rows.append(r)
        if pct == 10:
            per_h, d10 = pg, d
    report(rows, per_h, "hospital", args.out_dir, "eicu_hospital")

    # which hospitals dominate the global bottom decile
    flagged = d10[d10["low_global"] == 1]
    top = (flagged.groupby("hospitalid").size()
                  .sort_values(ascending=False).head(10)
                  .rename("stays_in_global_decile").to_frame())
    top["share_of_global_decile"] = top["stays_in_global_decile"] / len(flagged)
    top["share_of_own_stays"] = top.index.map(per_h)
    print("\n  hospitals contributing most to the global bottom decile")
    print(top.round(4).to_string())
    top.to_csv(args.out_dir / "eicu_top_contributors.csv")
    cum = (flagged.groupby("hospitalid").size().sort_values(ascending=False)
                  .cumsum() / len(flagged))
    n_half = int((cum < 0.5).sum()) + 1
    print(f"\n  {n_half} of {e['hospitalid'].nunique()} hospitals account for "
          f"half of all stays in the global bottom decile")

    # ---------------- MIMIC: care units, internal contrast ----------------
    m = counts(args.mimic_cache, "stay_id")
    ps = pd.read_csv(args.mimic_per_stay, usecols=["stay_id", "careunit", "los"])
    m = m.merge(ps, on="stay_id", how="inner")
    m = m[(m["los"] >= 1.0) & (m["n_records"] >= MIN_RECORDS)]
    keep = m["careunit"].value_counts()
    m = m[m["careunit"].isin(keep[keep >= MIN_GROUP].index)]
    print("\n" + "=" * 78)
    print(f"MIMIC (internal contrast): {len(m):,} stays, "
          f"{m['careunit'].nunique()} care units")
    print("GLOBAL vs UNIT-SPECIFIC LOW-DENSITY THRESHOLD")
    print("=" * 78)
    rows, per_u = [], None
    for pct in PCTS:
        r, pg, _ = agreement(m, "careunit", "n_records", pct, "care unit")
        rows.append(r)
        if pct == 10:
            per_u = pg
    report(rows, per_u, "care unit", args.out_dir, "mimic_unit")

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    print("Both thresholds flag the same number of stays. Where they disagree, "
          "the designation is carried by the group rather than the stay.")
    print("The contrast between the two panels is the point: the same "
          "procedure applied within one institution and across many.")
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
