#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy"]
# ///
"""
Diagnose the composition of per_stay.n_hr_24h.

CONTEXT
-------
Two independent builds of "heart-rate observations in the first 24 h" agree with
each other (physiology_24h.hr_n and mimic_physiology_v2.n_hr_24h_rebuilt: 99.3%
exact, r = 1.0000).  per_stay.n_hr_24h is systematically ~6 records higher in
every stay.  The manuscript exposure, Table 1 and the percentile cutoffs are all
built on n_hr_24h, so its composition has to be established.

TWO QUESTIONS
-------------
1. WHICH ITEMIDS.  Search d_items for every heart-rate-related item, count each
   per stay in the first 24 h, then test which subset reproduces n_hr_24h.
2. WHEN ARE THEY CHARTED.  If the extra items are alarm limits or similar
   shift-anchored documentation, they cluster at handover hours.  The cyclic
   temporal finding (MIMIC peak 06:00-08:00) would then be partly an artifact of
   which records the exposure counts.  The script reports the charting-hour
   distribution of each item so this can be checked directly.

This script diagnoses only.  It changes nothing and writes no cohort files.

Usage:
  python paper17_diagnose_exposure.py \
      --mimic-root     ~/physionet.org/files/mimiciv/3.1 \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --out-dir        ~/bcst/exposure_diagnosis

Runtime: one chartevents pass, comparable to the build stage.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_H = 24.0
CHUNKSIZE = 5_000_000
LABEL_PATTERN = r"heart rate|^hr\b|pulse"


def find_table(root: Path, *candidates: str) -> Path:
    for c in candidates:
        p = root / c
        if p.exists():
            return p
        alt = p.with_suffix("")
        if alt.exists():
            return alt
    raise FileNotFoundError(f"none of {candidates} under {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-root", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./exposure_diagnosis"))
    ap.add_argument("--max-subset", type=int, default=4,
                    help="largest itemid combination to test")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- candidate items ----------------------------------------------------
    di = pd.read_csv(find_table(args.mimic_root, "icu/d_items.csv.gz"),
                     usecols=["itemid", "label", "abbreviation", "unitname",
                              "param_type"])
    cand = di[di["label"].str.contains(LABEL_PATTERN, case=False, na=False,
                                       regex=True)]
    print("Candidate heart-rate-family items in d_items:")
    print(cand.to_string(index=False))
    ids = cand["itemid"].tolist()
    if 220045 not in ids:
        ids.append(220045)
    print(f"\nscanning chartevents for {len(ids)} itemids ...")

    # ---- one chartevents pass ----------------------------------------------
    stays = pd.read_csv(find_table(args.mimic_root, "icu/icustays.csv.gz"),
                        usecols=["stay_id", "intime"], parse_dates=["intime"])
    kept, scanned = [], 0
    for i, chunk in enumerate(pd.read_csv(
            find_table(args.mimic_root, "icu/chartevents.csv.gz"),
            usecols=["stay_id", "itemid", "charttime", "valuenum"],
            parse_dates=["charttime"], chunksize=CHUNKSIZE)):
        scanned += len(chunk)
        chunk = chunk[chunk["itemid"].isin(ids)]
        if chunk.empty:
            continue
        chunk = chunk.merge(stays, on="stay_id", how="inner")
        h = (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600.0
        chunk = chunk[(h >= 0) & (h < WINDOW_H)].copy()
        if not chunk.empty:
            chunk["charthour"] = chunk["charttime"].dt.hour
            kept.append(chunk[["stay_id", "itemid", "valuenum", "charthour"]])
        if (i + 1) % 10 == 0:
            print(f"  ...{scanned:,} rows scanned")

    ev = pd.concat(kept, ignore_index=True)
    print(f"in-window records: {len(ev):,}")

    # ---- per-item volume and charting hour ---------------------------------
    per_item = (ev.groupby("itemid")
                  .agg(records=("valuenum", "size"),
                       stays=("stay_id", "nunique"),
                       nonnull_valuenum=("valuenum", "count"))
                  .reset_index()
                  .merge(di[["itemid", "label"]], on="itemid", how="left")
                  .sort_values("records", ascending=False))
    per_item["records_per_stay"] = (per_item["records"]
                                    / per_item["stays"]).round(2)
    print("\nPer-item volume in the first 24 h:")
    print(per_item.to_string(index=False))
    per_item.to_csv(args.out_dir / "per_item_volume.csv", index=False)

    hours = (ev.groupby(["itemid", "charthour"]).size()
               .rename("records").reset_index())
    hours["share"] = hours.groupby("itemid")["records"].transform(
        lambda s: s / s.sum())
    hours.to_csv(args.out_dir / "charting_hour_by_item.csv", index=False)

    print("\nCharting-hour concentration (share of an item's records in its "
          "three busiest hours; a flat item sits near 0.125):")
    conc = (hours.sort_values("share", ascending=False)
                 .groupby("itemid").head(3)
                 .groupby("itemid")
                 .agg(top3_share=("share", "sum"),
                      top_hours=("charthour", lambda s: sorted(s.tolist())))
                 .reset_index()
                 .merge(di[["itemid", "label"]], on="itemid", how="left")
                 .sort_values("top3_share", ascending=False))
    print(conc.to_string(index=False))
    conc.to_csv(args.out_dir / "charting_hour_concentration.csv", index=False)

    # ---- which subset reproduces n_hr_24h ----------------------------------
    counts = (ev.groupby(["stay_id", "itemid"]).size()
                .unstack(fill_value=0))
    ps = pd.read_csv(args.mimic_per_stay, usecols=["stay_id", "n_hr_24h"])
    m = ps.merge(counts, on="stay_id", how="inner")
    present = [c for c in counts.columns if c in m.columns]
    print(f"\nreconstructing n_hr_24h over {len(m):,} stays "
          f"from {len(present)} items ...")

    rows = []
    for k in range(1, min(args.max_subset, len(present)) + 1):
        for combo in itertools.combinations(present, k):
            tot = m[list(combo)].sum(axis=1)
            rows.append({
                "items": "+".join(str(c) for c in combo),
                "k": k,
                "exact_agreement": float((tot == m["n_hr_24h"]).mean()),
                "mean_diff": float((m["n_hr_24h"] - tot).mean()),
                "mean_abs_diff": float((m["n_hr_24h"] - tot).abs().mean()),
            })
    res = pd.DataFrame(rows).sort_values(
        ["exact_agreement", "mean_abs_diff"], ascending=[False, True])
    res.to_csv(args.out_dir / "subset_reconstruction.csv", index=False)
    print("\nBest reconstructions of n_hr_24h:")
    print(res.head(12).to_string(index=False))

    best = res.iloc[0]
    print(f"\nBEST: {best['items']}  exact {best['exact_agreement']:.4f}  "
          f"mean residual {best['mean_diff']:+.2f}")
    if best["exact_agreement"] < 0.90:
        print("No itemid combination reproduces n_hr_24h. The difference is "
              "then a window or filter difference, not item composition — read "
              "the extraction script that produced per_stay.")
    elif "+" in str(best["items"]):
        print("n_hr_24h pools multiple items. Check the charting-hour "
              "concentration above for the non-220045 members: if they are "
              "shift-anchored, the cyclic finding in the manuscript is partly "
              "a property of the exposure definition.")

    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
