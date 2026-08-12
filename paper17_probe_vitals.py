#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy"]
# ///
"""
Paper 17: probe for the cross-vital-sign replication.

WHY A PROBE FIRST
-----------------
The heart-rate analysis rests on two facts that had to be discovered rather than
assumed: per_stay.n_hr_24h reconstructed as 220045 + 220046 + 220047, two of
which are shift-anchored alarm limits; and eICU nurseCharting carries heart rate
under exactly one value-name string.  Neither was predictable from documentation.

Respiratory rate and SpO2 are the candidates for replication.  Before any
extraction, this reports:

  MIMIC   every chartevents item whose label matches the variable, with its
          in-window volume per stay and its charting-hour concentration, so the
          alarm-limit contamination can be assessed for each variable
          separately.  SpO2 is known to have alarm items (223769, 223770) with
          the same 00/08/20 profile as the heart-rate ones; whether respiratory
          rate does is unknown.

  eICU    every nurseCharting value-name and value-label string matching the
          variable, with record counts, so the extraction can use verified
          strings rather than a regex guess.  Also the vitalPeriodic columns
          available for each variable, since the nurse-versus-monitor contrast
          requires both.

Nothing is extracted or computed here beyond volumes and charting hours.

Usage:
  python paper17_probe_vitals.py \
      --mimic-root ~/physionet.org/files/mimiciv/3.1 \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/vitals_probe

Runtime: one chartevents pass and one nurseCharting pass, roughly 15-25 min.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_H = 24.0
CHUNKSIZE = 5_000_000

# Deliberately over-broad: the alarm-limit items are labelled "Resp Alarm -
# High" rather than "Respiratory Rate Alarm", so a pattern anchored on the
# variable name alone would miss the contamination this probe exists to detect.
VARIABLES = {
    "respiratory_rate": r"respiratory|resp rate|resp alarm|\brr\b",
    "spo2": r"o2 saturation|spo2|oxygen saturation|pulse ?ox|sao2",
}


def find_table(root: Path, *cands: str) -> Path:
    for c in cands:
        p = root / c
        if p.exists():
            return p
        alt = p.with_suffix("")
        if alt.exists():
            return alt
    raise FileNotFoundError(f"none of {cands} under {root}")


def probe_mimic(root: Path, out_dir: Path) -> None:
    print("=" * 78)
    print("MIMIC-IV chartevents")
    print("=" * 78)
    di = pd.read_csv(find_table(root, "icu/d_items.csv.gz"))
    ids, meta = [], []
    for var, pat in VARIABLES.items():
        m = di[di["label"].str.contains(pat, case=False, na=False, regex=True)]
        if "linksto" in m.columns:
            m = m[m["linksto"] == "chartevents"]
        cols = [c for c in ("itemid", "label", "abbreviation", "unitname",
                            "param_type") if c in m.columns]
        print(f"\n[{var}] {len(m)} candidate items")
        print(m[cols].to_string(index=False))
        ids += m["itemid"].tolist()
        meta += [{"variable": var, **r} for r in m[cols].to_dict("records")]
    pd.DataFrame(meta).to_csv(out_dir / "mimic_candidate_items.csv", index=False)

    stays = pd.read_csv(find_table(root, "icu/icustays.csv.gz"),
                        usecols=["stay_id", "intime"], parse_dates=["intime"])
    print(f"\nscanning chartevents for {len(ids)} itemids ...")
    kept, scanned = [], 0
    for i, chunk in enumerate(pd.read_csv(
            find_table(root, "icu/chartevents.csv.gz"),
            usecols=["stay_id", "itemid", "charttime", "valuenum"],
            parse_dates=["charttime"], chunksize=CHUNKSIZE)):
        scanned += len(chunk)
        chunk = chunk[chunk["itemid"].isin(ids)]
        if chunk.empty:
            continue
        chunk = chunk.merge(stays, on="stay_id", how="inner")
        h = (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600.0
        chunk = chunk[(h >= 0) & (h < WINDOW_H)].copy()
        chunk["charthour"] = chunk["charttime"].dt.hour
        if not chunk.empty:
            kept.append(chunk[["stay_id", "itemid", "valuenum", "charthour"]])
        if (i + 1) % 10 == 0:
            print(f"  ...{scanned:,} rows scanned")
    ev = pd.concat(kept, ignore_index=True)
    print(f"in-window records: {len(ev):,}")

    vol = (ev.groupby("itemid")
             .agg(records=("valuenum", "size"), stays=("stay_id", "nunique"),
                  nonnull=("valuenum", "count"))
             .reset_index()
             .merge(di[["itemid", "label"]], on="itemid", how="left"))
    vol["records_per_stay"] = (vol["records"] / vol["stays"]).round(2)

    hours = ev.groupby(["itemid", "charthour"]).size().rename("n").reset_index()
    hours["share"] = hours.groupby("itemid")["n"].transform(lambda s: s / s.sum())
    conc = (hours.sort_values("share", ascending=False).groupby("itemid").head(3)
                 .groupby("itemid")
                 .agg(top3_share=("share", "sum"),
                      top_hours=("charthour", lambda s: sorted(s.tolist())))
                 .reset_index())
    vol = vol.merge(conc, on="itemid", how="left").sort_values(
        "records", ascending=False)
    print("\nin-window volume and charting-hour concentration "
          "(flat = 0.125):")
    print(vol.to_string(index=False))
    vol.to_csv(out_dir / "mimic_item_volume.csv", index=False)
    print("\n  An item with a top-3 share near 0.53 at hours 00/08/20 is "
          "shift-anchored, as the heart-rate alarm limits were. Such items are "
          "documentation events, not observations, and must be excluded.")


def probe_eicu(root: Path, out_dir: Path) -> None:
    print("\n" + "=" * 78)
    print("eICU-CRD nurseCharting and vitalPeriodic")
    print("=" * 78)

    print("\nvitalPeriodic columns available:")
    head = pd.read_csv(find_table(root, "vitalPeriodic.csv.gz",
                                  "vitalperiodic.csv.gz"), nrows=5)
    print("  " + ", ".join(head.columns))

    print("\nscanning nurseCharting for value-name strings ...")
    names, labels, scanned = [], [], 0
    for i, chunk in enumerate(pd.read_csv(
            find_table(root, "nurseCharting.csv.gz", "nursecharting.csv.gz"),
            usecols=["nursingchartcelltypevallabel",
                     "nursingchartcelltypevalname"],
            chunksize=CHUNKSIZE, low_memory=False)):
        scanned += len(chunk)
        names.append(chunk["nursingchartcelltypevalname"].value_counts())
        labels.append(chunk["nursingchartcelltypevallabel"].value_counts())
        if (i + 1) % 10 == 0:
            print(f"  ...{scanned:,} rows scanned")
    allnames = pd.concat(names).groupby(level=0).sum().sort_values(
        ascending=False)
    alllabels = pd.concat(labels).groupby(level=0).sum().sort_values(
        ascending=False)
    allnames.rename("records").to_csv(out_dir / "eicu_nc_valname_counts.csv")
    alllabels.rename("records").to_csv(out_dir / "eicu_nc_vallabel_counts.csv")
    print(f"\ndistinct valname strings: {len(allnames):,}; "
          f"distinct vallabel strings: {len(alllabels):,} "
          f"(full counts written to CSV)")

    rows = []
    for var, pat in VARIABLES.items():
        for kind, series in (("valname", allnames), ("vallabel", alllabels)):
            hit = series[series.index.astype(str).str.contains(
                pat, case=False, regex=True, na=False)]
            print(f"\n[{var}] {kind}: {len(hit)} matching strings")
            if len(hit):
                print(hit.head(15).to_string())
            rows += [{"variable": var, "field": kind, "string": k,
                      "records": int(v)} for k, v in hit.items()]
    pd.DataFrame(rows).to_csv(out_dir / "eicu_matched_strings.csv", index=False)
    print("\n  Use the strings above verbatim in the extraction. A single "
          "dominant string, as for heart rate, is the easy case; several "
          "comparable strings mean the variable is fragmented and the union "
          "must be justified.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-root", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./vitals_probe"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    probe_mimic(args.mimic_root, args.out_dir)
    probe_eicu(args.eicu_root, args.out_dir)
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
