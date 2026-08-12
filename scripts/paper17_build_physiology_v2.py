#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy"]
# ///
"""
Build harmonised first-24h physiology summaries for Paper 17.

WHY THIS EXISTS
---------------
The v1 residualisation regressed log(n_hr_24h + 1) on hr_min_24h, hr_max_24h,
map_min_24h and map_max_24h.  Those are ORDER STATISTICS of the same records
being counted: the sample minimum falls and the sample maximum rises with n by
construction (extreme-value effect).  The MIMIC residualisation R^2 (0.133) is
therefore inflated by a mechanical relationship, while the eICU model
(log_apache, a curated composite) is not exposed to it.  The reported fourfold
acuity-coupling contrast is partly an artifact of that asymmetry.

FIX
---
Summarise each cohort's own vital stream with CONSISTENT estimators that are not
mechanically driven by record count:
    median  -> location
    IQR     -> dispersion
Both are computed identically in MIMIC chartevents and eICU vitalPeriodic, so
the residualisation model is the same in both cohorts (same variables, same
functional form) and neither side carries the extreme-value confound.

GCS is deliberately excluded: it is not present in eICU vitalPeriodic, so
including it would reintroduce the non-harmonisation it is meant to remove.

HR carries the primary model.  It is near-complete in both streams (96% of eICU
patients per the eICU-CRD descriptor).  MAP is emitted separately by modality
(invasive vs cuff in MIMIC; invasive only in eICU, since vitalPeriodic holds
arterial pressure and the cuff equivalent sits in vitalAperiodic) and is used
only in a sensitivity model, because arterial-line presence is acuity-related
and complete-case MAP models therefore select on severity.

VERIFY BEFORE TRUSTING
----------------------
The MIMIC itemids below are asserted, not verified in this script.  Run:
    python paper17_build_physiology_v2.py --check-itemids --mimic-root ...
to print the matching rows of d_items.csv.gz and confirm labels/units before
using the output.

Usage:
  python paper17_build_physiology_v2.py \
      --mimic-root /Users/yngve/physionet.org/files/mimiciv/3.1 \
      --eicu-root  /Users/yngve/physionet.org/files/eicu-crd/2.0 \
      --out-dir    ~/bcst/physiology_v2

Runtime: 30-60 min for the MIMIC chartevents pass, 10-20 min for eICU
vitalPeriodic.  Both are cached; rerun with --force to rebuild.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --- MIMIC itemids (VERIFY with --check-itemids before relying on output) ----
# MAP is split by modality on purpose. eICU vitalPeriodic.systemicmean is
# INVASIVE arterial pressure only (non-invasive lives in vitalAperiodic), so
# pooling MIMIC's invasive and non-invasive items would put a different
# construct on each side of the cross-cohort comparison. Arterial-line presence
# is also acuity-related, so complete-case models on MAP select on severity.
HR_IDS = [220045]                       # Heart Rate
MAP_INV_IDS = [220052, 225312]          # ABP mean, ART BP Mean  (arterial line)
MAP_NIBP_IDS = [220181]                 # NBP mean              (cuff)
ALL_IDS = HR_IDS + MAP_INV_IDS + MAP_NIBP_IDS

HR_RANGE = (20.0, 250.0)
MAP_RANGE = (20.0, 200.0)

WINDOW_H = 24.0
CHUNKSIZE = 5_000_000


def find_table(root: Path, *candidates: str) -> Path:
    for c in candidates:
        p = root / c
        if p.exists():
            return p
        alt = p.with_suffix("")          # allow uncompressed
        if alt.exists():
            return alt
    raise FileNotFoundError(
        f"none of {candidates} found under {root} — check --mimic-root/--eicu-root"
    )


def iqr(s: pd.Series) -> float:
    return float(s.quantile(0.75) - s.quantile(0.25))


def summarise(df: pd.DataFrame, key: str, value: str, prefix: str) -> pd.DataFrame:
    g = df.groupby(key)[value]
    out = pd.DataFrame({
        f"{prefix}_median_24h": g.median(),
        f"{prefix}_iqr_24h": g.agg(iqr),
        f"{prefix}_n_24h": g.size(),
    })
    return out.reset_index()


# ----------------------------------------------------------------------------
# MIMIC
# ----------------------------------------------------------------------------

def check_itemids(mimic_root: Path) -> None:
    p = find_table(mimic_root, "icu/d_items.csv.gz")
    d = pd.read_csv(p, usecols=["itemid", "label", "abbreviation", "unitname"])
    print(d[d["itemid"].isin(ALL_IDS)].to_string(index=False))
    print("\nConfirm these are the intended HR and mean-arterial-pressure items "
          "before using the built physiology file.")


def build_mimic(mimic_root: Path, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        print(f"MIMIC physiology cached at {out_path} (use --force to rebuild)")
        return

    stays = pd.read_csv(
        find_table(mimic_root, "icu/icustays.csv.gz"),
        usecols=["stay_id", "intime"],
        parse_dates=["intime"],
    )
    print(f"MIMIC: {len(stays):,} ICU stays")

    ce = find_table(mimic_root, "icu/chartevents.csv.gz")
    kept = []
    n_rows = 0
    for i, chunk in enumerate(pd.read_csv(
        ce,
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
        chunksize=CHUNKSIZE,
    )):
        n_rows += len(chunk)
        chunk = chunk[chunk["itemid"].isin(ALL_IDS)]
        chunk = chunk.dropna(subset=["valuenum", "stay_id", "charttime"])
        if chunk.empty:
            continue
        chunk = chunk.merge(stays, on="stay_id", how="inner")
        h = (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600.0
        chunk = chunk[(h >= 0) & (h < WINDOW_H)]
        if not chunk.empty:
            kept.append(chunk[["stay_id", "itemid", "valuenum"]])
        if (i + 1) % 10 == 0:
            print(f"  ...{n_rows:,} chartevents rows scanned")

    ev = pd.concat(kept, ignore_index=True)
    print(f"MIMIC: {len(ev):,} in-window HR/MAP records")

    hr = ev[ev["itemid"].isin(HR_IDS)]
    hr = hr[hr["valuenum"].between(*HR_RANGE)]
    inv = ev[ev["itemid"].isin(MAP_INV_IDS)]
    inv = inv[inv["valuenum"].between(*MAP_RANGE)]
    nib = ev[ev["itemid"].isin(MAP_NIBP_IDS)]
    nib = nib[nib["valuenum"].between(*MAP_RANGE)]

    out = summarise(hr, "stay_id", "valuenum", "hr")
    for frame, prefix in ((inv, "map_inv"), (nib, "map_nibp")):
        out = out.merge(summarise(frame, "stay_id", "valuenum", prefix),
                        on="stay_id", how="outer")
    out = out.rename(columns={"hr_n_24h": "n_hr_24h_rebuilt"})
    n_stay = len(stays)
    print(f"  availability: HR {out['hr_median_24h'].notna().sum() / n_stay:.3f}, "
          f"invasive MAP {out['map_inv_median_24h'].notna().sum() / n_stay:.3f}, "
          f"NIBP {out['map_nibp_median_24h'].notna().sum() / n_stay:.3f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"MIMIC physiology -> {out_path}  ({len(out):,} stays)")
    print("  NOTE: compare n_hr_24h_rebuilt against the n_hr_24h in your "
          "per-stay file. A mismatch means the two were built with different "
          "itemid sets or window definitions, and must be reconciled before "
          "any residualisation is interpreted.")


# ----------------------------------------------------------------------------
# eICU
# ----------------------------------------------------------------------------

def build_eicu(eicu_root: Path, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        print(f"eICU physiology cached at {out_path} (use --force to rebuild)")
        return

    vp = find_table(eicu_root, "vitalPeriodic.csv.gz", "vitalperiodic.csv.gz")
    kept = []
    n_rows = 0
    for i, chunk in enumerate(pd.read_csv(
        vp,
        usecols=["patientunitstayid", "observationoffset",
                 "heartrate", "systemicmean"],
        chunksize=CHUNKSIZE,
    )):
        n_rows += len(chunk)
        chunk = chunk[(chunk["observationoffset"] >= 0)
                      & (chunk["observationoffset"] < WINDOW_H * 60)]
        if not chunk.empty:
            kept.append(chunk[["patientunitstayid", "heartrate", "systemicmean"]])
        if (i + 1) % 10 == 0:
            print(f"  ...{n_rows:,} vitalPeriodic rows scanned")

    ev = pd.concat(kept, ignore_index=True)
    print(f"eICU: {len(ev):,} in-window vitalPeriodic records")

    hr = ev.dropna(subset=["heartrate"])
    hr = hr[hr["heartrate"].between(*HR_RANGE)]
    # systemicmean is INVASIVE arterial pressure; the cuff equivalent is in
    # vitalAperiodic and is deliberately not used, to keep the MAP construct
    # identical to the MIMIC map_inv_* columns.
    mp = ev.dropna(subset=["systemicmean"])
    mp = mp[mp["systemicmean"].between(*MAP_RANGE)]

    out = (summarise(hr, "patientunitstayid", "heartrate", "hr")
           .merge(summarise(mp, "patientunitstayid", "systemicmean", "map_inv"),
                  on="patientunitstayid", how="outer"))
    out = out.rename(columns={"hr_n_24h": "n_hr_24h_rebuilt"})
    print(f"  availability: HR {out['hr_median_24h'].notna().mean():.3f}, "
          f"invasive MAP {out['map_inv_median_24h'].notna().mean():.3f} "
          f"(of stays with any vitalPeriodic record)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"eICU physiology -> {out_path}  ({len(out):,} stays)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-root", type=Path)
    ap.add_argument("--eicu-root", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./physiology_v2"))
    ap.add_argument("--check-itemids", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.check_itemids:
        if args.mimic_root is None:
            raise SystemExit("--check-itemids requires --mimic-root")
        check_itemids(args.mimic_root)
        return

    if args.mimic_root is None or args.eicu_root is None:
        raise SystemExit("--mimic-root and --eicu-root are both required")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    build_mimic(args.mimic_root, args.out_dir / "mimic_physiology_v2.csv",
                args.force)
    build_eicu(args.eicu_root, args.out_dir / "eicu_physiology_v2.csv",
               args.force)


if __name__ == "__main__":
    main()
