#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy"]
# ///
"""
Harmonised severity components for Paper 17, both cohorts.

WHY
---
The v2 run showed the cross-cohort mortality reversal exists ONLY under
cohort-specific adjustment (MIMIC OASIS-style components vs eICU APACHE-IV):

    full adjustment     MIMIC bottom-10% OR 0.678   eICU 1.375   -> reversal
    minimal adjustment  MIMIC bottom-10% OR 0.349   eICU 0.658   -> agreement

So "the association does not transport" is currently confounded with "the two
cohorts were adjusted with different severity instruments".  That has to be
removed before any transportability claim can be made.  This script builds the
SAME severity component set in both cohorts.

WHAT IS NOT DONE
----------------
The OASIS score is NOT computed.  Reproducing Johnson et al. (2013) point
cutpoints from memory would risk inventing them, and a fixed-weight score is a
worse adjuster than the same components entered as free covariates.  The output
is components, not a score.

THE "WORST VALUE" RULE
----------------------
eICU apacheApsVar stores one worst-in-24h value per variable.  MIMIC yields a
min and a max, so it must be reduced to one.

A first attempt used a symmetric "furthest from the normal midpoint" rule.  It
failed empirically: MIMIC hr_worst came back with median 71 against eICU's 104,
because bradycardia often deviates further in raw units while APACHE's selection
weights tachycardia far more heavily.  The two columns were not the same
variable.

The rule is therefore DIRECTIONAL, matching which tail APACHE actually scores:

    HR    -> maximum          (tachycardia)
    MAP   -> minimum          (hypotension)
    RR    -> maximum          (tachypnoea)
    GCS   -> minimum          (depressed consciousness)
    Temp  -> furthest from 37.2 C, either direction (both tails scored)

Raw *_min and *_max are written out alongside, so revising this rule does not
require another chartevents pass.  Compare the MIMIC and eICU medians per
component after building; a large gap means the selection rules still disagree.

Residual difference: eICU's selection is APACHE's exact scoring rule, MIMIC's is
the directional approximation above.  Much smaller than OASIS-vs-APACHE, but not
zero, and it belongs in the limitations.

GCS is the minimum of the summed eye+verbal+motor total in MIMIC, and
eyes+motor+verbal from apacheApsVar in eICU.

Usage:
  python paper17_build_severity_v3.py --check-itemids --mimic-root ...
  python paper17_build_severity_v3.py \
      --mimic-root ~/physionet.org/files/mimiciv/3.1 \
      --eicu-root  ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir    ~/bcst/severity_v3

Runtime: one chartevents pass plus one outputevents pass (~10 min), then a few
small eICU table reads.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_H = 24.0
CHUNKSIZE = 5_000_000

HR_IDS = [220045]
MAP_IDS = [220052, 225312, 220181]          # invasive + cuff; severity, not exposure
RR_IDS = [220210, 224690]                   # Respiratory Rate, Respiratory Rate (Total)
TEMP_F_IDS = [223761]                       # Temperature Fahrenheit
TEMP_C_IDS = [223762]                       # Temperature Celsius
GCS_IDS = [220739, 223900, 223901]          # eye, verbal, motor
VITAL_IDS = HR_IDS + MAP_IDS + RR_IDS + TEMP_F_IDS + TEMP_C_IDS + GCS_IDS

# Matches mimic-code urine_output: GU Irrigant Volume In (227488) enters as a
# NEGATIVE volume, because the instilled irrigant reappears in GU Irrigant/Urine
# Volume Out (227489). Omitting it credits irrigation as urine.
URINE_IDS = [226559, 226560, 226561, 226584, 226563, 226564, 226565,
             226567, 226557, 226558, 227488, 227489]
URINE_NEGATE_IDS = [227488]

RANGES = {"hr": (20.0, 250.0), "map": (20.0, 200.0),
          "rr": (2.0, 80.0), "temp": (25.0, 45.0)}
# Which tail APACHE effectively scores for each variable.
WORST_RULE = {"hr": "max", "map": "min", "rr": "max", "temp": "extreme"}
MIDPOINT = {"temp": 37.2}


def find_table(root: Path, *candidates: str) -> Path:
    for c in candidates:
        p = root / c
        if p.exists():
            return p
        alt = p.with_suffix("")
        if alt.exists():
            return alt
    raise FileNotFoundError(f"none of {candidates} under {root}")


def worst_of(lo: pd.Series, hi: pd.Series, rule: str,
             mid: float | None = None) -> pd.Series:
    """Reduce a min/max pair to the single value APACHE would have scored."""
    if rule == "max":
        return hi
    if rule == "min":
        return lo
    if rule == "extreme":
        if mid is None:
            raise ValueError("rule 'extreme' needs a midpoint")
        return hi.where((hi - mid).abs() >= (mid - lo).abs(), lo)
    raise ValueError(f"unknown worst rule: {rule}")


def check_itemids(mimic_root: Path) -> None:
    di = pd.read_csv(find_table(mimic_root, "icu/d_items.csv.gz"),
                     usecols=["itemid", "label", "abbreviation", "unitname",
                              "linksto"])
    print("VITALS / GCS")
    print(di[di["itemid"].isin(VITAL_IDS)].to_string(index=False))
    print("\nURINE OUTPUT")
    print(di[di["itemid"].isin(URINE_IDS)].to_string(index=False))
    missing = sorted(set(VITAL_IDS + URINE_IDS) - set(di["itemid"]))
    if missing:
        print(f"\nNOT FOUND in d_items: {missing}")
    print("\nConfirm every row above is the variable intended, and that the "
          "urine set is complete for this MIMIC version, before building.")


def build_mimic(mimic_root: Path, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        print(f"MIMIC severity cached at {out_path}")
        return

    stays = pd.read_csv(find_table(mimic_root, "icu/icustays.csv.gz"),
                        usecols=["stay_id", "intime"], parse_dates=["intime"])

    # ---- chartevents: vitals + GCS ----
    kept, scanned = [], 0
    for i, chunk in enumerate(pd.read_csv(
            find_table(mimic_root, "icu/chartevents.csv.gz"),
            usecols=["stay_id", "itemid", "charttime", "valuenum"],
            parse_dates=["charttime"], chunksize=CHUNKSIZE)):
        scanned += len(chunk)
        chunk = chunk[chunk["itemid"].isin(VITAL_IDS)].dropna(
            subset=["valuenum", "stay_id", "charttime"])
        if chunk.empty:
            continue
        chunk = chunk.merge(stays, on="stay_id", how="inner")
        h = (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600.0
        chunk = chunk[(h >= 0) & (h < WINDOW_H)]
        if not chunk.empty:
            kept.append(chunk[["stay_id", "itemid", "charttime", "valuenum"]])
        if (i + 1) % 10 == 0:
            print(f"  ...{scanned:,} chartevents rows scanned")
    ev = pd.concat(kept, ignore_index=True)
    print(f"MIMIC: {len(ev):,} in-window vital/GCS records")

    def minmax(ids, name, conv=None):
        s = ev[ev["itemid"].isin(ids)].copy()
        if conv is not None:
            s["valuenum"] = conv(s["valuenum"])
        lo, hi = RANGES[name]
        s = s[s["valuenum"].between(lo, hi)]
        g = s.groupby("stay_id")["valuenum"]
        return pd.DataFrame({f"{name}_min": g.min(), f"{name}_max": g.max()})

    parts = [minmax(HR_IDS, "hr"), minmax(MAP_IDS, "map"),
             minmax(RR_IDS, "rr")]
    tc = ev[ev["itemid"].isin(TEMP_C_IDS)].copy()
    tf = ev[ev["itemid"].isin(TEMP_F_IDS)].copy()
    tf["valuenum"] = (tf["valuenum"] - 32.0) * 5.0 / 9.0
    temp = pd.concat([tc, tf], ignore_index=True)
    temp = temp[temp["valuenum"].between(*RANGES["temp"])]
    g = temp.groupby("stay_id")["valuenum"]
    parts.append(pd.DataFrame({"temp_min": g.min(), "temp_max": g.max()}))

    # GCS: sum the three components per charttime, take the stay minimum
    gcs = ev[ev["itemid"].isin(GCS_IDS)]
    tot = (gcs.groupby(["stay_id", "charttime"])
              .agg(total=("valuenum", "sum"), n=("valuenum", "size"))
              .reset_index())
    tot = tot[tot["n"] == 3]                      # complete assessments only
    parts.append(tot.groupby("stay_id")["total"].min().rename("gcs_worst")
                    .to_frame())

    sev = pd.concat(parts, axis=1).reset_index()

    # ---- outputevents: urine volume in first 24h ----
    uo = pd.read_csv(find_table(mimic_root, "icu/outputevents.csv.gz"),
                     usecols=["stay_id", "itemid", "charttime", "value"],
                     parse_dates=["charttime"])
    uo = uo[uo["itemid"].isin(URINE_IDS)].dropna(subset=["value"])
    uo = uo.merge(stays, on="stay_id", how="inner")
    h = (uo["charttime"] - uo["intime"]).dt.total_seconds() / 3600.0
    uo = uo[(h >= 0) & (h < WINDOW_H)].copy()
    neg = uo["itemid"].isin(URINE_NEGATE_IDS) & (uo["value"] > 0)
    uo.loc[neg, "value"] = -uo.loc[neg, "value"]
    print(f"  irrigant-in rows negated: {int(neg.sum()):,}")
    urine = uo.groupby("stay_id")["value"].sum().rename("urine_24h").to_frame()
    n_neg = int((urine["urine_24h"] < 0).sum())
    if n_neg:
        print(f"  net-negative 24h urine totals: {n_neg:,} stays "
              f"(irrigant in exceeded volume out; left unclipped)")
    sev = sev.merge(urine.reset_index(), on="stay_id", how="left")
    print(f"MIMIC: urine recorded for {sev['urine_24h'].notna().sum():,} stays")

    for v in ("hr", "map", "rr", "temp"):
        sev[f"{v}_worst"] = worst_of(sev[f"{v}_min"], sev[f"{v}_max"],
                                     WORST_RULE[v], MIDPOINT.get(v))
    keep = ["stay_id", "hr_worst", "map_worst", "rr_worst", "temp_worst",
            "gcs_worst", "urine_24h",
            "hr_min", "hr_max", "map_min", "map_max",
            "rr_min", "rr_max", "temp_min", "temp_max"]
    sev = sev[keep]
    for c in keep[1:]:
        print(f"  {c:12s} present {sev[c].notna().mean():.3f}  "
              f"median {sev[c].median():.1f}")
    print("  compare the *_worst medians against the eICU block below; a large "
          "gap means the selection rules still disagree")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sev.to_csv(out_path, index=False)
    print(f"MIMIC severity -> {out_path}  ({len(sev):,} stays)")


def build_eicu(eicu_root: Path, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        print(f"eICU severity cached at {out_path}")
        return

    aps = pd.read_csv(find_table(eicu_root, "apacheApsVar.csv.gz",
                                 "apacheapsvar.csv.gz"))
    print(f"apacheApsVar: {len(aps):,} rows, columns: {list(aps.columns)}")

    need = ["patientunitstayid", "eyes", "motor", "verbal", "heartrate",
            "meanbp", "respiratoryrate", "temperature", "urine", "vent",
            "intubated"]
    missing = [c for c in need if c not in aps.columns]
    if missing:
        raise KeyError(f"apacheApsVar missing expected columns: {missing}")
    aps = aps[need].copy()

    # apacheApsVar uses -1 as the missing sentinel for unmeasured variables.
    for c in need[1:]:
        aps[c] = pd.to_numeric(aps[c], errors="coerce")
        aps.loc[aps[c] < 0, c] = np.nan

    gcs = aps[["eyes", "motor", "verbal"]]
    aps["gcs_worst"] = gcs.sum(axis=1).where(gcs.notna().all(axis=1))
    aps["mech_vent_aps"] = ((aps["vent"] == 1) | (aps["intubated"] == 1)).astype(float)
    aps.loc[aps["vent"].isna() & aps["intubated"].isna(), "mech_vent_aps"] = np.nan

    out = aps.rename(columns={"heartrate": "hr_worst", "meanbp": "map_worst",
                              "respiratoryrate": "rr_worst",
                              "temperature": "temp_worst",
                              "urine": "urine_24h"})
    for v, name in (("hr_worst", "hr"), ("map_worst", "map"),
                    ("rr_worst", "rr"), ("temp_worst", "temp")):
        lo, hi = RANGES[name]
        out.loc[~out[v].between(lo, hi), v] = np.nan

    # elective surgery, if apachePredVar carries it
    try:
        pv = pd.read_csv(find_table(eicu_root, "apachePredVar.csv.gz",
                                    "apachepredvar.csv.gz"))
        print(f"apachePredVar columns: {list(pv.columns)}")
        if "electivesurgery" in pv.columns:
            e = pv[["patientunitstayid", "electivesurgery"]].copy()
            e["electivesurgery"] = pd.to_numeric(e["electivesurgery"],
                                                 errors="coerce")
            e.loc[e["electivesurgery"] < 0, "electivesurgery"] = np.nan
            out = out.merge(e, on="patientunitstayid", how="left")
        else:
            print("  electivesurgery ABSENT — the elective term cannot be "
                  "harmonised and must be dropped from BOTH cohorts")
    except FileNotFoundError:
        print("  apachePredVar not found — elective term unavailable")

    keep = [c for c in ["patientunitstayid", "hr_worst", "map_worst",
                        "rr_worst", "temp_worst", "gcs_worst", "urine_24h",
                        "mech_vent_aps", "electivesurgery"] if c in out.columns]
    out = out[keep]
    for c in keep[1:]:
        print(f"  {c:16s} present {out[c].notna().mean():.3f}  "
              f"median {out[c].median():.1f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"eICU severity -> {out_path}  ({len(out):,} stays)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-root", type=Path)
    ap.add_argument("--eicu-root", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./severity_v3"))
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
    build_mimic(args.mimic_root, args.out_dir / "mimic_severity_v3.csv",
                args.force)
    build_eicu(args.eicu_root, args.out_dir / "eicu_severity_v3.csv",
               args.force)
    print("\nComponent coverage differs between cohorts. Check the 'present' "
          "shares above: any component missing in a large share of one cohort "
          "will drive complete-case selection when it enters the model.")


if __name__ == "__main__":
    main()
