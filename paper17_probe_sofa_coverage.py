#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy"]
# ///
"""
Coverage probe for a harmonised modified SOFA, both cohorts.

WHY A PROBE FIRST
-----------------
Every previous step in this analysis broke on an assumption about what a table
contained: n_hr_24h pooled alarm-limit itemids, apacheApsVar was missing for 23%
of eICU stays (enriched 6x in the exposure's bottom decile), urine was present
for 42%, electivesurgery for 17%.  This script asserts nothing and builds
nothing.  It searches for candidate items by label, measures first-24h coverage
per stay in both cohorts, and reports.  The scorer is written afterwards, once
the coverage is known.

COMPONENTS OF THE PLANNED SCORE
-------------------------------
  Respiration   PaO2/FiO2                     lab + chart      stream-independent*
  Coagulation   platelets                     lab              stream-independent
  Liver         bilirubin (total)             lab              stream-independent
  Renal         creatinine                    lab              stream-independent
  CNS           GCS                           chart            already built
  Cardiovascular  MAP >=70 / <70 / vasoactive chart + drugs    SHARES the exposure stream

* FiO2 is charted rather than resulted, so respiration is only partly
  stream-independent.  This is the component most likely to fail on coverage,
  because PaO2 requires an ABG and ABGs are drawn mainly in ventilated patients.

Systolic BP is probed too, for a qSOFA full-cohort robustness check against the
severity-coverage confound.  Not part of SOFA.

Usage:
  python paper17_probe_sofa_coverage.py \
      --mimic-root ~/physionet.org/files/mimiciv/3.1 \
      --eicu-root  ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir    ~/bcst/sofa_probe

Runtime: one labevents pass and one eICU lab pass; roughly 10-20 min.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_H = 24.0
CHUNKSIZE = 5_000_000

# Label patterns, deliberately broad. The probe reports what matches; it does
# not decide which itemid is correct.
LAB_PATTERNS = {
    "platelets": r"platelet",
    "bilirubin": r"bilirubin",
    "creatinine": r"creatinine",
    "pao2": r"\bpo2\b|partial pressure of oxygen|^oxygen$|pO2",
}
CHART_PATTERNS = {
    "fio2": r"fio2|inspired o2 fraction",
    "sbp": r"blood pressure systolic|arterial blood pressure systolic|"
           r"non invasive blood pressure systolic|art bp systolic",
}
# Brand and hyphenated spellings matter: eICU drugname is free text entered per
# site, so "Neo-Synephrine", "Levophed" and spacing variants all appear.
VASOACTIVE_PATTERN = (r"norepinephrine|noradrenaline|epinephrine|adrenaline|"
                      r"dopamine|dobutamine|phenylephrine|vasopressin|"
                      r"levophed|neo[-\s]?synephrine|milrinone")

EICU_LAB_PATTERNS = LAB_PATTERNS | {"fio2": r"fio2", "gcs": r"gcs"}


def find_table(root: Path, *candidates: str) -> Path:
    for c in candidates:
        p = root / c
        if p.exists():
            return p
        alt = p.with_suffix("")
        if alt.exists():
            return alt
    raise FileNotFoundError(f"none of {candidates} under {root}")


def match_labels(df, label_col, pattern):
    return df[df[label_col].str.contains(pattern, case=False, na=False,
                                         regex=True)]


# ---------------------------------------------------------------------------
# MIMIC
# ---------------------------------------------------------------------------

def probe_mimic(root: Path, out_dir: Path) -> None:
    print("=" * 78)
    print("MIMIC-IV")
    print("=" * 78)

    stays = pd.read_csv(find_table(root, "icu/icustays.csv.gz"),
                        usecols=["stay_id", "hadm_id", "intime"],
                        parse_dates=["intime"])
    n_stays = len(stays)
    print(f"ICU stays: {n_stays:,}")

    # ---- candidate lab items ----
    dli = pd.read_csv(find_table(root, "hosp/d_labitems.csv.gz"))
    lab_ids, rows = {}, []
    print("\nCandidate lab items (d_labitems):")
    for comp, pat in LAB_PATTERNS.items():
        m = match_labels(dli, "label", pat)
        if "fluid" in m.columns:
            m = m[m["fluid"].str.contains("blood", case=False, na=False)]
        lab_ids[comp] = m["itemid"].tolist()
        print(f"\n  [{comp}] {len(m)} matches")
        cols = [c for c in ("itemid", "label", "fluid", "category")
                if c in m.columns]
        print(m[cols].head(12).to_string(index=False))
        rows += [{"component": comp, **r} for r in m[cols].to_dict("records")]
    pd.DataFrame(rows).to_csv(out_dir / "mimic_candidate_labitems.csv",
                              index=False)

    # ---- labevents pass ----
    all_lab_ids = sorted({i for v in lab_ids.values() for i in v})
    print(f"\nscanning labevents for {len(all_lab_ids)} itemids ...")
    hadm_intime = stays.dropna(subset=["hadm_id"])[["hadm_id", "stay_id",
                                                    "intime"]]
    kept, scanned = [], 0
    for i, chunk in enumerate(pd.read_csv(
            find_table(root, "hosp/labevents.csv.gz"),
            usecols=["hadm_id", "itemid", "charttime", "valuenum"],
            parse_dates=["charttime"], chunksize=CHUNKSIZE)):
        scanned += len(chunk)
        chunk = chunk[chunk["itemid"].isin(all_lab_ids)].dropna(
            subset=["valuenum", "hadm_id"])
        if chunk.empty:
            continue
        chunk = chunk.merge(hadm_intime, on="hadm_id", how="inner")
        h = (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600.0
        chunk = chunk[(h >= 0) & (h < WINDOW_H)]
        if not chunk.empty:
            kept.append(chunk[["stay_id", "itemid", "valuenum"]])
        if (i + 1) % 10 == 0:
            print(f"  ...{scanned:,} labevents rows scanned")
    ev = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(
        columns=["stay_id", "itemid", "valuenum"])
    print(f"in-window lab records: {len(ev):,}")

    cov = []
    print("\nMIMIC first-24h coverage (share of ICU stays with >=1 value):")
    for comp, ids in lab_ids.items():
        s = ev[ev["itemid"].isin(ids)]
        share = s["stay_id"].nunique() / n_stays
        med = s["valuenum"].median() if len(s) else np.nan
        print(f"  {comp:12s} {share:.3f}   median {med:.2f}   "
              f"records {len(s):,}")
        cov.append({"cohort": "MIMIC", "component": comp, "coverage": share,
                    "median": med, "n_records": int(len(s))})

    # ---- chart items: FiO2, systolic BP; and vasoactives ----
    di = pd.read_csv(find_table(root, "icu/d_items.csv.gz"))
    print("\nCandidate chart items:")
    for comp, pat in CHART_PATTERNS.items():
        m = match_labels(di, "label", pat)
        print(f"\n  [{comp}] {len(m)} matches")
        cols = [c for c in ("itemid", "label", "linksto", "unitname")
                if c in m.columns]
        print(m[cols].head(12).to_string(index=False))

    print("\nCandidate vasoactive items (inputevents):")
    vaso = match_labels(di, "label", VASOACTIVE_PATTERN)
    if "linksto" in vaso.columns:
        vaso = vaso[vaso["linksto"] == "inputevents"]
    cols = [c for c in ("itemid", "label", "linksto", "unitname")
            if c in vaso.columns]
    print(vaso[cols].to_string(index=False))
    vaso.to_csv(out_dir / "mimic_candidate_vasoactives.csv", index=False)

    if len(vaso):
        ie = pd.read_csv(find_table(root, "icu/inputevents.csv.gz"),
                         usecols=["stay_id", "itemid", "starttime"],
                         parse_dates=["starttime"])
        ie = ie[ie["itemid"].isin(vaso["itemid"])].merge(
            stays[["stay_id", "intime"]], on="stay_id", how="inner")
        h = (ie["starttime"] - ie["intime"]).dt.total_seconds() / 3600.0
        ie = ie[(h >= 0) & (h < WINDOW_H)]
        share = ie["stay_id"].nunique() / n_stays
        print(f"\n  vasoactive in first 24h: {share:.3f} of stays")
        cov.append({"cohort": "MIMIC", "component": "vasoactive",
                    "coverage": share, "median": np.nan,
                    "n_records": int(len(ie))})
        by_item = (ie.groupby("itemid")["stay_id"].nunique()
                     .rename("stays").reset_index()
                     .merge(di[["itemid", "label"]], on="itemid", how="left")
                     .sort_values("stays", ascending=False))
        print(by_item.to_string(index=False))
        by_item.to_csv(out_dir / "mimic_vasoactive_usage.csv", index=False)

    pd.DataFrame(cov).to_csv(out_dir / "mimic_sofa_coverage.csv", index=False)


# ---------------------------------------------------------------------------
# eICU
# ---------------------------------------------------------------------------

def probe_eicu(root: Path, out_dir: Path) -> None:
    print("\n" + "=" * 78)
    print("eICU-CRD")
    print("=" * 78)

    pat = pd.read_csv(find_table(root, "patient.csv.gz"),
                      usecols=["patientunitstayid"])
    n_stays = len(pat)
    print(f"unit stays: {n_stays:,}")

    print("\nscanning lab ...")
    names, kept, scanned = [], [], 0
    for i, chunk in enumerate(pd.read_csv(
            find_table(root, "lab.csv.gz"),
            usecols=["patientunitstayid", "labresultoffset", "labname",
                     "labresult"], chunksize=CHUNKSIZE)):
        scanned += len(chunk)
        names.append(chunk["labname"].value_counts())
        win = chunk[(chunk["labresultoffset"] >= 0)
                    & (chunk["labresultoffset"] < WINDOW_H * 60)]
        win = win.dropna(subset=["labresult"])
        if not win.empty:
            kept.append(win[["patientunitstayid", "labname", "labresult"]])
        if (i + 1) % 5 == 0:
            print(f"  ...{scanned:,} lab rows scanned")

    allnames = pd.concat(names).groupby(level=0).sum().sort_values(
        ascending=False)
    allnames.rename("records").to_csv(out_dir / "eicu_labname_counts.csv")
    print(f"\ndistinct labname values: {len(allnames):,} "
          f"(full counts written to eicu_labname_counts.csv)")

    ev = pd.concat(kept, ignore_index=True)
    print(f"in-window lab records: {len(ev):,}")

    cov = []
    print("\neICU first-24h coverage (share of unit stays with >=1 value):")
    for comp, p in EICU_LAB_PATTERNS.items():
        hits = [n for n in allnames.index
                if re.search(p, str(n), flags=re.I)]
        s = ev[ev["labname"].isin(hits)]
        share = s["patientunitstayid"].nunique() / n_stays
        med = pd.to_numeric(s["labresult"], errors="coerce").median() \
            if len(s) else np.nan
        print(f"  {comp:12s} {share:.3f}   median {med:.2f}   "
              f"labnames matched: {hits[:6]}")
        cov.append({"cohort": "eICU", "component": comp, "coverage": share,
                    "median": med, "n_records": int(len(s)),
                    "labnames": "|".join(map(str, hits))})

    print("\nscanning infusionDrug for vasoactive naming ...")
    try:
        inf = pd.read_csv(find_table(root, "infusionDrug.csv.gz",
                                     "infusiondrug.csv.gz"),
                          usecols=["patientunitstayid", "infusionoffset",
                                   "drugname"])
        drugs = inf["drugname"].value_counts()
        drugs.rename("records").to_csv(out_dir / "eicu_drugname_counts.csv")
        print(f"  distinct drugname values: {len(drugs):,} "
              f"(written to eicu_drugname_counts.csv)")
        hit = inf[inf["drugname"].str.contains(VASOACTIVE_PATTERN, case=False,
                                               na=False, regex=True)]
        hit = hit[(hit["infusionoffset"] >= 0)
                  & (hit["infusionoffset"] < WINDOW_H * 60)]
        share = hit["patientunitstayid"].nunique() / n_stays
        print(f"  vasoactive in first 24h: {share:.3f} of stays")
        top = hit["drugname"].value_counts().head(25)
        print("\n  top matched drugname strings:")
        print(top.to_string())
        top.rename("records").to_csv(out_dir / "eicu_vasoactive_strings.csv")
        cov.append({"cohort": "eICU", "component": "vasoactive",
                    "coverage": share, "median": np.nan,
                    "n_records": int(len(hit)), "labnames": ""})
    except FileNotFoundError:
        print("  infusionDrug not found")

    pd.DataFrame(cov).to_csv(out_dir / "eicu_sofa_coverage.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-root", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./sofa_probe"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    probe_mimic(args.mimic_root, args.out_dir)
    probe_eicu(args.eicu_root, args.out_dir)

    print("\n" + "=" * 78)
    print("READ THE COVERAGE SHARES BEFORE THE SCORER IS WRITTEN")
    print("=" * 78)
    print("A component below roughly 0.80 in either cohort should be dropped "
          "from BOTH rather than lose stays to listwise deletion — missingness "
          "here tracks severity, and severity tracks the exposure.")
    print("PaO2 is the expected casualty: it needs an ABG, and ABGs are drawn "
          "mainly in ventilated patients.")
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
