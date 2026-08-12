#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Paper 17: cross-vital-sign replication.

QUESTION
--------
The central findings — source stream determines the measure, site dominates the
variance, unit contributes little — were established for heart rate.  Are they a
property of heart-rate documentation, or of routinely charted vital-sign process
measures generally?

VARIABLES
---------
  SpO2              MIMIC 220277 (O2 saturation pulseoxymetry, 25.35 rec/stay)
                    eICU  nurseCharting valname 'O2 Saturation'
                          vitalPeriodic column sao2
  Respiratory rate  MIMIC 220210 (Respiratory Rate, 25.66 rec/stay)
                    eICU  nurseCharting valname 'Respiratory Rate'
                          vitalPeriodic column respiration

EXCLUDED, AND WHY
-----------------
Alarm-limit and desaturation-limit items (223769, 223770, 226253, 224161,
224162) are documentation events, not observations: each appears at roughly 2.8
records per stay with 53% of records in three clock hours (00, 08, 20), the same
shift-anchored profile as the heart-rate alarm items.  SpO2 carries three such
items rather than two.

Ventilator-derived respiratory rate (224688 Set, 224689 spontaneous, 224690
Total) is excluded: 4.5 records per stay in roughly 35,000 stays on a distinct
charting profile.  It is a device stream, and mixing it into a nurse stream is
the error this study documents.  Respiratory rate therefore uses 220210 alone,
as heart rate used 220045 alone.

REPLICATED ANALYSES (deliberately compact)
------------------------------------------
  A  nurse-versus-monitor bottom-decile concordance within eICU
  B  database eta^2, MIMIC paired against each eICU stream
  C  hospital versus unit-within-hospital variance in the nurse streams
  D  MIMIC's position in the distribution of eICU units

The mortality, admission-hour and full timing-metric analyses are not repeated.

Usage:
  python paper17_cross_vitals.py \
      --mimic-root ~/physionet.org/files/mimiciv/3.1 \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --out-dir ~/bcst/cross_vitals

Runtime: one chartevents pass, one nurseCharting pass, one vitalPeriodic pass;
roughly 40 min. All cached to parquet; rerun with --force to rebuild.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_MIN = 24 * 60
WINDOW_H = 24.0
CHUNKSIZE = 5_000_000
MIN_RECORDS = 3
MIN_HOSP = 500
MIN_CELL = 50

VARS = {
    "spo2": {
        "mimic_itemid": 220277,
        "eicu_valname": "O2 Saturation",
        "vp_column": "sao2",
        "range": (50.0, 100.0),
    },
    "resp_rate": {
        "mimic_itemid": 220210,
        "eicu_valname": "Respiratory Rate",
        "vp_column": "respiration",
        "range": (4.0, 60.0),
    },
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


def metrics(ev, key, off):
    ev = ev.sort_values([key, off])
    g = ev.groupby(key)[off]
    out = pd.DataFrame({"n_records": g.size()})
    ev = ev.assign(gap=g.diff())
    gg = ev.dropna(subset=["gap"]).groupby(key)["gap"]
    out["median_interval_min"] = gg.median()
    out["max_interval_min"] = gg.max()
    out["frac_time_in_gaps"] = gg.apply(
        lambda s: float(s[s > 120].sum()) / WINDOW_MIN)
    return out.reset_index()


def eta2(y, groups):
    y = np.asarray(y, float); groups = np.asarray(groups)
    ok = ~np.isnan(y); y, groups = y[ok], groups[ok]
    if len(y) < 10:
        return np.nan
    grand = y.mean(); sst = ((y - grand) ** 2).sum()
    if sst <= 0:
        return np.nan
    d = pd.DataFrame({"y": y, "g": groups}).groupby("g")["y"].agg(["mean", "size"])
    return float((d["size"] * (d["mean"] - grand) ** 2).sum() / sst)


# ---------------------------------------------------------------- extraction --

def extract_mimic(root, cache, force):
    if cache.exists() and not force:
        print(f"  cached: {cache.name}")
        return pd.read_parquet(cache)
    ids = {v["mimic_itemid"]: k for k, v in VARS.items()}
    stays = pd.read_csv(find_table(root, "icu/icustays.csv.gz"),
                        usecols=["stay_id", "intime"], parse_dates=["intime"])
    kept, scanned = [], 0
    for i, ch in enumerate(pd.read_csv(
            find_table(root, "icu/chartevents.csv.gz"),
            usecols=["stay_id", "itemid", "charttime", "valuenum"],
            parse_dates=["charttime"], chunksize=CHUNKSIZE)):
        scanned += len(ch)
        ch = ch[ch["itemid"].isin(ids)].dropna(
            subset=["valuenum", "stay_id", "charttime"])
        if ch.empty:
            continue
        ch = ch.merge(stays, on="stay_id", how="inner")
        h = (ch["charttime"] - ch["intime"]).dt.total_seconds() / 3600.0
        ch = ch[(h >= 0) & (h < WINDOW_H)].copy()
        ch["offset_min"] = h[ch.index] * 60.0
        ch["variable"] = ch["itemid"].map(ids)
        keep = []
        for var, d in ch.groupby("variable"):
            lo, hi = VARS[var]["range"]
            keep.append(d[d["valuenum"].between(lo, hi)])
        ch = pd.concat(keep) if keep else ch.iloc[:0]
        if not ch.empty:
            kept.append(ch[["stay_id", "variable", "offset_min"]])
        if (i + 1) % 10 == 0:
            print(f"    ...{scanned:,} chartevents rows")
    ev = pd.concat(kept, ignore_index=True)
    ev.to_parquet(cache, index=False)
    print(f"  cached {len(ev):,} MIMIC records")
    return ev


def extract_eicu_nc(root, cache, force):
    if cache.exists() and not force:
        print(f"  cached: {cache.name}")
        return pd.read_parquet(cache)
    want = {v["eicu_valname"]: k for k, v in VARS.items()}
    kept, scanned = [], 0
    for i, ch in enumerate(pd.read_csv(
            find_table(root, "nurseCharting.csv.gz", "nursecharting.csv.gz"),
            usecols=["patientunitstayid", "nursingchartoffset",
                     "nursingchartcelltypevalname", "nursingchartvalue"],
            chunksize=CHUNKSIZE, low_memory=False)):
        scanned += len(ch)
        ch = ch[ch["nursingchartcelltypevalname"].isin(want)]
        if ch.empty:
            continue
        ch = ch.assign(
            variable=ch["nursingchartcelltypevalname"].map(want),
            v=pd.to_numeric(ch["nursingchartvalue"], errors="coerce")
        ).dropna(subset=["v"])
        ch = ch[(ch["nursingchartoffset"] >= 0)
                & (ch["nursingchartoffset"] < WINDOW_MIN)]
        keep = []
        for var, d in ch.groupby("variable"):
            lo, hi = VARS[var]["range"]
            keep.append(d[d["v"].between(lo, hi)])
        ch = pd.concat(keep) if keep else ch.iloc[:0]
        if not ch.empty:
            kept.append(ch[["patientunitstayid", "variable",
                            "nursingchartoffset"]]
                        .rename(columns={"nursingchartoffset": "offset_min"}))
        if (i + 1) % 10 == 0:
            print(f"    ...{scanned:,} nurseCharting rows")
    ev = pd.concat(kept, ignore_index=True)
    ev.to_parquet(cache, index=False)
    print(f"  cached {len(ev):,} eICU nurse records")
    return ev


def extract_eicu_vp(root, cache, force):
    if cache.exists() and not force:
        print(f"  cached: {cache.name}")
        return pd.read_parquet(cache)
    cols = ["patientunitstayid", "observationoffset"] + \
           [v["vp_column"] for v in VARS.values()]
    kept, scanned = [], 0
    for i, ch in enumerate(pd.read_csv(
            find_table(root, "vitalPeriodic.csv.gz", "vitalperiodic.csv.gz"),
            usecols=cols, chunksize=CHUNKSIZE)):
        scanned += len(ch)
        ch = ch[(ch["observationoffset"] >= 0)
                & (ch["observationoffset"] < WINDOW_MIN)]
        if ch.empty:
            continue
        for var, spec in VARS.items():
            c = spec["vp_column"]; lo, hi = spec["range"]
            d = ch[["patientunitstayid", "observationoffset", c]].dropna(
                subset=[c])
            d = d[d[c].between(lo, hi)]
            if not d.empty:
                kept.append(d[["patientunitstayid", "observationoffset"]]
                            .assign(variable=var)
                            .rename(columns={"observationoffset": "offset_min"}))
        if (i + 1) % 10 == 0:
            print(f"    ...{scanned:,} vitalPeriodic rows")
    ev = pd.concat(kept, ignore_index=True)
    ev.to_parquet(cache, index=False)
    print(f"  cached {len(ev):,} eICU monitor records")
    return ev


# ------------------------------------------------------------------ analyses --

def run_variable(var, mim_ev, nc_ev, vp_ev, ps, pat, out_dir):
    print("\n" + "=" * 78)
    print(f"{var.upper()}")
    print("=" * 78)

    mim = metrics(mim_ev[mim_ev["variable"] == var][["stay_id", "offset_min"]],
                  "stay_id", "offset_min").merge(ps, on="stay_id", how="inner")
    mim = mim[(mim["los"] >= 1.0) & (mim["n_records"] >= MIN_RECORDS)]
    mim["unit_id"] = "MIMIC:" + mim["careunit"]

    streams = {}
    for name, ev, key in (("nurseCharting", nc_ev, "patientunitstayid"),
                          ("vitalPeriodic", vp_ev, "patientunitstayid")):
        d = metrics(ev[ev["variable"] == var][[key, "offset_min"]], key,
                    "offset_min").merge(pat, on=key, how="inner")
        d = d[(d["unitdischargeoffset"] >= WINDOW_MIN)
              & (d["n_records"] >= MIN_RECORDS)]
        d["unit_id"] = ("eICU-" + d["hospitalid"].astype(str) + ":"
                        + d["unittype"].astype(str))
        streams[name] = d

    print(f"  MIMIC {len(mim):,} stays (median {mim['n_records'].median():.0f} "
          f"records, interval {mim['median_interval_min'].median():.0f} min)")
    for n, d in streams.items():
        print(f"  eICU {n:14s} {len(d):,} stays (median "
              f"{d['n_records'].median():.0f} records, interval "
              f"{d['median_interval_min'].median():.0f} min)")

    rows = {"variable": var}

    # A. nurse vs monitor within eICU
    a = streams["vitalPeriodic"][["patientunitstayid", "n_records"]].rename(
        columns={"n_records": "n_vp"})
    b = streams["nurseCharting"][["patientunitstayid", "n_records"]].rename(
        columns={"n_records": "n_nc"})
    both = a.merge(b, on="patientunitstayid", how="inner")
    lo_vp = both["n_vp"] <= both["n_vp"].quantile(0.10)
    lo_nc = both["n_nc"] <= both["n_nc"].quantile(0.10)
    inter = int((lo_vp & lo_nc).sum())
    rows.update({
        "n_both_streams": len(both),
        "spearman": float(both["n_vp"].rank().corr(both["n_nc"].rank())),
        "decile_positive_agreement": inter / max(int(lo_vp.sum()), 1),
        "decile_jaccard": inter / max(int((lo_vp | lo_nc).sum()), 1),
    })
    print(f"\n  A. same stays, two tables (n={len(both):,}): "
          f"Spearman {rows['spearman']:.3f}; bottom-decile positive agreement "
          f"{rows['decile_positive_agreement']:.3f} (chance 0.100), "
          f"Jaccard {rows['decile_jaccard']:.3f}")

    # B. database eta^2 under each pairing
    print("\n  B. database eta^2, MIMIC paired against each eICU stream")
    for m in ("n_records", "median_interval_min"):
        for n, d in streams.items():
            pool = pd.concat([mim[[m]].assign(db="MIMIC"),
                              d[[m]].assign(db="eICU")], ignore_index=True)
            e = eta2(pool[m], pool["db"])
            rows[f"db_eta2_{m}_vs_{n}"] = e
            print(f"     {m:22s} vs {n:14s} {e:.3f}")

    # C. hospital vs unit within the nurse stream
    print("\n  C. eICU nurse stream: hospital vs unit-within-hospital eta^2")
    nc = streams["nurseCharting"]
    cnt = nc["hospitalid"].value_counts()
    nc = nc[nc["hospitalid"].isin(cnt[cnt >= MIN_HOSP].index)]
    for m in ("n_records", "median_interval_min", "frac_time_in_gaps"):
        eh = eta2(nc[m], nc["hospitalid"])
        eu = eta2(nc[m], nc["unit_id"]) - eh
        rows[f"hosp_eta2_{m}"] = eh
        rows[f"unit_eta2_{m}"] = eu
        print(f"     {m:22s} hospital {eh:.3f}   unit within hospital {eu:.3f}")
    print(f"     ({nc['hospitalid'].nunique()} hospitals, {len(nc):,} stays)")

    # D. MIMIC's position among eICU units
    print("\n  D. MIMIC percentile within the eICU unit distribution")
    for n, d in streams.items():
        cnt = d["unit_id"].value_counts()
        prof = d[d["unit_id"].isin(cnt[cnt >= MIN_CELL].index)] \
            .groupby("unit_id")["n_records"].median()
        mcnt = mim["unit_id"].value_counts()
        mprof = mim[mim["unit_id"].isin(mcnt[mcnt >= MIN_CELL].index)] \
            .groupby("unit_id")["n_records"].median()
        pcts = [(prof.values < v).mean() for v in mprof.values]
        rows[f"mimic_pctile_min_vs_{n}"] = float(np.min(pcts))
        rows[f"mimic_pctile_max_vs_{n}"] = float(np.max(pcts))
        print(f"     vs {n:14s} ({len(prof)} units): "
              f"{np.min(pcts):.2f}-{np.max(pcts):.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-root", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./cross_vitals"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("extraction")
    mim_ev = extract_mimic(args.mimic_root, args.out_dir / "mimic.parquet",
                           args.force)
    nc_ev = extract_eicu_nc(args.eicu_root, args.out_dir / "eicu_nc.parquet",
                            args.force)
    vp_ev = extract_eicu_vp(args.eicu_root, args.out_dir / "eicu_vp.parquet",
                            args.force)

    ps = pd.read_csv(args.mimic_per_stay,
                     usecols=["stay_id", "careunit", "los"])
    pat = pd.read_csv(args.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])

    out = [run_variable(v, mim_ev, nc_ev, vp_ev, ps, pat, args.out_dir)
           for v in VARS]
    pd.DataFrame(out).to_csv(args.out_dir / "cross_vitals_summary.csv",
                             index=False)

    print("\n" + "=" * 78)
    print("COMPARE AGAINST HEART RATE")
    print("=" * 78)
    print("  bottom-decile agreement between streams   0.106 (chance 0.100)")
    print("  database eta^2, count, vs monitor / nurse  0.960 / 0.026")
    print("  eICU hospital eta^2, median interval       0.761")
    print("  eICU unit within hospital, median interval 0.003")
    print("\nIf the two variables above reproduce this pattern, the findings "
          "are a property of routinely charted vital-sign process measures "
          "rather than of heart-rate documentation. If they do not, "
          "measurement transportability is variable-specific, which is a "
          "stronger warning and must be reported as such.")
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
