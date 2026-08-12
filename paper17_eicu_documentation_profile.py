#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "statsmodels", "pyarrow"]
# ///
"""
Documentation profile by unit type and by hospital, eICU-CRD.

COUNTERPART TO paper17_unit_documentation_profile.py
----------------------------------------------------
MIMIC-IV returned a homogeneous profile: every care unit at a 60-minute median
interval, zero gaps beyond 2 h, eta^2|density of 0.02-0.07 on the gap metrics.
Only two things varied — time to first observation (eta^2 0.30, rising to 0.318
conditional on density) and on-the-hour charting fraction (0.071 -> 0.174).
Documentation volume is protocolised; documentation timing is not.

eICU adds a level MIMIC cannot: 208 hospitals on one EHR platform, so
institution varies while recording architecture is approximately fixed.  That
gives the decomposition its pivot.  If between-hospital variation within eICU is
comparable to the MIMIC-vs-eICU difference, the cross-database contrast is two
draws from a distribution of institutional practice and "recording architecture"
adds nothing.

TWO SOURCES, DELIBERATELY
-------------------------
  vitalPeriodic  five-minute monitor medians. Machine-generated: gaps mean
                 monitor dropout, disconnection or transport off-unit.
  nurseCharting  nurse-entered observations. This is the true counterpart to
                 MIMIC chartevents; vitalPeriodic is not.

Paper 17 v9's Table 1 contrasts MIMIC "hybrid nurse-plus-monitor" against eICU
"monitor-only".  If nurseCharting carries heart rate at a nurse cadence, that
contrast was comparing a nurse stream against a monitor stream rather than two
recording architectures — which is a measurement-equivalence problem, not an
architectural finding.  Run --source both to settle it.

CLOCK TIME
----------
eICU offsets are minutes from unit admission, not wall-clock. Clock hour needs
patient.unitadmittime24. The script checks for it and skips the on-the-hour and
rhythm metrics if absent rather than assuming.

Usage:
  python paper17_eicu_documentation_profile.py \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --source both \
      --out-dir  ~/bcst/unit_profile_eicu

Runtime: vitalPeriodic ~10 min, nurseCharting ~15 min. Both cached to parquet.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

WINDOW_MIN = 24 * 60
CHUNKSIZE = 5_000_000
HR_RANGE = (20.0, 250.0)
MIN_RECORDS = 3
GAP_THRESHOLDS = [30, 120, 240]

METRICS = ["n_records", "t_first_h", "median_interval_min",
           "iqr_interval_min", "max_interval_min", "n_gaps_gt30m",
           "n_gaps_gt2h", "n_gaps_gt4h", "frac_time_in_gaps"]


def find_table(root: Path, *candidates: str) -> Path:
    for c in candidates:
        p = root / c
        if p.exists():
            return p
        alt = p.with_suffix("")
        if alt.exists():
            return alt
    raise FileNotFoundError(f"none of {candidates} under {root}")


def load_patient(root: Path) -> pd.DataFrame:
    p = pd.read_csv(find_table(root, "patient.csv.gz"))
    print(f"patient: {len(p):,} rows")
    have = [c for c in ("patientunitstayid", "hospitalid", "unittype",
                        "unitdischargeoffset", "unitadmittime24")
            if c in p.columns]
    missing = {"hospitalid", "unittype", "unitdischargeoffset"} - set(have)
    if missing:
        raise KeyError(f"patient lacks {missing}")
    if "unitadmittime24" not in have:
        print("  NOTE: unitadmittime24 absent — clock-time metrics skipped")
    return p[have]


def extract_vitalperiodic(root: Path, cache: Path, force: bool) -> pd.DataFrame:
    if cache.exists() and not force:
        print(f"using cached vitalPeriodic offsets: {cache}")
        return pd.read_parquet(cache)
    kept, scanned = [], 0
    for i, chunk in enumerate(pd.read_csv(
            find_table(root, "vitalPeriodic.csv.gz", "vitalperiodic.csv.gz"),
            usecols=["patientunitstayid", "observationoffset", "heartrate"],
            chunksize=CHUNKSIZE)):
        scanned += len(chunk)
        chunk = chunk.dropna(subset=["heartrate"])
        chunk = chunk[chunk["heartrate"].between(*HR_RANGE)]
        chunk = chunk[(chunk["observationoffset"] >= 0)
                      & (chunk["observationoffset"] < WINDOW_MIN)]
        if not chunk.empty:
            kept.append(chunk[["patientunitstayid", "observationoffset"]]
                        .astype("int32"))
        if (i + 1) % 10 == 0:
            print(f"  ...{scanned:,} vitalPeriodic rows scanned")
    ev = pd.concat(kept, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(cache, index=False)
    print(f"cached {len(ev):,} in-window vitalPeriodic HR records")
    return ev


def extract_nursecharting(root: Path, cache: Path, force: bool) -> pd.DataFrame:
    if cache.exists() and not force:
        print(f"using cached nurseCharting offsets: {cache}")
        return pd.read_parquet(cache)
    labels_seen = []
    kept, scanned = [], 0
    for i, chunk in enumerate(pd.read_csv(
            find_table(root, "nurseCharting.csv.gz", "nursecharting.csv.gz"),
            usecols=["patientunitstayid", "nursingchartoffset",
                     "nursingchartcelltypevallabel",
                     "nursingchartcelltypevalname", "nursingchartvalue"],
            chunksize=CHUNKSIZE, low_memory=False)):
        scanned += len(chunk)
        lab = chunk["nursingchartcelltypevallabel"].astype(str)
        nam = chunk["nursingchartcelltypevalname"].astype(str)
        hr = chunk[lab.str.contains("heart rate", case=False, na=False)
                   | nam.str.contains("heart rate", case=False, na=False)]
        if len(hr):
            labels_seen.append(
                hr["nursingchartcelltypevalname"].value_counts())
            v = pd.to_numeric(hr["nursingchartvalue"], errors="coerce")
            hr = hr.assign(v=v).dropna(subset=["v"])
            hr = hr[hr["v"].between(*HR_RANGE)]
            hr = hr[(hr["nursingchartoffset"] >= 0)
                    & (hr["nursingchartoffset"] < WINDOW_MIN)]
            if not hr.empty:
                kept.append(hr[["patientunitstayid", "nursingchartoffset"]]
                            .rename(columns={"nursingchartoffset":
                                             "observationoffset"})
                            .astype("int32"))
        if (i + 1) % 10 == 0:
            print(f"  ...{scanned:,} nurseCharting rows scanned")
    if labels_seen:
        seen = pd.concat(labels_seen).groupby(level=0).sum()
        print("\n  heart-rate valname strings matched:")
        print(seen.sort_values(ascending=False).head(10).to_string())
    ev = pd.concat(kept, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(cache, index=False)
    print(f"cached {len(ev):,} in-window nurseCharting HR records")
    return ev


def per_stay_metrics(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.sort_values(["patientunitstayid", "observationoffset"])
    g = ev.groupby("patientunitstayid")["observationoffset"]
    out = pd.DataFrame({"n_records": g.size(), "t_first_h": g.min() / 60.0})

    ev = ev.assign(gap_min=g.diff())
    gg = ev.dropna(subset=["gap_min"]).groupby("patientunitstayid")["gap_min"]
    out["median_interval_min"] = gg.median()
    out["iqr_interval_min"] = gg.quantile(0.75) - gg.quantile(0.25)
    out["max_interval_min"] = gg.max()
    for t in GAP_THRESHOLDS:
        name = ("n_gaps_gt30m" if t == 30 else
                "n_gaps_gt2h" if t == 120 else "n_gaps_gt4h")
        out[name] = gg.apply(lambda s, t=t: int((s > t).sum()))
    out["frac_time_in_gaps"] = gg.apply(
        lambda s: float(s[s > 120].sum()) / WINDOW_MIN)
    return out.reset_index()


def eta_sq(df, metric, group, control=None):
    sub = df.dropna(subset=[metric, group] + ([control] if control else []))
    if sub[group].nunique() < 2 or len(sub) < 100:
        return float("nan"), len(sub)
    if control:
        base = smf.ols(f"{metric} ~ {control}", data=sub).fit()
        if base.rsquared > 0.999:
            # density alone explains everything; the partial is undefined
            return float("nan"), len(sub)
        full = smf.ols(f"{metric} ~ {control} + C({group})", data=sub).fit()
        return float((full.rsquared - base.rsquared) / (1 - base.rsquared)), len(sub)
    return float(smf.ols(f"{metric} ~ C({group})", data=sub).fit().rsquared), len(sub)


def report(full, label, out_dir, min_hosp_stays):
    print("\n" + "=" * 78)
    print(f"{label}: PROFILE BY UNIT TYPE (medians)")
    print("=" * 78)
    prof = (full.groupby("unittype")[METRICS].median()
                .join(full.groupby("unittype").size().rename("n_stays"))
                .sort_values("n_stays", ascending=False))
    prof = prof[["n_stays"] + METRICS]
    print(prof.round(3).to_string())
    prof.to_csv(out_dir / f"{label}_unittype_profiles.csv")

    counts = full["hospitalid"].value_counts()
    big = counts[counts >= min_hosp_stays].index
    hosp = full[full["hospitalid"].isin(big)].copy()
    print(f"\nhospitals with >= {min_hosp_stays} stays: {len(big)} "
          f"({len(hosp):,} stays)")

    print("\n" + "=" * 78)
    print(f"{label}: DOES THE PROFILE TRANSPORT?")
    print("=" * 78)
    print(f"{'metric':22s} {'eta2 unit':>10s} {'|density':>9s} "
          f"{'eta2 hosp':>10s} {'|density':>9s} {'hosp p10':>9s} "
          f"{'hosp p90':>9s}")
    print("-" * 78)
    rows = []
    for m in METRICS:
        eu, _ = eta_sq(full, m, "unittype")
        eud, _ = eta_sq(full, m, "unittype", control="log_n")
        eh, _ = eta_sq(hosp, m, "hospitalid")
        ehd, _ = eta_sq(hosp, m, "hospitalid", control="log_n")
        hm = hosp.groupby("hospitalid")[m].median()
        p10, p90 = hm.quantile(0.10), hm.quantile(0.90)
        print(f"{m:22s} {eu:10.3f} {eud:9.3f} {eh:10.3f} {ehd:9.3f} "
              f"{p10:9.2f} {p90:9.2f}")
        rows.append({"source": label, "metric": m, "eta2_unittype": eu,
                     "eta2_unittype_given_density": eud,
                     "eta2_hospital": eh,
                     "eta2_hospital_given_density": ehd,
                     "hospital_median_p10": p10, "hospital_median_p90": p90})
    pd.DataFrame(rows).to_csv(out_dir / f"{label}_transport_eta2.csv",
                              index=False)
    print("\nhosp p10/p90 are the 10th and 90th percentiles of the "
          "per-hospital median — the spread of institutional practice.")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--source", choices=["vitalperiodic", "nursecharting",
                                         "both"], default="both")
    ap.add_argument("--out-dir", type=Path, default=Path("./unit_profile_eicu"))
    ap.add_argument("--min-hosp-stays", type=int, default=500)
    ap.add_argument("--min-los-min", type=int, default=1440)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pat = load_patient(args.eicu_root)

    sources = (["vitalperiodic", "nursecharting"] if args.source == "both"
               else [args.source])
    allrows = []
    for src in sources:
        print("\n" + "#" * 78)
        print(f"# {src}")
        print("#" * 78)
        cache = args.out_dir / f"{src}_offsets.parquet"
        ev = (extract_vitalperiodic(args.eicu_root, cache, args.force)
              if src == "vitalperiodic"
              else extract_nursecharting(args.eicu_root, cache, args.force))
        m = per_stay_metrics(ev).merge(pat, on="patientunitstayid", how="inner")
        full = m[(m["unitdischargeoffset"] >= args.min_los_min)
                 & (m["n_records"] >= MIN_RECORDS)].copy()
        print(f"stays with records: {len(m):,}; completing the 24h window "
              f"with >= {MIN_RECORDS} records: {len(full):,}")
        full["log_n"] = np.log(full["n_records"])
        allrows.append(report(full, src, args.out_dir, args.min_hosp_stays))

    if allrows:
        pd.concat(allrows).to_csv(args.out_dir / "transport_eta2_all.csv",
                                  index=False)

    print("\n" + "=" * 78)
    print("READ IN THIS ORDER")
    print("=" * 78)
    print("1. nurseCharting record counts. If heart rate is charted there at a "
          "nurse cadence, Table 1's 'monitor-only' description of eICU is "
          "wrong and the v9 architectural contrast compared a nurse stream "
          "against a monitor stream.")
    print("2. eta2_hospital vs MIMIC's between-unit eta2 (0.02-0.07 on gap "
          "metrics, 0.30 on t_first_h). Larger here means institutional "
          "practice varies more than practice within one institution.")
    print("3. hosp p10/p90 spread. If it brackets MIMIC's values, the "
          "cross-database difference sits inside the eICU hospital "
          "distribution and is not architectural.")
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
