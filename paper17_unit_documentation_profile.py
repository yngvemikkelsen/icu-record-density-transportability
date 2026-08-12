#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "statsmodels", "pyarrow"]
# ///
"""
Does documentation practice transport across care units within one hospital?

RATIONALE
---------
The MIMIC-IV vs eICU-CRD comparison could not separate recording architecture
from clinical and documentation practice, and measurement equivalence failed
upstream of any transportability claim: n_hr_24h pooled 220045 with two
shift-anchored alarm-limit items, no severity instrument could be harmonised
without sharing the exposure's data stream, and apacheApsVar was missing for 23%
of eICU stays in a manner correlated with the exposure.

Within MIMIC-IV all of that is held constant by construction: one EHR, one
institution, one extraction, one itemid, one severity instrument, one outcome
definition.  What varies across care units is clinical and documentation
practice.  So the question becomes answerable in isolation:

    Does the documentation profile of a stay transport across care units
    within a single hospital?

If it does not — if between-unit variation within one institution is comparable
to what was seen between two databases — then "transportability across recording
architectures" was the wrong frame, and the operative heterogeneity is practice,
not architecture.

NO OUTCOME MODEL
----------------
Nothing here regresses on mortality.  Density tracks acuity within every unit,
so a per-unit density-outcome coefficient inherits the identification problem
that sank the cross-database analysis.  This script characterises the
documentation process itself, which is confound-free because it makes no claim
about patient outcomes.

WHAT IS MEASURED, PER STAY (heart rate, itemid 220045, first 24 h)
------------------------------------------------------------------
  n_records            record count (the exposure, cleanly defined)
  t_first_h            hours from ICU admission to first charted observation
  median_interval_min  typical spacing between consecutive observations
  iqr_interval_min     spacing dispersion (robust to MIMIC's on-the-hour clustering)
  max_interval_min     longest single gap
  n_gaps_gt2h/gt4h     count of gaps beyond 2 h and 4 h (nurse cadence is ~1 h)
  frac_time_in_gaps    share of the 24 h window falling inside gaps > 2 h
  on_the_hour_frac     share of records charted in the first 5 min of an hour

The last one is a pure documentation-practice signature: it reflects when the
unit charts, not how sick the patient is.

TRANSPORT IS QUANTIFIED as eta^2 of each metric on care unit, reported both raw
and conditional on log record count.  The conditional version answers the
sharper question: does the SHAPE of documentation differ across units beyond how
much of it there is?

Usage:
  python paper17_unit_documentation_profile.py \
      --mimic-root ~/physionet.org/files/mimiciv/3.1 \
      --per-stay   ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --out-dir    ~/bcst/unit_profile

Runtime: one chartevents pass, roughly 6-10 min.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HR_ITEMID = 220045
HR_RANGE = (20.0, 250.0)
WINDOW_H = 24.0
CHUNKSIZE = 5_000_000
MIN_RECORDS = 3          # intervals need at least two gaps to be meaningful


def find_table(root: Path, *candidates: str) -> Path:
    for c in candidates:
        p = root / c
        if p.exists():
            return p
        alt = p.with_suffix("")
        if alt.exists():
            return alt
    raise FileNotFoundError(f"none of {candidates} under {root}")


def extract_hr(mimic_root: Path, cache: Path, force: bool) -> pd.DataFrame:
    if cache.exists() and not force:
        print(f"using cached HR timestamps: {cache}")
        return pd.read_parquet(cache)

    stays = pd.read_csv(find_table(mimic_root, "icu/icustays.csv.gz"),
                        usecols=["stay_id", "intime", "los"],
                        parse_dates=["intime"])
    kept, scanned = [], 0
    for i, chunk in enumerate(pd.read_csv(
            find_table(mimic_root, "icu/chartevents.csv.gz"),
            usecols=["stay_id", "itemid", "charttime", "valuenum"],
            parse_dates=["charttime"], chunksize=CHUNKSIZE)):
        scanned += len(chunk)
        chunk = chunk[chunk["itemid"] == HR_ITEMID].dropna(
            subset=["valuenum", "stay_id", "charttime"])
        chunk = chunk[chunk["valuenum"].between(*HR_RANGE)]
        if chunk.empty:
            continue
        chunk = chunk.merge(stays[["stay_id", "intime"]], on="stay_id",
                            how="inner")
        h = (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600.0
        chunk = chunk[(h >= 0) & (h < WINDOW_H)].copy()
        chunk["offset_min"] = h[chunk.index] * 60.0
        if not chunk.empty:
            kept.append(chunk[["stay_id", "charttime", "offset_min"]])
        if (i + 1) % 10 == 0:
            print(f"  ...{scanned:,} chartevents rows scanned")

    ev = pd.concat(kept, ignore_index=True)
    ev = ev.merge(stays[["stay_id", "los"]], on="stay_id", how="left")
    cache.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(cache, index=False)
    print(f"cached {len(ev):,} in-window HR records -> {cache}")
    return ev


def per_stay_metrics(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.sort_values(["stay_id", "offset_min"])
    g = ev.groupby("stay_id")

    out = pd.DataFrame({
        "n_records": g.size(),
        "t_first_h": g["offset_min"].min() / 60.0,
    })

    diffs = ev.groupby("stay_id")["offset_min"].diff()
    ev = ev.assign(gap_min=diffs)
    gg = ev.dropna(subset=["gap_min"]).groupby("stay_id")["gap_min"]
    out["median_interval_min"] = gg.median()
    out["iqr_interval_min"] = gg.quantile(0.75) - gg.quantile(0.25)
    out["max_interval_min"] = gg.max()
    out["n_gaps_gt2h"] = gg.apply(lambda s: int((s > 120).sum()))
    out["n_gaps_gt4h"] = gg.apply(lambda s: int((s > 240).sum()))
    out["frac_time_in_gaps"] = gg.apply(
        lambda s: float(s[s > 120].sum()) / (WINDOW_H * 60.0))

    # documentation-practice signature: charting on the hour
    minute = ev["charttime"].dt.minute
    out["on_the_hour_frac"] = (ev.assign(on_hour=(minute < 5).astype(float))
                                 .groupby("stay_id")["on_hour"].mean())
    return out.reset_index()


def eta_sq(df: pd.DataFrame, metric: str, group: str = "careunit",
           control: str | None = None) -> tuple[float, int]:
    sub = df.dropna(subset=[metric, group] + ([control] if control else []))
    if len(sub) < 100:
        return float("nan"), len(sub)
    if control:
        base = smf.ols(f"{metric} ~ {control}", data=sub).fit()
        if base.rsquared > 0.999:
            # density alone explains everything; the partial is undefined
            return float("nan"), len(sub)
        full = smf.ols(f"{metric} ~ {control} + C({group})", data=sub).fit()
        partial = (full.rsquared - base.rsquared) / (1 - base.rsquared)
        return float(partial), len(sub)
    fit = smf.ols(f"{metric} ~ C({group})", data=sub).fit()
    return float(fit.rsquared), len(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-root", required=True, type=Path)
    ap.add_argument("--per-stay", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./unit_profile"))
    ap.add_argument("--min-los-days", type=float, default=1.0,
                    help="restrict to stays completing the 24h window; gap "
                         "metrics are truncation-biased below this")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ev = extract_hr(args.mimic_root, args.out_dir / "hr_timestamps.parquet",
                    args.force)

    print("\ncomputing per-stay documentation metrics ...")
    m = per_stay_metrics(ev)

    ps = pd.read_csv(args.per_stay, usecols=["stay_id", "careunit", "los"])
    m = m.merge(ps, on="stay_id", how="inner")
    print(f"stays with HR records: {len(m):,}")

    full = m[(m["los"] >= args.min_los_days) & (m["n_records"] >= MIN_RECORDS)].copy()
    print(f"stays completing the 24h window (los >= {args.min_los_days}d) "
          f"with >= {MIN_RECORDS} records: {len(full):,}")
    full["log_n"] = np.log(full["n_records"])

    METRICS = ["n_records", "t_first_h", "median_interval_min",
               "iqr_interval_min", "max_interval_min", "n_gaps_gt2h",
               "n_gaps_gt4h", "frac_time_in_gaps", "on_the_hour_frac"]

    # ---- unit profiles ----
    prof = (full.groupby("careunit")[METRICS].median()
                .join(full.groupby("careunit").size().rename("n_stays")))
    prof = prof[["n_stays"] + METRICS].sort_values("n_stays", ascending=False)
    print("\n" + "=" * 78)
    print("UNIT DOCUMENTATION PROFILES (medians)")
    print("=" * 78)
    print(prof.round(3).to_string())
    prof.to_csv(args.out_dir / "unit_profiles.csv")

    # ---- spread across units ----
    print("\n" + "=" * 78)
    print("DOES THE PROFILE TRANSPORT ACROSS UNITS?")
    print("=" * 78)
    print(f"{'metric':22s} {'min unit':>10s} {'max unit':>10s} "
          f"{'ratio':>7s} {'eta2':>7s} {'eta2|density':>13s}")
    print("-" * 78)
    rows = []
    for metric in METRICS:
        e_raw, n = eta_sq(full, metric)
        e_adj, _ = eta_sq(full, metric, control="log_n")
        lo, hi = prof[metric].min(), prof[metric].max()
        ratio = hi / lo if lo not in (0, np.nan) and lo > 0 else np.nan
        print(f"{metric:22s} {lo:10.3f} {hi:10.3f} "
              f"{ratio:7.2f} {e_raw:7.3f} {e_adj:13.3f}")
        rows.append({"metric": metric, "min_unit_median": lo,
                     "max_unit_median": hi, "ratio": ratio,
                     "eta2_unit": e_raw, "eta2_unit_given_density": e_adj,
                     "n": n})
    pd.DataFrame(rows).to_csv(args.out_dir / "unit_transport_eta2.csv",
                              index=False)
    print("\neta2       = share of between-stay variance attributable to unit")
    print("eta2|density = the same, after conditioning on log record count;")
    print("             non-trivial values mean the SHAPE of documentation")
    print("             differs across units beyond how much of it there is.")

    # ---- the low-density tail ----
    cut = full["n_records"].quantile(0.10)
    tail = full[full["n_records"] <= cut]
    print("\n" + "=" * 78)
    print(f"LOW-DENSITY TAIL (bottom decile, n_records <= {cut:.0f}; "
          f"{len(tail):,} stays)")
    print("=" * 78)
    tail_prof = (tail.groupby("careunit")
                     .agg(n_stays=("stay_id", "size"),
                          median_n=("n_records", "median"),
                          max_gap_h=("max_interval_min",
                                     lambda s: s.median() / 60.0),
                          frac_in_gaps=("frac_time_in_gaps", "median"),
                          median_interval_min=("median_interval_min", "median"))
                     .sort_values("n_stays", ascending=False))
    print(tail_prof.round(3).to_string())
    tail_prof.to_csv(args.out_dir / "low_density_tail_by_unit.csv")
    print("\nGap-dominated vs uniformly-thin: a stay whose median interval is "
          "near the unit norm but whose max gap is long is gap-dominated; one "
          "whose median interval is stretched throughout is uniformly thin.")

    # ---- charting rhythm by unit ----
    hours = ev.assign(hour=ev["charttime"].dt.hour).merge(
        ps[["stay_id", "careunit"]], on="stay_id", how="inner")
    rhythm = (hours.groupby(["careunit", "hour"]).size().rename("records")
                   .reset_index())
    rhythm["share"] = rhythm.groupby("careunit")["records"].transform(
        lambda s: s / s.sum())
    rhythm.to_csv(args.out_dir / "charting_hour_by_unit.csv", index=False)
    conc = (rhythm.sort_values("share", ascending=False)
                  .groupby("careunit").head(3)
                  .groupby("careunit")
                  .agg(top3_share=("share", "sum"),
                       top_hours=("hour", lambda s: sorted(s.tolist())))
                  .sort_values("top3_share", ascending=False))
    print("\n" + "=" * 78)
    print("CHARTING RHYTHM BY UNIT (share in three busiest clock hours; "
          "flat = 0.125)")
    print("=" * 78)
    print(conc.round(3).to_string())

    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
