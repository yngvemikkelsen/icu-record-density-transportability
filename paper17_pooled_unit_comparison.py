#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "statsmodels", "pyarrow"]
# ///
"""
Where does the variance in nurse charting sit: database, institution, or unit?

Two analyses, both from caches already built. No new data passes.

A. POOLED DECOMPOSITION
   Every stay from both cohorts in one model. Sequential R^2 for database, then
   hospital within database, then unit within hospital. If database explains a
   small share while hospital explains a large one, the architectural framing is
   quantitatively dead rather than merely implausible.

B. UNIT BY UNIT
   For each MIMIC care unit, its position within the distribution of eICU units
   (hospital x unittype cells), metric by metric, as a percentile rank with a
   bootstrap interval. Pooled components give one number and can hide structure;
   this asks whether MIMIC's MICU is a typical eICU MICU. It need not be uniform
   across units — MIMIC's CVICU was already the outlier on timing metrics
   (t_first_h 2.57 h against ~0.25 h elsewhere).

   A matched-type comparison is also reported, where the labels correspond. That
   is the sharper test: comparing like-labelled units removes case-mix as an
   explanation, so a difference is documentation practice.

NURSE STREAMS ONLY
------------------
MIMIC chartevents 220045 against eICU nurseCharting heart rate. eICU
vitalPeriodic is deliberately excluded: it is the monitor stream, it is
invariant everywhere (eta^2 0.000-0.021 across 8 unit types and 79 hospitals),
and pairing it against MIMIC's nurse stream is the exact stream-selection error
this analysis exists to document.

IMBALANCE
---------
MIMIC contributes one hospital against eICU's 68. The pooled distribution is
overwhelmingly eICU by construction, so the database component is estimated from
a single institution on one side and must be read as such — it is not a
two-sample comparison of architectures but a question of whether one institution
sits inside another's distribution.

Usage:
  python paper17_pooled_unit_comparison.py \
      --mimic-cache ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/pooled_comparison

Runtime: a few minutes.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

WINDOW_MIN = 24 * 60
MIN_RECORDS = 3
MIN_CELL_STAYS = 50
N_BOOT = 200

METRICS = ["n_records", "t_first_h", "median_interval_min", "iqr_interval_min",
           "max_interval_min", "n_gaps_gt30m", "n_gaps_gt2h",
           "frac_time_in_gaps"]

# MIMIC careunit -> eICU unittype. A judgement call, stated rather than buried;
# cells with no defensible counterpart are left unmapped.
TYPE_MAP = {
    "Medical Intensive Care Unit (MICU)": "MICU",
    "Surgical Intensive Care Unit (SICU)": "SICU",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)": "Med-Surg ICU",
    "Coronary Care Unit (CCU)": "Cardiac ICU",
    "Cardiac Vascular Intensive Care Unit (CVICU)": "CTICU",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)": "Neuro ICU",
    "Trauma SICU (TSICU)": "SICU",
}


def metrics_from_offsets(ev, key, offset_col):
    ev = ev.sort_values([key, offset_col])
    g = ev.groupby(key)[offset_col]
    out = pd.DataFrame({"n_records": g.size(), "t_first_h": g.min() / 60.0})
    ev = ev.assign(gap_min=g.diff())
    gg = ev.dropna(subset=["gap_min"]).groupby(key)["gap_min"]
    out["median_interval_min"] = gg.median()
    out["iqr_interval_min"] = gg.quantile(0.75) - gg.quantile(0.25)
    out["max_interval_min"] = gg.max()
    out["n_gaps_gt30m"] = gg.apply(lambda s: int((s > 30).sum()))
    out["n_gaps_gt2h"] = gg.apply(lambda s: int((s > 120).sum()))
    out["frac_time_in_gaps"] = gg.apply(
        lambda s: float(s[s > 120].sum()) / WINDOW_MIN)
    return out.reset_index()


def load_mimic(cache, per_stay):
    ev = pd.read_parquet(cache)
    m = metrics_from_offsets(ev[["stay_id", "offset_min"]], "stay_id",
                             "offset_min")
    ps = pd.read_csv(per_stay, usecols=["stay_id", "careunit", "los"])
    m = m.merge(ps, on="stay_id", how="inner")
    m = m[(m["los"] >= 1.0) & (m["n_records"] >= MIN_RECORDS)].copy()
    m["database"] = "MIMIC"
    m["hospital"] = "MIMIC-BIDMC"
    m["unittype"] = m["careunit"]
    m["unit_id"] = "MIMIC:" + m["careunit"]
    print(f"MIMIC nurse-stream stays: {len(m):,}")
    return m


def load_eicu(cache, eicu_root):
    ev = pd.read_parquet(cache)
    m = metrics_from_offsets(ev, "patientunitstayid", "observationoffset")
    pat = pd.read_csv(eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    m = m.merge(pat, on="patientunitstayid", how="inner")
    m = m[(m["unitdischargeoffset"] >= WINDOW_MIN)
          & (m["n_records"] >= MIN_RECORDS)].copy()
    m["database"] = "eICU"
    m["hospital"] = "eICU-" + m["hospitalid"].astype(str)
    m["unit_id"] = m["hospital"] + ":" + m["unittype"].astype(str)
    print(f"eICU nurse-stream stays: {len(m):,}")
    return m


def pooled_decomposition(pool, out_dir):
    print("\n" + "=" * 78)
    print("A. POOLED VARIANCE DECOMPOSITION (sequential R^2)")
    print("=" * 78)
    print(f"{'metric':22s} {'database':>9s} {'+hospital':>10s} {'+unit':>8s} "
          f"{'total':>7s}")
    print("-" * 78)
    rows = []
    for m in METRICS:
        sub = pool.dropna(subset=[m])
        if sub[m].std() == 0 or len(sub) < 100:
            print(f"{m:22s} {'constant across all stays — skipped':>45s}")
            rows.append({"metric": m, "r2_database": np.nan,
                         "r2_hospital_increment": np.nan,
                         "r2_unit_increment": np.nan, "r2_total": np.nan,
                         "n": len(sub)})
            continue
        r_db = smf.ols(f"{m} ~ C(database)", data=sub).fit().rsquared
        r_h = smf.ols(f"{m} ~ C(hospital)", data=sub).fit().rsquared
        r_u = smf.ols(f"{m} ~ C(unit_id)", data=sub).fit().rsquared
        print(f"{m:22s} {r_db:9.3f} {r_h - r_db:10.3f} {r_u - r_h:8.3f} "
              f"{r_u:7.3f}")
        rows.append({"metric": m, "r2_database": r_db,
                     "r2_hospital_increment": r_h - r_db,
                     "r2_unit_increment": r_u - r_h, "r2_total": r_u,
                     "n": len(sub)})
    pd.DataFrame(rows).to_csv(out_dir / "pooled_decomposition.csv", index=False)
    print("\nColumns are nested increments: what database explains alone, what "
          "hospital adds beyond it, what unit adds beyond hospital.")
    print("MIMIC is a single hospital, so the database column also carries "
          "everything specific to BIDMC — read it as an upper bound.")


def unit_profiles(pool, min_cell):
    counts = pool.groupby("unit_id").size()
    keep = counts[counts >= min_cell].index
    prof = (pool[pool["unit_id"].isin(keep)]
            .groupby(["unit_id", "database", "unittype"])[METRICS]
            .median().reset_index())
    prof["n_stays"] = prof["unit_id"].map(counts)
    return prof


def unit_by_unit(pool, prof, out_dir):
    mim = prof[prof["database"] == "MIMIC"]
    eic = prof[prof["database"] == "eICU"]
    print("\n" + "=" * 78)
    print(f"B. UNIT BY UNIT: MIMIC units against {len(eic)} eICU units")
    print("=" * 78)

    rows = []
    for _, u in mim.iterrows():
        print(f"\n  {u['unittype']}  (n={int(u['n_stays']):,})")
        print(f"    {'metric':22s} {'MIMIC':>9s} {'eICU p10':>9s} "
              f"{'eICU med':>9s} {'eICU p90':>9s} {'pctile':>7s} {'[95% CI]':>16s}")
        stays = pool[(pool["unit_id"] == u["unit_id"])]
        for m in METRICS:
            dist = eic[m].dropna().values
            if len(dist) < 10:
                continue
            val = u[m]
            pct = float((dist < val).mean())
            boot = []
            vals = stays[m].dropna().values
            if len(vals) > 10:
                rng = np.random.default_rng(17)
                for _ in range(N_BOOT):
                    bm = np.median(rng.choice(vals, len(vals), replace=True))
                    boot.append((dist < bm).mean())
                lo, hi = np.percentile(boot, [2.5, 97.5])
            else:
                lo = hi = np.nan
            print(f"    {m:22s} {val:9.2f} {np.percentile(dist, 10):9.2f} "
                  f"{np.median(dist):9.2f} {np.percentile(dist, 90):9.2f} "
                  f"{pct:7.3f} {f'[{lo:.2f}, {hi:.2f}]':>16s}")
            rows.append({"mimic_unit": u["unittype"], "metric": m,
                         "mimic_value": val, "eicu_p10": np.percentile(dist, 10),
                         "eicu_median": float(np.median(dist)),
                         "eicu_p90": np.percentile(dist, 90),
                         "percentile": pct, "pctile_lo": lo, "pctile_hi": hi,
                         "n_eicu_units": len(dist)})
    pd.DataFrame(rows).to_csv(out_dir / "unit_by_unit_percentiles.csv",
                              index=False)
    print("\npercentile = share of eICU units below the MIMIC value. Near 0.5 "
          "means MIMIC sits mid-distribution; near 0 or 1 means it is an "
          "outlier against the eICU cloud.")


def matched_type(pool, prof, out_dir):
    print("\n" + "=" * 78)
    print("C. MATCHED UNIT TYPE (like-labelled units only; case-mix held "
          "approximately fixed)")
    print("=" * 78)
    rows = []
    mim = prof[prof["database"] == "MIMIC"]
    eic = prof[prof["database"] == "eICU"]
    for _, u in mim.iterrows():
        target = TYPE_MAP.get(u["unittype"])
        if target is None:
            continue
        d = eic[eic["unittype"] == target]
        if len(d) < 5:
            print(f"\n  {u['unittype']} -> {target}: only {len(d)} eICU units, "
                  f"skipped")
            continue
        print(f"\n  {u['unittype']} -> eICU '{target}' ({len(d)} units)")
        for m in METRICS:
            dist = d[m].dropna().values
            if len(dist) < 5:
                continue
            pct = float((dist < u[m]).mean())
            flag = "  <-- outside" if pct < 0.10 or pct > 0.90 else ""
            print(f"    {m:22s} MIMIC {u[m]:8.2f}   eICU median "
                  f"{np.median(dist):8.2f}   pctile {pct:.3f}{flag}")
            rows.append({"mimic_unit": u["unittype"], "eicu_unittype": target,
                         "metric": m, "mimic_value": u[m],
                         "eicu_median": float(np.median(dist)),
                         "percentile": pct, "n_eicu_units": len(dist)})
    pd.DataFrame(rows).to_csv(out_dir / "matched_type_comparison.csv",
                              index=False)
    print("\nType mapping is a judgement call and is listed in TYPE_MAP at the "
          "top of this script. Unmapped MIMIC units are omitted rather than "
          "forced.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-cache", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-cache", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./pooled_comparison"))
    ap.add_argument("--min-cell-stays", type=int, default=MIN_CELL_STAYS)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mim = load_mimic(args.mimic_cache, args.mimic_per_stay)
    eic = load_eicu(args.eicu_cache, args.eicu_root)
    cols = METRICS + ["database", "hospital", "unittype", "unit_id"]
    pool = pd.concat([mim[cols], eic[cols]], ignore_index=True)
    print(f"pooled stays: {len(pool):,} "
          f"({pool['hospital'].nunique()} hospitals, "
          f"{pool['unit_id'].nunique()} unit cells)")

    pooled_decomposition(pool, args.out_dir)
    prof = unit_profiles(pool, args.min_cell_stays)
    prof.to_csv(args.out_dir / "unit_profiles_pooled.csv", index=False)
    print(f"\nunit cells with >= {args.min_cell_stays} stays: {len(prof)} "
          f"(MIMIC {int((prof['database'] == 'MIMIC').sum())}, "
          f"eICU {int((prof['database'] == 'eICU').sum())})")
    unit_by_unit(pool, prof, args.out_dir)
    matched_type(pool, prof, args.out_dir)
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
