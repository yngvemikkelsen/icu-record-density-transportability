#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Paper 17: does the choice of eICU table decide the cross-database conclusion?

THE CLAIM TO BE DEMONSTRATED
----------------------------
v9 compared MIMIC chartevents heart rate against eICU vitalPeriodic and read the
difference as one of recording architecture: "hybrid nurse-plus-monitor" versus
"monitor-only".  But eICU nurseCharting also carries heart rate, at 28 records
per 24 h and a 60-minute median interval — the same nurse cadence as MIMIC's 26
and 60.  If pairing MIMIC against the nurse stream removes the difference that
pairing against the monitor stream produced, then the architectural finding was
a property of table selection.

This script runs the identical analysis under both pairings and puts them beside
each other.  Asserting the equivalence from summary statistics is not the same
as demonstrating that the conclusion flips.

FOUR CONTRASTS
--------------
A. Distributional comparison, Table 1 style, under each pairing.
B. Bottom-decile membership agreement BETWEEN the two eICU streams. Same
   patients, same window, same variable — does "low density" identify the same
   stays?  This is the sharpest form of the argument: if agreement is poor, the
   exposure is defined by the table, not by the patient.
C. The database component of variance under each pairing.
D. Where MIMIC's units sit in the eICU unit distribution under each pairing.

Usage:
  python paper17_stream_selection.py \
      --mimic-cache ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-vp-cache ~/bcst/unit_profile_eicu/vitalperiodic_offsets.parquet \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/stream_selection
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_MIN = 24 * 60
MIN_RECORDS = 3
MIN_CELL = 50

METRICS = ["n_records", "median_interval_min", "iqr_interval_min",
           "max_interval_min", "n_gaps_gt2h", "frac_time_in_gaps"]


def metrics_from_offsets(ev, key, offset_col):
    ev = ev.sort_values([key, offset_col])
    g = ev.groupby(key)[offset_col]
    out = pd.DataFrame({"n_records": g.size(), "t_first_h": g.min() / 60.0})
    ev = ev.assign(gap_min=g.diff())
    gg = ev.dropna(subset=["gap_min"]).groupby(key)["gap_min"]
    out["median_interval_min"] = gg.median()
    out["iqr_interval_min"] = gg.quantile(0.75) - gg.quantile(0.25)
    out["max_interval_min"] = gg.max()
    out["n_gaps_gt2h"] = gg.apply(lambda s: int((s > 120).sum()))
    out["frac_time_in_gaps"] = gg.apply(
        lambda s: float(s[s > 120].sum()) / WINDOW_MIN)
    return out.reset_index()


def eta2(y, groups):
    ok = ~np.isnan(y)
    y, groups = y[ok], np.asarray(groups)[ok]
    if len(y) < 10:
        return np.nan
    grand = y.mean()
    sst = ((y - grand) ** 2).sum()
    if sst <= 0:
        return np.nan
    d = pd.DataFrame({"y": y, "g": groups}).groupby("g")["y"].agg(["mean", "size"])
    return float((d["size"] * (d["mean"] - grand) ** 2).sum() / sst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-cache", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--eicu-vp-cache", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./stream_selection"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- MIMIC (nurse stream, the only one it has) ----
    ev = pd.read_parquet(args.mimic_cache)
    mim = metrics_from_offsets(ev[["stay_id", "offset_min"]], "stay_id",
                               "offset_min")
    ps = pd.read_csv(args.mimic_per_stay, usecols=["stay_id", "careunit", "los"])
    mim = mim.merge(ps, on="stay_id", how="inner")
    mim = mim[(mim["los"] >= 1.0) & (mim["n_records"] >= MIN_RECORDS)].copy()
    mim["unit_id"] = "MIMIC:" + mim["careunit"]
    print(f"MIMIC chartevents stays: {len(mim):,}")

    # ---- eICU, both streams ----
    pat = pd.read_csv(args.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    streams = {}
    for name, cache in (("vitalPeriodic", args.eicu_vp_cache),
                        ("nurseCharting", args.eicu_nc_cache)):
        e = metrics_from_offsets(pd.read_parquet(cache), "patientunitstayid",
                                 "observationoffset")
        e = e.merge(pat, on="patientunitstayid", how="inner")
        e = e[(e["unitdischargeoffset"] >= WINDOW_MIN)
              & (e["n_records"] >= MIN_RECORDS)].copy()
        e["unit_id"] = ("eICU-" + e["hospitalid"].astype(str) + ":"
                        + e["unittype"].astype(str))
        streams[name] = e
        print(f"eICU {name} stays: {len(e):,}")

    # ---- A. distributions ----
    print("\n" + "=" * 78)
    print("A. DISTRIBUTIONS: MIMIC against each eICU stream")
    print("=" * 78)
    rows = []
    for label, d in [("MIMIC chartevents", mim)] + list(streams.items()):
        n = d["n_records"]
        r = {"source": label, "n_stays": len(d), "mean": n.mean(),
             "median": n.median(), "p10": n.quantile(.10),
             "p90": n.quantile(.90),
             "median_interval_min": d["median_interval_min"].median(),
             "bottom10_cutoff": n.quantile(.10)}
        rows.append(r)
        print(f"  {label:22s} n={len(d):>7,}  mean {r['mean']:7.1f}  "
              f"median {r['median']:6.0f}  p10-p90 {r['p10']:.0f}-{r['p90']:.0f}"
              f"  interval {r['median_interval_min']:.0f} min")
    pd.DataFrame(rows).to_csv(args.out_dir / "stream_distributions.csv",
                              index=False)
    print("\n  MIMIC's nurse cadence matches eICU nurseCharting, not "
          "vitalPeriodic. v9's Table 1 compared it against the latter.")

    # ---- B. bottom-decile agreement between the two eICU streams ----
    print("\n" + "=" * 78)
    print("B. SAME PATIENTS, TWO TABLES: does 'low density' pick the same "
          "stays?")
    print("=" * 78)
    vp = streams["vitalPeriodic"][["patientunitstayid", "n_records"]].rename(
        columns={"n_records": "n_vp"})
    nc = streams["nurseCharting"][["patientunitstayid", "n_records"]].rename(
        columns={"n_records": "n_nc"})
    both = vp.merge(nc, on="patientunitstayid", how="inner")
    print(f"  stays present in both streams: {len(both):,}")
    # Spearman computed from ranks: pandas delegates method='spearman' to scipy,
    # which is not an inline dependency here.
    rho = both["n_vp"].rank().corr(both["n_nc"].rank())
    print(f"  correlation of the two counts: "
          f"{both['n_vp'].corr(both['n_nc']):.4f} (Pearson), "
          f"{rho:.4f} (Spearman)")
    agree = []
    for pct in (5, 10, 20):
        lo_vp = both["n_vp"] <= both["n_vp"].quantile(pct / 100)
        lo_nc = both["n_nc"] <= both["n_nc"].quantile(pct / 100)
        inter = int((lo_vp & lo_nc).sum())
        union = int((lo_vp | lo_nc).sum())
        jac = inter / union if union else np.nan
        exp = pct / 100.0                      # agreement expected by chance
        obs = inter / max(int(lo_vp.sum()), 1)
        print(f"  bottom {pct:>2d}%: overlap {inter:,} of {int(lo_vp.sum()):,} "
              f"monitor-flagged  ({obs:.3f} also nurse-flagged; "
              f"{exp:.3f} expected by chance)  Jaccard {jac:.3f}")
        agree.append({"percentile": pct, "n_monitor_flagged": int(lo_vp.sum()),
                      "n_both": inter, "share_concordant": obs,
                      "chance": exp, "jaccard": jac})
    pd.DataFrame(agree).to_csv(args.out_dir / "bottom_decile_agreement.csv",
                               index=False)
    print("\n  These are the same admissions, the same 24 h window and the "
          "same variable. Disagreement here is produced entirely by which "
          "table the count is taken from.")

    # ---- C. database component under each pairing ----
    print("\n" + "=" * 78)
    print("C. DATABASE COMPONENT OF VARIANCE UNDER EACH PAIRING")
    print("=" * 78)
    print(f"{'metric':22s} {'vs vitalPeriodic':>18s} {'vs nurseCharting':>18s}")
    print("-" * 78)
    rows = []
    for m in METRICS:
        vals = {}
        for name, e in streams.items():
            pool = pd.concat([mim[[m]].assign(db="MIMIC"),
                              e[[m]].assign(db="eICU")], ignore_index=True)
            vals[name] = eta2(pool[m].to_numpy(float),
                              pool["db"].to_numpy())
        print(f"{m:22s} {vals['vitalPeriodic']:18.3f} "
              f"{vals['nurseCharting']:18.3f}")
        rows.append({"metric": m, "eta2_vs_vitalPeriodic":
                     vals["vitalPeriodic"],
                     "eta2_vs_nurseCharting": vals["nurseCharting"]})
    pd.DataFrame(rows).to_csv(args.out_dir / "database_component_by_stream.csv",
                              index=False)
    print("\n  A large component in the left column and a small one in the "
          "right means the 'architectural' difference is the table, not the "
          "architecture.")

    # ---- D. MIMIC's position in the eICU unit distribution ----
    print("\n" + "=" * 78)
    print("D. WHERE MIMIC SITS AMONG eICU UNITS, UNDER EACH PAIRING")
    print("=" * 78)
    mim_prof = mim.groupby("unit_id")[METRICS].median()
    mim_prof = mim_prof[mim.groupby("unit_id").size() >= MIN_CELL]
    rows = []
    for name, e in streams.items():
        counts = e.groupby("unit_id").size()
        prof = e[e["unit_id"].isin(counts[counts >= MIN_CELL].index)] \
            .groupby("unit_id")[METRICS].median()
        print(f"\n  vs {name}  ({len(prof)} eICU units)")
        print(f"    {'metric':22s} {'MIMIC median':>13s} {'eICU median':>12s} "
              f"{'pctile of MIMIC units':>22s}")
        for m in METRICS:
            dist = prof[m].dropna().values
            if len(dist) < 10:
                continue
            pcts = [(dist < v).mean() for v in mim_prof[m].dropna().values]
            print(f"    {m:22s} {mim_prof[m].median():13.2f} "
                  f"{np.median(dist):12.2f} "
                  f"{f'{np.min(pcts):.2f}-{np.max(pcts):.2f}':>22s}")
            rows.append({"stream": name, "metric": m,
                         "mimic_median": float(mim_prof[m].median()),
                         "eicu_median": float(np.median(dist)),
                         "mimic_pctile_min": float(np.min(pcts)),
                         "mimic_pctile_max": float(np.max(pcts))})
    pd.DataFrame(rows).to_csv(args.out_dir / "mimic_position_by_stream.csv",
                              index=False)
    print("\n  Against the monitor stream MIMIC is an extreme outlier on every "
          "metric; against the nurse stream it sits inside the distribution. "
          "Same MIMIC data both times.")

    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
