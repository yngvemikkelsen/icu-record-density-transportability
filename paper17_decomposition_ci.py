#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Paper 17: inferential scaffolding for the variance decomposition, plus the two
open diagnostic checks.

WHY THIS EXISTS
---------------
The decomposition is currently point estimates on clustered data. A reviewer
will ask for intervals, and the clustering is by hospital, so a naive
stay-level bootstrap would be wrong. This script resamples HOSPITALS with
replacement and recomputes the whole decomposition inside each replicate.

Two checks are folded in because both bear on how the result is written up:

  CHECK 1  n_gaps_gt30m is the only metric where the database component is
           large (0.249 against 0.016-0.071 elsewhere). If MIMIC's sub-hourly
           gap count is a mechanical consequence of rigid 60-minute charting —
           i.e. its gaps cluster tightly at 60 min while eICU's are dispersed —
           then the metric measures cadence regularity rather than gapping, and
           it should be reported as such rather than as a gap metric.

  CHECK 2  vitalPeriodic yielded 79 hospitals with >=500 stays; nurseCharting
           yielded 68. If the 11 missing hospitals differ systematically, the
           nurse-stream analysis is on a selected subset of institutions, and
           that belongs in the limitations with a number attached.

DECOMPOSITION METHOD
--------------------
One-way eta^2 computed analytically from group sums of squares rather than by
fitting OLS, so 500 replicates is cheap. The density-conditional version
residualises on log record count by least squares first, then decomposes the
residual — a linear approximation to partial eta^2, adequate here and orders of
magnitude faster than refitting C(hospital) 500 times.

THE DATABASE COMPONENT CANNOT BE BOOTSTRAPPED
---------------------------------------------
MIMIC is one hospital. A hospital-level resample cannot produce a sampling
distribution for a component estimated from a single cluster on one side, so the
database column is reported as a point estimate with that stated, and the
hospital and unit components — which carry the paper's claim — are bootstrapped
within eICU where 68 clusters exist.

Usage:
  python paper17_decomposition_ci.py \
      --mimic-cache ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-vp-cache ~/bcst/unit_profile_eicu/vitalperiodic_offsets.parquet \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/decomposition_ci
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_MIN = 24 * 60
MIN_RECORDS = 3
N_BOOT = 500
SEED = 17

METRICS = ["n_records", "t_first_h", "median_interval_min", "iqr_interval_min",
           "max_interval_min", "n_gaps_gt30m", "n_gaps_gt2h",
           "frac_time_in_gaps"]


def metrics_from_offsets(ev, key, offset_col, keep_gaps=False):
    ev = ev.sort_values([key, offset_col])
    g = ev.groupby(key)[offset_col]
    out = pd.DataFrame({"n_records": g.size(), "t_first_h": g.min() / 60.0})
    ev = ev.assign(gap_min=g.diff())
    gaps = ev.dropna(subset=["gap_min"])
    gg = gaps.groupby(key)["gap_min"]
    out["median_interval_min"] = gg.median()
    out["iqr_interval_min"] = gg.quantile(0.75) - gg.quantile(0.25)
    out["max_interval_min"] = gg.max()
    out["n_gaps_gt30m"] = gg.apply(lambda s: int((s > 30).sum()))
    out["n_gaps_gt2h"] = gg.apply(lambda s: int((s > 120).sum()))
    out["frac_time_in_gaps"] = gg.apply(
        lambda s: float(s[s > 120].sum()) / WINDOW_MIN)
    out = out.reset_index()
    return (out, gaps[[key, "gap_min"]]) if keep_gaps else (out, None)


def eta2(y: np.ndarray, groups: np.ndarray) -> float:
    """One-way eta^2 from group sums of squares."""
    ok = ~np.isnan(y)
    y, groups = y[ok], groups[ok]
    if len(y) < 10:
        return np.nan
    grand = y.mean()
    sst = ((y - grand) ** 2).sum()
    if sst <= 0:
        return np.nan
    df = pd.DataFrame({"y": y, "g": groups})
    agg = df.groupby("g")["y"].agg(["mean", "size"])
    ssb = (agg["size"] * (agg["mean"] - grand) ** 2).sum()
    return float(ssb / sst)


def residualise(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    ok = ~(np.isnan(y) | np.isnan(x))
    out = np.full_like(y, np.nan, dtype=float)
    if ok.sum() < 10 or np.nanstd(x[ok]) == 0:
        return y
    b, a = np.polyfit(x[ok], y[ok], 1)
    out[ok] = y[ok] - (a + b * x[ok])
    return out


def load_mimic(cache, per_stay):
    ev = pd.read_parquet(cache)
    m, gaps = metrics_from_offsets(ev[["stay_id", "offset_min"]], "stay_id",
                                   "offset_min", keep_gaps=True)
    ps = pd.read_csv(per_stay, usecols=["stay_id", "careunit", "los"])
    m = m.merge(ps, on="stay_id", how="inner")
    m = m[(m["los"] >= 1.0) & (m["n_records"] >= MIN_RECORDS)].copy()
    m["database"], m["hospital"] = "MIMIC", "MIMIC-BIDMC"
    m["unit_id"] = "MIMIC:" + m["careunit"]
    gaps = gaps[gaps["stay_id"].isin(m["stay_id"])]
    print(f"MIMIC nurse-stream stays: {len(m):,}")
    return m, gaps


def load_eicu(cache, eicu_root):
    ev = pd.read_parquet(cache)
    m, gaps = metrics_from_offsets(ev, "patientunitstayid",
                                   "observationoffset", keep_gaps=True)
    pat = pd.read_csv(eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    m = m.merge(pat, on="patientunitstayid", how="inner")
    m = m[(m["unitdischargeoffset"] >= WINDOW_MIN)
          & (m["n_records"] >= MIN_RECORDS)].copy()
    m["database"] = "eICU"
    m["hospital"] = "eICU-" + m["hospitalid"].astype(str)
    m["unit_id"] = m["hospital"] + ":" + m["unittype"].astype(str)
    gaps = gaps[gaps["patientunitstayid"].isin(m["patientunitstayid"])]
    print(f"eICU nurse-stream stays: {len(m):,}")
    return m, gaps


def bootstrap_eicu(eic, out_dir):
    """Hospital-clustered bootstrap of the eICU decomposition."""
    print("\n" + "=" * 78)
    print(f"eICU DECOMPOSITION, {N_BOOT} hospital-clustered bootstrap "
          f"replicates")
    print("=" * 78)
    hosps = eic["hospital"].unique()
    idx_by_h = {h: g.index.values for h, g in eic.groupby("hospital")}
    rng = np.random.default_rng(SEED)
    eic = eic.copy()
    eic["log_n"] = np.log(eic["n_records"])

    point, boots = {}, {m: {"hosp": [], "unit": [], "hosp_d": []}
                        for m in METRICS}
    for m in METRICS:
        y = eic[m].to_numpy(float)
        point[m] = {
            "hosp": eta2(y, eic["hospital"].to_numpy()),
            "unit": eta2(y, eic["unit_id"].to_numpy())
                    - eta2(y, eic["hospital"].to_numpy()),
            "hosp_d": eta2(residualise(y, eic["log_n"].to_numpy()),
                           eic["hospital"].to_numpy()),
        }

    for b in range(N_BOOT):
        pick = rng.choice(hosps, len(hosps), replace=True)
        rows = np.concatenate([idx_by_h[h] for h in pick])
        # relabel so a hospital drawn twice contributes as two clusters
        rep = eic.loc[rows]
        tag = np.concatenate([np.full(len(idx_by_h[h]), i)
                              for i, h in enumerate(pick)])
        hg = tag.astype(str)
        ug = np.char.add(hg, rep["unit_id"].to_numpy().astype(str))
        for m in METRICS:
            y = rep[m].to_numpy(float)
            eh = eta2(y, hg)
            boots[m]["hosp"].append(eh)
            boots[m]["unit"].append(eta2(y, ug) - eh)
            boots[m]["hosp_d"].append(
                eta2(residualise(y, rep["log_n"].to_numpy()), hg))
        if (b + 1) % 100 == 0:
            print(f"  ...{b + 1}/{N_BOOT}")

    print(f"\n{'metric':22s} {'hospital eta2 [95% CI]':>30s} "
          f"{'|density':>22s} {'unit within hosp':>22s}")
    print("-" * 78)
    rows = []
    for m in METRICS:
        def ci(k):
            a = np.array(boots[m][k], float)
            a = a[~np.isnan(a)]
            return (np.percentile(a, 2.5), np.percentile(a, 97.5)) \
                if len(a) else (np.nan, np.nan)
        hl, hh = ci("hosp"); dl, dh = ci("hosp_d"); ul, uh = ci("unit")
        print(f"{m:22s} {point[m]['hosp']:8.3f} [{hl:.3f}, {hh:.3f}]"
              f"{point[m]['hosp_d']:12.3f} [{dl:.3f}, {dh:.3f}]"
              f"{point[m]['unit']:9.3f} [{ul:.3f}, {uh:.3f}]")
        rows.append({"metric": m, "eta2_hospital": point[m]["hosp"],
                     "hosp_lo": hl, "hosp_hi": hh,
                     "eta2_hospital_given_density": point[m]["hosp_d"],
                     "hosp_d_lo": dl, "hosp_d_hi": dh,
                     "eta2_unit_within_hospital": point[m]["unit"],
                     "unit_lo": ul, "unit_hi": uh})
    pd.DataFrame(rows).to_csv(out_dir / "eicu_decomposition_ci.csv",
                              index=False)


def bootstrap_mimic_units(mim, out_dir):
    """MIMIC has one hospital, so the unit component is bootstrapped over stays
    within unit rather than over clusters."""
    print("\n" + "=" * 78)
    print(f"MIMIC BETWEEN-UNIT eta^2, {N_BOOT} stay-level bootstrap replicates")
    print("=" * 78)
    rng = np.random.default_rng(SEED)
    mim = mim.copy()
    mim["log_n"] = np.log(mim["n_records"])
    g = mim["unit_id"].to_numpy()
    n = len(mim)
    rows = []
    for m in METRICS:
        y = mim[m].to_numpy(float)
        pt = eta2(y, g)
        ptd = eta2(residualise(y, mim["log_n"].to_numpy()), g)
        b = [eta2(y[i], g[i]) for i in
             (rng.integers(0, n, n) for _ in range(N_BOOT))]
        b = np.array([x for x in b if not np.isnan(x)])
        lo, hi = (np.percentile(b, 2.5), np.percentile(b, 97.5)) \
            if len(b) else (np.nan, np.nan)
        print(f"{m:22s} {pt:8.4f} [{lo:.4f}, {hi:.4f}]   |density {ptd:8.4f}")
        rows.append({"metric": m, "eta2_unit": pt, "lo": lo, "hi": hi,
                     "eta2_unit_given_density": ptd})
    pd.DataFrame(rows).to_csv(out_dir / "mimic_unit_ci.csv", index=False)


def check_gap30(mim_gaps, eic_gaps, out_dir):
    print("\n" + "=" * 78)
    print("CHECK 1: is n_gaps_gt30m measuring gapping or cadence regularity?")
    print("=" * 78)
    mg = mim_gaps["gap_min"].to_numpy(float)
    eg = eic_gaps["gap_min"].to_numpy(float)
    rows = []
    for name, a in (("MIMIC", mg), ("eICU", eg)):
        sub = a[a > 30]
        q = np.percentile(sub, [10, 25, 50, 75, 90]) if len(sub) else [np.nan] * 5
        near60 = float(((sub >= 55) & (sub <= 65)).mean()) if len(sub) else np.nan
        print(f"  {name}: gaps > 30 min, n={len(sub):,}")
        print(f"    p10 {q[0]:.0f}  p25 {q[1]:.0f}  median {q[2]:.0f}  "
              f"p75 {q[3]:.0f}  p90 {q[4]:.0f}")
        print(f"    share falling in 55-65 min: {near60:.3f}")
        rows.append({"cohort": name, "n_gaps_gt30": len(sub), "p10": q[0],
                     "p25": q[1], "median": q[2], "p75": q[3], "p90": q[4],
                     "share_55_65min": near60})
    pd.DataFrame(rows).to_csv(out_dir / "check_gap30.csv", index=False)
    print("\n  A high share near 60 min in MIMIC with a dispersed eICU "
          "distribution means the metric counts routine hourly intervals, not "
          "interruptions — report it as a cadence-regularity measure.")


def check_hospital_coverage(eic_nc, vp_cache, eicu_root, out_dir):
    print("\n" + "=" * 78)
    print("CHECK 2: hospitals present in vitalPeriodic but absent from "
          "nurseCharting")
    print("=" * 78)
    if not Path(vp_cache).exists():
        print(f"  vitalPeriodic cache not found at {vp_cache} — skipped")
        return
    vp = pd.read_parquet(vp_cache)
    pat = pd.read_csv(eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid",
                               "unitdischargeoffset"])
    vp = vp[["patientunitstayid"]].drop_duplicates().merge(
        pat, on="patientunitstayid", how="inner")
    vp = vp[vp["unitdischargeoffset"] >= WINDOW_MIN]
    vp_h = vp.groupby("hospitalid").size()
    nc_h = (eic_nc.assign(hid=eic_nc["hospital"].str.replace("eICU-", "",
                                                             regex=False)
                          .astype(int))
            .groupby("hid").size())
    big_vp = set(vp_h[vp_h >= 500].index)
    big_nc = set(nc_h[nc_h >= 500].index)
    only_vp = sorted(big_vp - big_nc)
    print(f"  hospitals >=500 stays in vitalPeriodic: {len(big_vp)}")
    print(f"  hospitals >=500 stays in nurseCharting: {len(big_nc)}")
    print(f"  in vitalPeriodic only: {len(only_vp)} -> {only_vp}")
    rows = []
    for h in only_vp:
        rows.append({"hospitalid": h, "vp_stays": int(vp_h.get(h, 0)),
                     "nc_stays": int(nc_h.get(h, 0)),
                     "nc_share": float(nc_h.get(h, 0)) / max(vp_h.get(h, 1), 1)})
    if rows:
        d = pd.DataFrame(rows).sort_values("vp_stays", ascending=False)
        print("\n" + d.to_string(index=False))
        d.to_csv(out_dir / "check_hospital_coverage.csv", index=False)
        zero = int((d["nc_stays"] == 0).sum())
        print(f"\n  {zero} of {len(d)} contribute NO nurse-charted heart rate "
              f"at all. If that is a site-level data-contribution difference "
              f"rather than a charting difference, the nurse-stream analysis "
              f"is on a selected set of institutions and must say so.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-cache", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-cache", required=True, type=Path)
    ap.add_argument("--eicu-vp-cache", type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./decomposition_ci"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mim, mim_gaps = load_mimic(args.mimic_cache, args.mimic_per_stay)
    eic, eic_gaps = load_eicu(args.eicu_cache, args.eicu_root)

    bootstrap_eicu(eic, args.out_dir)
    bootstrap_mimic_units(mim, args.out_dir)
    check_gap30(mim_gaps, eic_gaps, args.out_dir)
    if args.eicu_vp_cache:
        check_hospital_coverage(eic, args.eicu_vp_cache, args.eicu_root,
                                args.out_dir)
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
