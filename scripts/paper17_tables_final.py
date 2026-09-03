#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "statsmodels", "pyarrow"]
# ///
"""
Paper 17 revision: every cell of Tables 3 and 5, both cohorts.

Earlier stages covered six metrics; Table 3 has eight rows, and the variance
partition column was computed for five. Rather than assemble the table from
several partial runs, this recomputes every cell in one place so the table is
internally consistent and each number has one provenance.

TABLE 3, per metric
  pooled sequential eta-squared: database, hospital increment, unit increment
  eICU-CRD hospital eta-squared with a hospital-clustered bootstrap interval
  hospital variance partition coefficient from a three-component mixed model

TABLE 5, per variable
  hospital and unit eta-squared for record count and median interval, on each
  variable's own plausibility floor

Both are produced for the unrestricted and the restricted cohort. Output is
printed as table-ready rows.

Usage:
  python paper17_tables_final.py \
      --mimic-cache ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --cross-dir ~/bcst/cross_vitals \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/revision_tables
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

WINDOW_MIN = 24 * 60
MIN_RECORDS = 3
MIN_HOSP = 500
FLOOR = 12
N_BOOT = 500
SEED = 17

ROWS = [("n_records", "Record count"),
        ("t_first_h", "Hours to first record"),
        ("median_interval_min", "Median interval"),
        ("iqr_interval_min", "Interval IQR"),
        ("max_interval_min", "Longest interval"),
        ("n_gaps_gt30m", "Gaps >30 min"),
        ("n_gaps_gt2h", "Gaps >2 h"),
        ("frac_time_in_gaps", "Fraction of window in gaps")]


def eta2(y, g):
    y = np.asarray(y, float); g = np.asarray(g)
    ok = ~np.isnan(y); y, g = y[ok], g[ok]
    if len(y) < 10:
        return np.nan
    grand = y.mean(); sst = ((y - grand) ** 2).sum()
    if sst <= 0:
        return np.nan
    d = pd.DataFrame({"y": y, "g": g}).groupby("g")["y"].agg(["mean", "size"])
    return float((d["size"] * (d["mean"] - grand) ** 2).sum() / sst)


def metrics(ev, key, off):
    ev = ev.sort_values([key, off])
    g = ev.groupby(key)[off]
    out = pd.DataFrame({"n_records": g.size(), "t_first_h": g.min() / 60.0})
    ev = ev.assign(gap=g.diff())
    gg = ev.dropna(subset=["gap"]).groupby(key)["gap"]
    out["median_interval_min"] = gg.median()
    out["iqr_interval_min"] = gg.quantile(0.75) - gg.quantile(0.25)
    out["max_interval_min"] = gg.max()
    out["n_gaps_gt30m"] = gg.apply(lambda s: int((s > 30).sum()))
    out["n_gaps_gt2h"] = gg.apply(lambda s: int((s > 120).sum()))
    out["frac_time_in_gaps"] = gg.apply(
        lambda s: float(s[s > 120].sum()) / WINDOW_MIN)
    return out.reset_index()


def load(a):
    mim = metrics(pd.read_parquet(a.mimic_cache)[["stay_id", "offset_min"]],
                  "stay_id", "offset_min")
    ps = pd.read_csv(a.mimic_per_stay, usecols=["stay_id", "careunit", "los"])
    mim = mim.merge(ps, on="stay_id", how="inner")
    mim = mim[(mim["los"] >= 1.0) & (mim["n_records"] >= MIN_RECORDS)].copy()
    mim["database"], mim["hospital"] = "MIMIC", "MIMIC-BIDMC"
    mim["unit_id"] = "MIMIC:" + mim["careunit"]

    e = metrics(pd.read_parquet(a.eicu_nc_cache), "patientunitstayid",
                "observationoffset")
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    e = e.merge(pat, on="patientunitstayid", how="inner")
    e = e[(e["unitdischargeoffset"] >= WINDOW_MIN)
          & (e["n_records"] >= MIN_RECORDS)].copy()
    e["database"] = "eICU"
    e["hospital"] = "eICU-" + e["hospitalid"].astype(str)
    e["unit_id"] = e["hospital"] + ":" + e["unittype"].astype(str)
    return mim, e


def big(e):
    c = e["hospitalid"].value_counts()
    return e[e["hospitalid"].isin(c[c >= MIN_HOSP].index)]


def apply_floor(e, floor):
    med = e.groupby("hospitalid")["n_records"].median()
    return e[e["hospitalid"].isin(med[med >= floor].index)].copy()


def table3(mim, e, label, out_dir, rows):
    hosps = e["hospital"].unique()
    idx = {h: g.index.values for h, g in e.groupby("hospital")}
    rng = np.random.default_rng(SEED)
    cols = [c for c, _ in ROWS] + ["database", "hospital", "unit_id"]
    pool = pd.concat([mim[cols], e[cols]], ignore_index=True)

    print(f"\n  {label}: pooled {len(pool):,} stays, "
          f"{pool['hospital'].nunique()} hospitals, "
          f"{pool['unit_id'].nunique()} unit cells | eICU-CRD {len(e):,} stays, "
          f"{len(hosps)} hospitals")
    print(f"    {'Metric':28s} {'Database':>9s} {'+Hosp':>7s} {'+Unit':>7s} "
          f"{'eICU hospital eta2 (95% CI)':>28s} {'VPC':>6s}")
    print("    " + "-" * 92)
    for col, name in ROWS:
        sub = pool.dropna(subset=[col])
        r_db = eta2(sub[col], sub["database"])
        r_h = eta2(sub[col], sub["hospital"])
        r_u = eta2(sub[col], sub["unit_id"])

        y = e[col].to_numpy(float)
        pt = eta2(y, e["hospital"].to_numpy())
        b = []
        for _ in range(N_BOOT):
            pick = rng.choice(hosps, len(hosps), replace=True)
            r = np.concatenate([idx[h] for h in pick])
            tag = np.concatenate([np.full(len(idx[h]), i)
                                  for i, h in enumerate(pick)]).astype(str)
            b.append(eta2(e.loc[r, col].to_numpy(float), tag))
        lo, hi = np.nanpercentile(b, [2.5, 97.5])

        s = e.dropna(subset=[col]).copy()
        s["yz"] = (s[col] - s[col].mean()) / s[col].std()
        try:
            f = smf.mixedlm("yz ~ 1", data=s, groups=s["hospital"],
                            re_formula="1",
                            vc_formula={"unit": "0 + C(unit_id)"}).fit(reml=True)
            vh = float(f.cov_re.iloc[0, 0])
            vu = float(f.vcomp[0]) if len(f.vcomp) else 0.0
            vpc = vh / (vh + vu + float(f.scale))
        except Exception:
            vpc = np.nan
        ci = f"{pt:.3f} ({lo:.3f}-{hi:.3f})"
        print(f"    {name:28s} {r_db:9.3f} {r_h - r_db:7.3f} {r_u - r_h:7.3f} "
              f"{ci:>28s} {vpc:6.3f}")
        rows.append({"cohort": label, "metric": name, "database": r_db,
                     "hospital_increment": r_h - r_db,
                     "unit_increment": r_u - r_h, "eicu_eta2": pt,
                     "eta2_lo": lo, "eta2_hi": hi, "vpc_hospital": vpc})


def table5(a, out_dir, rows):
    print("\n" + "=" * 92)
    print("TABLE 5 — replication, each variable on its own plausibility floor")
    print("=" * 92)
    nc = pd.read_parquet(a.cross_dir / "eicu_nc.parquet")
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    src = {"heart_rate": pd.read_parquet(a.eicu_nc_cache)}
    for v in sorted(nc["variable"].unique()):
        src[v] = nc[nc["variable"] == v][["patientunitstayid", "offset_min"]] \
            .rename(columns={"offset_min": "observationoffset"})
    print(f"\n    {'variable':16s} {'cohort':14s} {'hosp':>5s} "
          f"{'count h/u':>16s} {'interval h/u':>16s}")
    print("    " + "-" * 72)
    for var, ev in src.items():
        d = metrics(ev, "patientunitstayid", "observationoffset").merge(
            pat, on="patientunitstayid", how="inner")
        d = d[(d["unitdischargeoffset"] >= WINDOW_MIN)
              & (d["n_records"] >= MIN_RECORDS)].copy()
        d["hospital"] = "eICU-" + d["hospitalid"].astype(str)
        d["unit_id"] = d["hospital"] + ":" + d["unittype"].astype(str)
        d = big(d)
        for lab, s in (("unrestricted", d), ("own floor", apply_floor(d, FLOOR))):
            hc = eta2(s["n_records"], s["hospital"])
            uc = eta2(s["n_records"], s["unit_id"]) - hc
            hi_ = eta2(s["median_interval_min"], s["hospital"])
            ui = eta2(s["median_interval_min"], s["unit_id"]) - hi_
            print(f"    {var:16s} {lab:14s} {s['hospitalid'].nunique():5d} "
                  f"{f'{hc:.3f} / {uc:.3f}':>16s} {f'{hi_:.3f} / {ui:.3f}':>16s}")
            rows.append({"variable": var, "cohort": lab,
                         "n_hospitals": int(s["hospitalid"].nunique()),
                         "n_stays": len(s), "eta2_hosp_count": hc,
                         "eta2_unit_count": uc, "eta2_hosp_interval": hi_,
                         "eta2_unit_interval": ui})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-cache", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--cross-dir", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./revision_tables"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    mim, e = load(a)
    eb = big(e)
    er = apply_floor(eb, FLOOR)
    print("=" * 92)
    print("TABLE 3 — every cell, both cohorts")
    print("=" * 92)
    r3 = []
    table3(mim, eb, "unrestricted", a.out_dir, r3)
    table3(mim, er, "restricted", a.out_dir, r3)
    pd.DataFrame(r3).to_csv(a.out_dir / "table3_cells.csv", index=False)

    r5 = []
    table5(a, a.out_dir, r5)
    pd.DataFrame(r5).to_csv(a.out_dir / "table5_cells.csv", index=False)
    print(f"\n-> {a.out_dir}")


if __name__ == "__main__":
    main()
