#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "statsmodels", "pyarrow"]
# ///
"""
Paper 17 revision, Stage 6: the two tables the restriction has not yet reached.

TABLE 5 — cross-vital replication
----------------------------------
The oxygen saturation and respiratory rate replication used the unrestricted
eICU-CRD cohort. If the heart-rate analysis is now reported on a
plausibility-restricted cohort, Table 5 is inconsistent unless it is recomputed
on the same basis.

The floor is applied per variable, using that variable's own median count,
because a hospital may contribute one nurse-charted variable and not another.
The alternative — applying the heart-rate floor to all three — would import a
heart-rate judgement into the other two. Both are computed so the choice is
visible.

TABLE 3 POOLED COLUMNS — MIMIC-IV consistency
----------------------------------------------
The pooled decomposition combines both cohorts. MIMIC-IV needs no plausibility
restriction: 0.12% of stays fail the record minimum and every care unit sits at
a 60-minute median interval. That has to be shown rather than asserted, so the
MIMIC-IV per-unit medians are printed against the same floor used for
eICU-CRD hospitals. The pooled decomposition is then recomputed with the
restricted eICU-CRD set, alongside the published version.

Usage:
  python paper17_stage6_tables.py \
      --mimic-cache ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-vp-cache ~/bcst/unit_profile_eicu/vitalperiodic_offsets.parquet \
      --cross-dir ~/bcst/cross_vitals \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/revision_stage6
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WINDOW_MIN = 24 * 60
MIN_RECORDS = 3
MIN_HOSP = 500
FLOOR = 12


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
    out = pd.DataFrame({"n_records": g.size()})
    ev = ev.assign(gap=g.diff())
    gg = ev.dropna(subset=["gap"]).groupby(key)["gap"]
    out["median_interval_min"] = gg.median()
    out["max_interval_min"] = gg.max()
    out["frac_time_in_gaps"] = gg.apply(
        lambda s: float(s[s > 120].sum()) / WINDOW_MIN)
    return out.reset_index()


# --------------------------------------------------- MIMIC-IV consistency --

def mimic_check(a):
    print("=" * 78)
    print("MIMIC-IV — does the plausibility floor exclude anything?")
    print("=" * 78)
    m = metrics(pd.read_parquet(a.mimic_cache)[["stay_id", "offset_min"]],
                "stay_id", "offset_min")
    ps = pd.read_csv(a.mimic_per_stay, usecols=["stay_id", "careunit", "los"])
    m = m.merge(ps, on="stay_id", how="inner")
    m = m[(m["los"] >= 1.0) & (m["n_records"] >= MIN_RECORDS)].copy()
    prof = m.groupby("careunit").agg(
        n_stays=("n_records", "size"),
        median_count=("n_records", "median"),
        median_interval=("median_interval_min", "median"))
    prof["meets_floor"] = prof["median_count"] >= FLOOR
    print(prof.round(2).to_string())
    n_fail = int((~prof["meets_floor"]).sum())
    print(f"\n  care units below a median of {FLOOR} records: {n_fail} of "
          f"{len(prof)}")
    print("  The floor is defined for hospitals; MIMIC-IV is a single")
    print("  institution, so it is applied here to care units as the closest")
    print("  analogue. No unit falls below it, so no MIMIC-IV stay is excluded")
    print("  and the cohort is identical under both the published and the")
    print("  restricted analysis.")
    m["database"], m["hospital"] = "MIMIC", "MIMIC-BIDMC"
    m["unit_id"] = "MIMIC:" + m["careunit"]
    return m


# ------------------------------------------------------ pooled Table 3 -----

def pooled(mim, a, out_dir):
    print("\n" + "=" * 78)
    print("TABLE 3 POOLED COLUMNS — published basis vs restricted eICU-CRD")
    print("=" * 78)
    d = metrics(pd.read_parquet(a.eicu_nc_cache), "patientunitstayid",
                "observationoffset")
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    d = d.merge(pat, on="patientunitstayid", how="inner")
    d = d[(d["unitdischargeoffset"] >= WINDOW_MIN)
          & (d["n_records"] >= MIN_RECORDS)].copy()
    d["database"] = "eICU"
    d["hospital"] = "eICU-" + d["hospitalid"].astype(str)
    d["unit_id"] = d["hospital"] + ":" + d["unittype"].astype(str)

    cnt = d["hospitalid"].value_counts()
    big = d[d["hospitalid"].isin(cnt[cnt >= MIN_HOSP].index)]
    med = big.groupby("hospitalid")["n_records"].median()
    keep = med[med >= FLOOR].index
    d_res = d[d["hospitalid"].isin(keep)]

    METRICS = ["n_records", "median_interval_min", "max_interval_min",
               "frac_time_in_gaps"]
    LABEL = {"n_records": "Record count",
             "median_interval_min": "Median interval",
             "max_interval_min": "Longest interval",
             "frac_time_in_gaps": "Fraction in gaps"}
    cols = METRICS + ["database", "hospital", "unit_id"]
    rows = []
    for lab, e in (("published (all eICU-CRD)", d),
                   ("restricted eICU-CRD", d_res)):
        pool = pd.concat([mim[cols], e[cols]], ignore_index=True)
        print(f"\n  {lab}: {len(pool):,} stays, "
              f"{pool['hospital'].nunique()} hospitals, "
              f"{pool['unit_id'].nunique()} unit cells")
        print(f"    {'metric':22s} {'database':>10s} {'+hospital':>11s} "
              f"{'+unit':>8s}")
        print("    " + "-" * 56)
        for m in METRICS:
            sub = pool.dropna(subset=[m])
            r_db = eta2(sub[m], sub["database"])
            r_h = eta2(sub[m], sub["hospital"])
            r_u = eta2(sub[m], sub["unit_id"])
            print(f"    {LABEL[m]:22s} {r_db:10.3f} {r_h - r_db:11.3f} "
                  f"{r_u - r_h:8.3f}")
            rows.append({"basis": lab, "metric": LABEL[m], "database": r_db,
                         "hospital_increment": r_h - r_db,
                         "unit_increment": r_u - r_h, "total": r_u,
                         "n_stays": len(pool)})
    pd.DataFrame(rows).to_csv(out_dir / "pooled_decomposition_bases.csv",
                              index=False)
    print("\n  These columns are sequential and therefore order-dependent")
    print("  (editorial comment 18). They are reported as a descriptive")
    print("  summary under a stated ordering, not as an attribution.")


# ------------------------------------------------------------- Table 5 -----

def table5(a, out_dir):
    print("\n" + "=" * 78)
    print("TABLE 5 — cross-vital replication on the restricted basis")
    print("=" * 78)
    nc = pd.read_parquet(a.cross_dir / "eicu_nc.parquet")
    vp = pd.read_parquet(a.cross_dir / "eicu_vp.parquet")
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    hr_nc = pd.read_parquet(a.eicu_nc_cache)

    # the heart-rate floor, for the alternative basis
    hr = metrics(hr_nc, "patientunitstayid", "observationoffset").merge(
        pat, on="patientunitstayid", how="inner")
    hr = hr[(hr["unitdischargeoffset"] >= WINDOW_MIN)
            & (hr["n_records"] >= MIN_RECORDS)]
    c = hr["hospitalid"].value_counts()
    hr = hr[hr["hospitalid"].isin(c[c >= MIN_HOSP].index)]
    hr_keep = set(hr.groupby("hospitalid")["n_records"].median()
                    .loc[lambda s: s >= FLOOR].index)

    rows = []
    for var in ["heart_rate"] + sorted(nc["variable"].unique()):
        if var == "heart_rate":
            e = metrics(hr_nc, "patientunitstayid", "observationoffset")
            v = (pd.read_parquet(a.eicu_vp_cache)
                   .groupby("patientunitstayid").size().rename("n_vp")
                   .reset_index())
        else:
            e = metrics(nc[nc["variable"] == var][["patientunitstayid",
                                                   "offset_min"]],
                        "patientunitstayid", "offset_min")
            v = (vp[vp["variable"] == var].groupby("patientunitstayid").size()
                   .rename("n_vp").reset_index())
        e = e.merge(pat, on="patientunitstayid", how="inner")
        e = e[(e["unitdischargeoffset"] >= WINDOW_MIN)
              & (e["n_records"] >= MIN_RECORDS)].copy()
        e["hospital"] = "eICU-" + e["hospitalid"].astype(str)
        e["unit_id"] = e["hospital"] + ":" + e["unittype"].astype(str)
        cc = e["hospitalid"].value_counts()
        e = e[e["hospitalid"].isin(cc[cc >= MIN_HOSP].index)]
        own_keep = set(e.groupby("hospitalid")["n_records"].median()
                        .loc[lambda s: s >= FLOOR].index)

        print(f"\n  {var}")
        for lab, keep in (("unrestricted", set(e["hospitalid"])),
                          ("own floor", own_keep),
                          ("heart-rate floor", hr_keep)):
            s = e[e["hospitalid"].isin(keep)]
            if s["hospitalid"].nunique() < 10:
                continue
            eh_c = eta2(s["n_records"], s["hospital"])
            eu_c = eta2(s["n_records"], s["unit_id"]) - eh_c
            eh_i = eta2(s["median_interval_min"], s["hospital"])
            eu_i = eta2(s["median_interval_min"], s["unit_id"]) - eh_i
            j = s[["patientunitstayid", "n_records"]].merge(
                v, on="patientunitstayid", how="inner")
            lo_v = j["n_vp"] <= j["n_vp"].quantile(.10)
            lo_n = j["n_records"] <= j["n_records"].quantile(.10)
            obs = int((lo_v & lo_n).sum()) / max(int(lo_v.sum()), 1)
            chance = float(lo_n.mean())
            print(f"    {lab:18s} {s['hospitalid'].nunique():3d} hosp  "
                  f"count {eh_c:.3f}/{eu_c:.3f}   interval {eh_i:.3f}/{eu_i:.3f}"
                  f"   stream ratio {obs / chance:.2f}")
            rows.append({"variable": var, "basis": lab,
                         "n_hospitals": int(s["hospitalid"].nunique()),
                         "n_stays": len(s), "eta2_hosp_count": eh_c,
                         "eta2_unit_count": eu_c,
                         "eta2_hosp_interval": eh_i,
                         "eta2_unit_interval": eu_i,
                         "stream_ratio": obs / chance})
    pd.DataFrame(rows).to_csv(out_dir / "table5_bases.csv", index=False)
    print("\n  'own floor' applies each variable's own median-count floor;")
    print("  'heart-rate floor' applies the hospitals retained for heart rate.")
    print("  Report one basis consistently and state which.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-cache", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--eicu-vp-cache", required=True, type=Path)
    ap.add_argument("--cross-dir", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./revision_stage6"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    mim = mimic_check(a)
    pooled(mim, a, a.out_dir)
    table5(a, a.out_dir)
    print(f"\n-> {a.out_dir}")


if __name__ == "__main__":
    main()
