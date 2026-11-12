#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Paper 17 — dispersion and denominator statistics requested at copyediting. v2.

v1 computed sections A, B(age) and C on every stay in the parquet caches. The
published figures are on the analytic sets, so those sections did not reproduce.
The archived outputs name the correct frames:

  stream_distributions.csv   MIMIC chartevents n=67,140  mean 27.9559 median 26
                             vitalPeriodic    n=130,294  mean 273.547 median 284
                             nurseCharting    n=119,317  mean 34.8627 median 29
  unit_profiles.csv          ten careunit rows summing to 67,140; CVICU
                             t_first_h 2.5719; per-unit median counts 23 to 27

v2 therefore builds:
  MIMIC analytic set    completed window (los >= 1 day) and >= 3 in-window records
  eICU eligible-stay    completed window and >= 3 in-window nurseCharting records
  eICU monitor-stream   completed window and >= 3 in-window vitalPeriodic records

and computes age as MIMIC-IV intends it, anchor_age + (admission year -
anchor_year), rather than anchor_age alone, which gave 65 against the published
67 and 66.

Sections D, E and F reproduced exactly in v1 and are unchanged here.

VERIFY BEFORE USE: each cohort size and each reproduced point estimate is
printed beside the published value. Where they differ, the dispersion beneath
is not comparable with the published median and must not be pasted in.

Usage:
  python paper17_copyedit_stats_v2.py \
      --mimic-cache  ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-vp-cache ~/bcst/unit_profile_eicu/vitalperiodic_offsets.parquet \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --mimic-root ~/physionet.org/files/mimiciv/3.1 \
      --out-dir ~/bcst/copyedit_stats_v2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_MIN = 24 * 60
WINDOW_H = 24.0
ROWS = []


def rec(section, label, value, expect=""):
    ROWS.append({"section": section, "label": label,
                 "value": value, "manuscript_says": expect})
    e = f"   [manuscript: {expect}]" if expect else ""
    print(f"    {label:<56s} {value}{e}")


def iqr(s):
    s = pd.Series(s).dropna()
    if not len(s):
        return "n/a"
    return f"{s.quantile(0.25):g}-{s.quantile(0.75):g}"


def med_iqr(s, dp=None):
    s = pd.Series(s).dropna()
    if not len(s):
        return "n/a"
    m = s.median()
    if dp is not None:
        return f"median {m:.{dp}f} (IQR {s.quantile(0.25):.{dp}f}-{s.quantile(0.75):.{dp}f})"
    return f"median {m:g} (IQR {iqr(s)})"


def mean_sd(s):
    s = pd.Series(s).dropna()
    if not len(s):
        return "n/a"
    return f"mean {s.mean():.4f} (SD {s.std(ddof=1):.4g})"


def per_stay_metrics(ev, key, off):
    """Identical to paper17_unit_documentation_profile.per_stay_metrics."""
    ev = ev.sort_values([key, off])
    g = ev.groupby(key)
    out = pd.DataFrame({"n_records": g.size(),
                        "t_first_h": g[off].min() / 60.0})
    diffs = ev.groupby(key)[off].diff()
    ev = ev.assign(gap_min=diffs)
    gg = ev.dropna(subset=["gap_min"]).groupby(key)["gap_min"]
    out["median_interval_min"] = gg.median()
    out["iqr_interval_min"] = gg.quantile(0.75) - gg.quantile(0.25)
    out["max_interval_min"] = gg.max()
    out["n_gaps_gt2h"] = gg.apply(lambda s: int((s > 120).sum()))
    out["frac_time_in_gaps"] = gg.apply(
        lambda s: float(s[s > 120].sum()) / (WINDOW_H * 60.0))
    return out.reset_index()


def in_window(df, off):
    return df[(df[off] >= 0) & (df[off] <= WINDOW_MIN)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-cache", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--eicu-vp-cache", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--mimic-root", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./copyedit_stats_v2"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("LOADING")
    print("=" * 78)
    mim = pd.read_parquet(a.mimic_cache)
    moff = "offset_min" if "offset_min" in mim.columns else mim.columns[-1]
    nc = pd.read_parquet(a.eicu_nc_cache)
    vp = pd.read_parquet(a.eicu_vp_cache)
    ps = pd.read_csv(a.mimic_per_stay)
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset", "age", "gender"])
    print(f"  MIMIC cache {len(mim):,} rows | nc {len(nc):,} | vp {len(vp):,}")
    print(f"  per-stay {len(ps):,} | eICU patient {len(pat):,}")

    # ------------------------------------------------- analytic frames ------
    print("\n" + "=" * 78)
    print("ANALYTIC FRAMES  (must reproduce before anything below is usable)")
    print("=" * 78)

    mim_w = in_window(mim, moff)
    m_all = per_stay_metrics(mim_w[["stay_id", moff]], "stay_id", moff)
    los = ps.set_index("stay_id")["los"] if "los" in ps.columns else None
    if los is None:
        raise SystemExit("per-stay table has no 'los' column")
    m_all["los"] = m_all["stay_id"].map(los)
    m_an = m_all[(m_all["los"] >= 1.0) & (m_all["n_records"] >= 3)]
    print(f"  MIMIC analytic set        : {len(m_an):,}"
          f"   [stream_distributions.csv: 67,140]")

    nc_w = in_window(nc, "observationoffset")
    n_all = per_stay_metrics(nc_w[["patientunitstayid", "observationoffset"]],
                             "patientunitstayid", "observationoffset")
    cw = pat[pat["unitdischargeoffset"] >= WINDOW_MIN]
    cwids = set(cw["patientunitstayid"])
    n_el = n_all[n_all["patientunitstayid"].isin(cwids)
                 & (n_all["n_records"] >= 3)]
    print(f"  eICU eligible-stay set    : {len(n_el):,}"
          f"   [stream_distributions.csv: 119,317]")

    vp_w = in_window(vp, "observationoffset")
    v_all = per_stay_metrics(vp_w[["patientunitstayid", "observationoffset"]],
                             "patientunitstayid", "observationoffset")
    v_el = v_all[v_all["patientunitstayid"].isin(cwids)
                 & (v_all["n_records"] >= 3)]
    print(f"  eICU monitor-stream set   : {len(v_el):,}"
          f"   [stream_distributions.csv: 130,294]")
    print(f"  eICU completed-window set : {len(cw):,}   [manuscript: 132,900]")

    # ------------------------------------------------ A. stream comparison --
    print("\n" + "=" * 78)
    print("A. STREAM COMPARISON  ('MIMIC-IV chartevents yielded a mean of ...')")
    print("=" * 78)
    for nm, d, em, emed in (
            ("MIMIC-IV chartevents", m_an, "27.9559 / 28.0", "26"),
            ("eICU nurseCharting", n_el, "34.8627 / 34.9", "29"),
            ("eICU vitalPeriodic", v_el, "273.547 / 273.5", "284")):
        rec("A", f"{nm}: records per 24 h", mean_sd(d["n_records"]), em)
        rec("A", f"{nm}: records per 24 h", med_iqr(d["n_records"]),
            f"median {emed}, IQR #-#")
        rec("A", f"{nm}: median interval (min)",
            med_iqr(d["median_interval_min"]), "median 60 / 60 / 5, IQR #-#")

    # ------------------------------------- B. MIMIC cohort construction -----
    print("\n" + "=" * 78)
    print("B. MIMIC-IV COHORT CONSTRUCTION  (removed vs retained)")
    print("=" * 78)
    if a.mimic_root:
        icu = pd.read_csv(a.mimic_root / "icu" / "icustays.csv.gz",
                          usecols=["subject_id", "hadm_id", "stay_id",
                                   "intime", "outtime", "los"])
        icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
        icu = icu.sort_values(["hadm_id", "intime"])
        icu["rk"] = icu.groupby("hadm_id")["intime"].rank(method="first")
        first = icu["rk"] == 1
        valid = icu["intime"].notna() & icu["outtime"].notna()
        removed, retained = icu[~first], icu[first & valid]
        print(f"  icustays {len(icu):,} | not first {int((~first).sum()):,} "
              f"| retained {len(retained):,}"
              f"   [94,458 / 9216 / 85,228]")

        pt = pd.read_csv(a.mimic_root / "hosp" / "patients.csv.gz",
                         usecols=["subject_id", "anchor_age", "anchor_year"])
        pt = pt.set_index("subject_id")
        for d in (retained, removed):
            d["anchor_age"] = d["subject_id"].map(pt["anchor_age"])
            d["anchor_year"] = d["subject_id"].map(pt["anchor_year"])
            d["age"] = d["anchor_age"] + (d["intime"].dt.year - d["anchor_year"])
        for nm, d in (("removed", removed), ("retained", retained)):
            rec("B", f"{nm} stays: age (years)", med_iqr(d["age"]),
                "removed 67 / retained 66")
            rec("B", f"{nm} stays: ICU length of stay (days)",
                med_iqr(d["los"], dp=2), "removed 2.56 / retained 1.92")
    else:
        print("  --mimic-root not supplied; section B skipped")

    # ------------------------------------------ C. MIMIC unit types ---------
    print("\n" + "=" * 78)
    print("C. MIMIC-IV UNIT TYPES  (on the 67,140 analytic set)")
    print("=" * 78)
    cu = ps[["stay_id", "careunit"]].dropna()
    m_u = m_an.merge(cu, on="stay_id", how="inner")
    print(f"  stays with a careunit: {len(m_u):,}   [unit_profiles.csv: 67,140]")
    print(f"  careunit values: {m_u['careunit'].nunique()} "
          f"(9 named plus OTHER)")
    per_unit = m_u.groupby("careunit").agg(
        n_stays=("n_records", "size"),
        med_count=("n_records", "median"), iqr_count=("n_records", iqr),
        med_interval=("median_interval_min", "median"),
        iqr_interval=("median_interval_min", iqr),
        med_max=("max_interval_min", "median"),
        iqr_max=("max_interval_min", iqr),
        med_gaps=("n_gaps_gt2h", "median"), iqr_gaps=("n_gaps_gt2h", iqr),
        med_tfirst=("t_first_h", "median"), iqr_tfirst=("t_first_h", iqr))
    print(per_unit.to_string())
    per_unit.to_csv(a.out_dir / "mimic_unit_dispersion.csv")
    named = per_unit.drop(index=[i for i in per_unit.index
                                 if str(i).strip().upper() == "OTHER"])
    print(f"\n  named-unit median count range: "
          f"{named['med_count'].min():g} to {named['med_count'].max():g}"
          f"   [manuscript: 23 to 27]")
    rec("C", "pooled analytic set: median charting interval (min)",
        med_iqr(m_u["median_interval_min"]), "every unit median 60")
    rec("C", "pooled analytic set: longest interval (min)",
        med_iqr(m_u["max_interval_min"]), "median 60")
    rec("C", "pooled analytic set: gaps beyond 2 hours",
        med_iqr(m_u["n_gaps_gt2h"]), "median 0")
    rec("C", "pooled analytic set: record count",
        med_iqr(m_u["n_records"]), "unit medians 23 to 27")
    for u, r in per_unit.iterrows():
        if "ardiac" in str(u):
            rec("C", f"{u}: hours to first record",
                f"median {r['med_tfirst']:.4f} (IQR {r['iqr_tfirst']})",
                "2.5719 / 2.57")

    # ----------------------------------- D. eICU cohorts and low-density ----
    print("\n" + "=" * 78)
    print("D. eICU-CRD COHORTS AND THE 8 LOW-DENSITY HOSPITALS")
    print("=" * 78)
    cnt = (nc_w.groupby("patientunitstayid").size()
                .rename("n_records").reset_index())
    cwc = cw.merge(cnt, on="patientunitstayid", how="left")
    cwc["n_records"] = cwc["n_records"].fillna(0).astype(int)
    elig = cwc[cwc["n_records"] >= 3]
    big = elig.groupby("hospitalid").size()
    big = big[big >= 500].index
    unres = elig[elig["hospitalid"].isin(big)]
    hmed = unres.groupby("hospitalid")["n_records"].median()
    low = hmed[hmed < 12].index
    restr = unres[~unres["hospitalid"].isin(low)]
    print(f"  eligible {len(elig):,}/{elig['hospitalid'].nunique()} | "
          f"unrestricted {len(unres):,}/{unres['hospitalid'].nunique()} | "
          f"restricted {len(restr):,}/{restr['hospitalid'].nunique()}"
          f"   [119,317/193 | 96,518/68 | 88,560/60]")
    print("\n  the 8 excluded hospitals:")
    for h in sorted(low, key=lambda x: hmed[x]):
        s = unres.loc[unres["hospitalid"] == h, "n_records"]
        b = cwc[cwc["hospitalid"] == h]
        nb = int((b["n_records"] < 3).sum())
        rec("D", f"hospital {h}: median count",
            f"{s.median():g} (IQR {iqr(s)}); below-3 share "
            f"{nb/len(b):.3%} ({nb:,}/{len(b):,})")
    print("\n  cohort below-3-record share, candidate frames "
          "(manuscript: median 0.5%):")
    for nm, ids in (("restricted 60 hospitals", set(restr['hospitalid'])),
                    ("unrestricted 68 hospitals", set(big)),
                    ("all 193 eligible-stay hospitals",
                     set(elig['hospitalid']))):
        sub = cwc[cwc["hospitalid"].isin(ids)]
        sh = sub.groupby("hospitalid").apply(
            lambda d: float((d["n_records"] < 3).mean()) * 100)
        rec("D", f"  {nm}", med_iqr(sh, dp=2), "median 0.5%")
    rec("D", "restricted cohort mean record count",
        mean_sd(restr["n_records"]), "39.01 (SD ##)")

    # -------------------------------------- E. hospital attributes ----------
    print("\n" + "=" * 78)
    print("E. HOSPITAL ATTRIBUTES  (bed capacity / teaching / region)")
    print("=" * 78)
    hf = a.eicu_root / "hospital.csv.gz"
    if hf.exists():
        hosp = pd.read_csv(hf)
        hosp.columns = [c.lower() for c in hosp.columns]
        rp = restr.merge(hosp, on="hospitalid", how="left").merge(
            n_all[["patientunitstayid", "median_interval_min"]],
            on="patientunitstayid", how="left")
        for attr in ("numbedscategory", "teachingstatus", "region"):
            if attr not in rp.columns:
                continue
            g = rp.groupby(attr).agg(
                n_hospitals=("hospitalid", "nunique"),
                n_stays=("n_records", "size"),
                med_count=("n_records", "median"),
                iqr_count=("n_records", iqr),
                med_interval=("median_interval_min", "median"),
                iqr_interval=("median_interval_min", iqr))
            print(f"\n  by {attr}:"); print(g.to_string())
            g.to_csv(a.out_dir / f"attribute_dispersion_{attr}.csv")
            for lv, r in g.iterrows():
                rec("E", f"{attr}={lv}",
                    f"count median {r['med_count']:g} (IQR {r['iqr_count']}); "
                    f"interval median {r['med_interval']:g} "
                    f"(IQR {r['iqr_interval']})")
    else:
        print(f"  hospital.csv.gz not found at {hf}")

    # ------------------------------------------------ F. denominators -------
    print("\n" + "=" * 78)
    print("F. DENOMINATORS FOR THE (n/N) PLACEHOLDERS")
    print("=" * 78)
    N = len(cwc)
    for lab, n, exp in (
            ("eICU three-record minimum", int((cwc["n_records"] < 3).sum()),
             "10.22% of 132,900"),
            ("  no nurse-charted record", int((cwc["n_records"] == 0).sum()),
             "8934 (6.72%)"),
            ("  one or two records",
             int(((cwc["n_records"] >= 1) & (cwc["n_records"] <= 2)).sum()),
             "4649 (3.50%)")):
        rec("F", lab, f"{n:,}/{N:,} = {n/N:.2%}", exp)
    for lab, d, exp in (("restricted", restr, "26 (43%) of 60"),
                        ("unrestricted", unres, "30 (44%) of 68"),
                        ("eligible-stay set", elig, "151 (78%) of 193")):
        k = int((d.groupby("hospitalid")["unittype"].nunique() == 1).sum())
        rec("F", f"single-unit-type hospitals, {lab}",
            f"{k}/{d['hospitalid'].nunique()}", exp)

    out = pd.DataFrame(ROWS)
    out.to_csv(a.out_dir / "copyedit_statistics_v2.csv", index=False)
    print("\n" + "=" * 78)
    print(f"{len(out)} statistics -> "
          f"{a.out_dir / 'copyedit_statistics_v2.csv'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
