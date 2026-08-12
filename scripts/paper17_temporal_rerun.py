#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "statsmodels", "pyarrow"]
# ///
"""
Paper 17 section 3.3, rerun on the clean exposure.

WHAT IS BEING RETESTED
----------------------
v9 reported that the odds of low vital-sign record density peaked for admissions
between 06:00 and 08:00 in MIMIC (adjusted OR 3.86, 95% CI 3.15-4.72, reference
18:00-20:00), with the Cardiac Vascular ICU as an a priori negative control
showing no effect (OR 1.01), and a replicated early-day peak in eICU (10:00-
12:00, OR 1.94; 70-87% of 108 hospitals peaking in the 00:00-14:00 window).

That analysis used per_stay.n_hr_24h, which reconstructs as itemids
220045 + 220046 + 220047: heart-rate observations PLUS the high and low alarm
limits. The alarm items are charted at shift boundaries — 53% of their records
fall in three clock hours (00, 08, 20) against 13% for true observations. A
cyclic finding built on a partly shift-anchored count needs retesting on the
observation stream alone.

TWO THREATS, TESTED SEPARATELY
------------------------------
1. COMPOSITION. Every model is fitted twice: once on the contaminated count and
   once on 220045 only. If the peak survives unchanged, composition was not
   driving it. The eICU analogue is vitalPeriodic versus nurseCharting, which is
   the stream-selection version of the same threat.

2. TRUNCATION. Each clock hour appears exactly once in a full 24 h window, so
   for stays completing the window the shift-anchored contribution is roughly
   constant regardless of admission hour. The route that survives is truncation:
   stays ending before 24 h span a variable number of shift-change rounds, and
   which ones depends on admission time. Every model is therefore also fitted
   restricted to stays completing the window.

Four fits per cohort: {contaminated, clean} x {all stays, completed window}.

Usage:
  python paper17_temporal_rerun.py \
      --mimic-cache ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-vp-cache ~/bcst/unit_profile_eicu/vitalperiodic_offsets.parquet \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/temporal_rerun
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

WINDOW_MIN = 24 * 60
REF_BIN = "18-20"
EARLY_WINDOW = range(0, 14)          # 00:00-14:00, as in v9
MIN_HOSP_STAYS = 500


def bin2h(h):
    lo = (h // 2) * 2
    return f"{lo:02d}-{lo + 2:02d}"


def counts_from_cache(path, key, offset_col):
    ev = pd.read_parquet(path)
    return (ev.groupby(key).size().rename("n_clean").reset_index())


def fit_hourly(df, outcome, covars, label, cluster=None):
    """Logistic model of low density on admission-hour bin."""
    f = f"{outcome} ~ C(adm_bin, Treatment(reference='{REF_BIN}')) + {covars}"
    try:
        if cluster is not None:
            m = smf.logit(f, data=df).fit(disp=0, maxiter=200,
                                          cov_type="cluster",
                                          cov_kwds={"groups": df[cluster]})
        else:
            m = smf.logit(f, data=df).fit(disp=0, maxiter=200, method="bfgs")
    except Exception as e:
        print(f"  {label}: FAILED ({e})")
        return None
    terms = [t for t in m.params.index if "adm_bin" in t]
    rows = []
    for t in terms:
        b = t.split("T.")[-1].rstrip("]")
        ci = m.conf_int().loc[t]
        rows.append({"bin": b, "OR": float(np.exp(m.params[t])),
                     "lo": float(np.exp(ci[0])), "hi": float(np.exp(ci[1])),
                     "p": float(m.pvalues[t])})
    r = pd.DataFrame(rows).sort_values("OR", ascending=False)
    top = r.iloc[0]
    print(f"  {label:38s} n={len(df):>7,}  peak {top['bin']}  "
          f"OR {top['OR']:.2f} [{top['lo']:.2f}, {top['hi']:.2f}]  "
          f"p={top['p']:.1g}")
    r["model"] = label
    return r


def run_cohort(df, name, covars, out_dir, cluster=None):
    print("\n" + "=" * 78)
    print(f"{name}: ODDS OF LOW RECORD DENSITY BY ADMISSION HOUR")
    print("=" * 78)
    allr = []
    for exp_label, col in (("contaminated", "n_contaminated"),
                           ("clean", "n_clean")):
        for restr_label, sub in (("all stays", df),
                                 ("completed 24h", df[df["completed"]])):
            s = sub.dropna(subset=[col]).copy()
            s["low"] = (s[col] <= s[col].quantile(0.10)).astype(int)
            r = fit_hourly(s, "low", covars,
                           f"{exp_label} / {restr_label}", cluster)
            if r is not None:
                r["exposure"], r["restriction"] = exp_label, restr_label
                allr.append(r)
    if allr:
        out = pd.concat(allr, ignore_index=True)
        out.to_csv(out_dir / f"{name.lower()}_hourly_or.csv", index=False)
        return out
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-cache", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--eicu-vp-cache", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./temporal_rerun"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ================= MIMIC =================
    ps = pd.read_csv(args.mimic_per_stay)
    clean = counts_from_cache(args.mimic_cache, "stay_id", "offset_min")
    m = ps.merge(clean, on="stay_id", how="left")
    m = m.rename(columns={"n_hr_24h": "n_contaminated"})
    m["adm_bin"] = m["adm_hour"].apply(bin2h)
    m["completed"] = m["los"] >= 1.0
    m = m.dropna(subset=["n_contaminated", "n_clean", "adm_bin"])
    print(f"MIMIC stays: {len(m):,}  "
          f"(completed 24h: {int(m['completed'].sum()):,})")

    mcov = ("age_z + C(gender) + n_comorbid_z + n_chapters_z "
            "+ C(anchor_year_group) + C(careunit)")
    mres = run_cohort(m, "MIMIC", mcov, args.out_dir)

    # negative control: per care unit, clean exposure, completed window
    print("\n  NEGATIVE CONTROL — per care unit (clean exposure, completed "
          "window)")
    sub = m[m["completed"]].copy()
    sub["low"] = (sub["n_clean"] <= sub["n_clean"].quantile(0.10)).astype(int)
    rows = []
    for unit, d in sub.groupby("careunit"):
        if len(d) < 800 or d["low"].sum() < 40 or d["adm_bin"].nunique() < 8:
            continue
        r = fit_hourly(d, "low", "age_z + C(gender) + n_comorbid_z",
                       f"    {unit[:34]}")
        if r is not None:
            rows.append(r.assign(careunit=unit))
    if rows:
        pd.concat(rows).to_csv(args.out_dir / "mimic_by_careunit.csv",
                               index=False)

    # ================= eICU =================
    pat = pd.read_csv(args.eicu_root / "patient.csv.gz")
    need = ["patientunitstayid", "hospitalid", "unittype",
            "unitdischargeoffset", "age", "gender"]
    if "unitadmittime24" not in pat.columns:
        print("\neICU: unitadmittime24 absent — clock-hour analysis not "
              "possible, eICU section skipped")
        return
    pat = pat[need + ["unitadmittime24"]]

    nc = counts_from_cache(args.eicu_nc_cache, "patientunitstayid",
                           "observationoffset")
    vp = (pd.read_parquet(args.eicu_vp_cache)
            .groupby("patientunitstayid").size().rename("n_contaminated")
            .reset_index())
    e = pat.merge(nc, on="patientunitstayid", how="inner") \
           .merge(vp, on="patientunitstayid", how="inner")

    hh = pd.to_datetime(e["unitadmittime24"], format="%H:%M:%S",
                        errors="coerce").dt.hour
    e["adm_hour"] = hh
    e = e.dropna(subset=["adm_hour"])
    e["adm_bin"] = e["adm_hour"].astype(int).apply(bin2h)
    e["completed"] = e["unitdischargeoffset"] >= WINDOW_MIN
    e["age_num"] = pd.to_numeric(e["age"].replace("> 89", "90"),
                                 errors="coerce")
    e["age_z"] = (e["age_num"] - e["age_num"].mean()) / e["age_num"].std()
    e = e.dropna(subset=["age_z", "gender"])
    print(f"\neICU stays with clock time: {len(e):,}  "
          f"(completed 24h: {int(e['completed'].sum()):,})")
    print("  note: 'contaminated' here is the vitalPeriodic monitor stream, "
          "'clean' is nurseCharting — the stream-selection version of the "
          "same threat")

    ecov = "age_z + C(gender) + C(unittype)"
    eres = run_cohort(e, "eICU", ecov, args.out_dir, cluster="hospitalid")

    # per-hospital peak window, clean exposure, completed stays
    print("\n  PER-HOSPITAL PEAK WINDOW (clean exposure, completed window)")
    s = e[e["completed"]].copy()
    s["low"] = (s["n_clean"] <= s["n_clean"].quantile(0.10)).astype(int)
    counts = s["hospitalid"].value_counts()
    big = counts[counts >= MIN_HOSP_STAYS].index
    peaks = []
    for h, d in s[s["hospitalid"].isin(big)].groupby("hospitalid"):
        rate = d.groupby("adm_bin")["low"].mean()
        if rate.empty:
            continue
        peak_bin = rate.idxmax()
        peaks.append({"hospitalid": h, "n_stays": len(d),
                      "peak_bin": peak_bin,
                      "peak_hour": int(peak_bin.split("-")[0]),
                      "peak_rate": float(rate.max())})
    pk = pd.DataFrame(peaks)
    if len(pk):
        share = float(pk["peak_hour"].isin(EARLY_WINDOW).mean())
        print(f"    hospitals with >= {MIN_HOSP_STAYS} stays: {len(pk)}")
        print(f"    peak low-density admission in 00:00-14:00: {share:.3f}")
        print(f"    v9 reported 70-87% of 108 hospitals on the contaminated "
              f"count")
        pk.to_csv(args.out_dir / "eicu_hospital_peaks.csv", index=False)

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("Compare the four MIMIC rows. If the peak bin and OR are stable "
          "across contaminated/clean, composition was not driving the cyclic "
          "finding and section 3.3 survives on the observation stream.")
    print("If the OR falls sharply under 'clean / completed 24h', the v9 "
          "result was partly an artefact of counting shift-anchored alarm "
          "records in truncated windows.")
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
