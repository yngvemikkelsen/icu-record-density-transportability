#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "statsmodels"]
# ///
"""
Conditional residualisation analysis, v2 (Paper 17).

CHANGES FROM v1 (bcst_residualization.py)
-----------------------------------------
1. HARMONISED RESIDUALISATION.  v1 fitted
       MIMIC: log_n_hr ~ hr_min + hr_max + map_min + map_max + gcs_min
                         + mech_vent + pre_icu_los + elective + age + sex
       eICU : log_n_hr ~ log_apache + age + sex
   Two problems.  (a) hr_min/hr_max/map_min/map_max are order statistics of the
   same records being counted, so log_n_hr was partly regressed on its own
   sample size (extreme-value effect); (b) the two models were not comparable,
   so the R^2 contrast (0.133 vs 0.032) was not a like-for-like measure of
   acuity coupling.  pre_icu_los_hours and elective are administrative, not
   acuity, and inflate the MIMIC side further with no eICU analogue.
   v2 fits the SAME model in both cohorts, from each cohort's own vital stream,
   using median and IQR (consistent estimators, not driven by record count).
   The legacy models are still fitted, side by side, to quantify how much of
   the original contrast was artifact.

2. MATCHED OUTCOME.  Hospital mortality is primary in BOTH cohorts, for both
   the continuous and the bottom-decile binary specification.  MIMIC 30-day is
   retained as secondary.  v1 ran the MIMIC binary on 30-day only.

3. CORRECTED DIRECTION LOGIC.  The two exposures face opposite ways: the binary
   flags the LOW tail (harm => OR > 1), the continuous is per SD INCREASE
   (harm => OR < 1).  v1's verdict text compared them as if they faced the same
   way.  v2 maps both onto an explicit harm flag before comparing.

4. NO SILENT FALLBACKS.  v1 set mech_vent_24h = 0 when the file was absent,
   putting a constant column into the model.  v2 raises instead.

Usage:
  python bcst_residualization_v2.py \
      --mimic-root      /Users/yngve/physionet.org/files/mimiciv/3.1 \
      --eicu-root       /Users/yngve/physionet.org/files/eicu-crd/2.0 \
      --mimic-per-stay  ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-per-stay   ~/bcst/eicu_replication_results/per_stay_eicu.csv \
      --mimic-physio    ~/bcst/mimic_oasis_adjustment_results/physiology_24h.csv \
      --mimic-vent      ~/bcst/mimic_oasis_adjustment_results/mech_vent_24h.csv \
      --physio-v2-dir   ~/bcst/physiology_v2 \
      --out-dir         ~/bcst/residualization_v2

Requires paper17_build_physiology_v2.py to have been run first.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ADMISSION_SOURCE_MAP = {
    "EMERGENCY ROOM": "ED",
    "TRANSFER FROM HOSPITAL": "Transfer-Out",
    "TRANSFER FROM SKILLED NURSING FACILITY": "Transfer-Out",
    "TRANSFER FROM OTHER HEALTHCARE FACILITY": "Transfer-Out",
    "PHYSICIAN REFERRAL": "Direct",
    "PHYSICIAN REFERRAL/NORMAL DELIVERY": "Direct",
    "WALK-IN/SELF REFERRAL": "Direct",
    "CLINIC REFERRAL": "Direct",
    "CLINIC REFERRAL/PREMATURE": "Direct",
    "PHYSICIAN REFERRAL/PREMATURE BIRTH": "Direct",
    "PROCEDURE SITE": "Post-Procedure",
    "AMBULATORY SURGERY TRANSFER": "Post-Procedure",
    "PACU": "Post-Procedure",
    "INTERNAL TRANSFER TO OR FROM PSYCH": "Internal-Transfer",
    "INFORMATION NOT AVAILABLE": "Unknown",
}

# Identical in both cohorts. No order statistics, no administrative terms.
HARMONISED_ACUITY = ("hr_median_24h_z + hr_iqr_24h_z + map_median_24h_z "
                     "+ map_iqr_24h_z + age_z + C(gender)")

LEGACY_MIMIC_ACUITY = ("hr_min_24h_z + hr_max_24h_z + map_min_24h_z "
                       "+ map_max_24h_z + gcs_min_24h_z + mech_vent_24h "
                       "+ pre_icu_los_hours_z + elective + age_z + C(gender)")
LEGACY_EICU_ACUITY = "log_apache + age_z + C(gender)"

PCTS = [5, 10, 15, 20, 25]


def recode_source(loc):
    if pd.isna(loc):
        return "Unknown"
    return ADMISSION_SOURCE_MAP.get(str(loc).strip(), "Other")


def standardise(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else s * 0


def require(path: Path, what: str) -> Path:
    if path.exists():
        return path
    alt = path.with_suffix("")
    if alt.exists():
        return alt
    raise FileNotFoundError(
        f"{what} not found at {path}. v2 does not substitute a default — "
        f"a missing file silently changes the model specification."
    )


def get_effect(fit, term):
    if term not in fit.params.index:
        return None
    ci = fit.conf_int().loc[term]
    return {"OR": float(np.exp(fit.params[term])),
            "OR_lo": float(np.exp(ci[0])),
            "OR_hi": float(np.exp(ci[1])),
            "p": float(fit.pvalues[term])}


def fmt(e):
    if e is None:
        return "NA"
    return f"OR {e['OR']:.3f} [{e['OR_lo']:.3f}, {e['OR_hi']:.3f}], p={e['p']:.2g}"


def ensure_columns(df, needed, source_path, key_col, label):
    missing = [c for c in needed if c not in df.columns]
    if not missing:
        return df
    print(f"  [{label}] loading missing columns: {missing}")
    src = pd.read_csv(require(source_path, label), usecols=[key_col] + missing)
    return df.merge(src, on=key_col, how="left")


def fit_logit(df, formula, label=""):
    try:
        return smf.logit(formula, data=df).fit(disp=0, maxiter=500, method="bfgs")
    except Exception as e:
        print(f"  {label}: FAILED ({e})")
        return None


# ---------------------------------------------------------------------------
# Cohort prep
# ---------------------------------------------------------------------------

def prep_mimic(args):
    print("\nMIMIC cohort prep ...")
    df = pd.read_csv(args.mimic_per_stay)
    df["intime_dt"] = pd.to_datetime(df["intime"], errors="coerce")
    df = ensure_columns(df,
                        needed=["admittime", "admission_type",
                                "admission_location", "hospital_expire_flag",
                                "deathtime"],
                        source_path=args.mimic_root / "hosp" / "admissions.csv.gz",
                        key_col="hadm_id", label="admissions")
    df = ensure_columns(df, needed=["dod"],
                        source_path=args.mimic_root / "hosp" / "patients.csv.gz",
                        key_col="subject_id", label="patients")

    df["admission_source"] = df["admission_location"].apply(recode_source)
    df["admittime_dt"] = pd.to_datetime(df["admittime"], errors="coerce")
    df["dod_dt"] = pd.to_datetime(df["dod"], errors="coerce")
    df["deathtime_dt"] = pd.to_datetime(df["deathtime"], errors="coerce")

    days_to_dod = (df["dod_dt"] - df["intime_dt"]).dt.total_seconds() / 86400
    df["mortality_30d"] = (df["dod_dt"].notna() & (days_to_dod <= 30)
                           & (days_to_dod >= 0)).astype(int)
    df["hospital_mortality"] = df["hospital_expire_flag"].fillna(0).astype(int)

    death_to_in = (df["deathtime_dt"] - df["intime_dt"]).dt.total_seconds() / 3600
    df["died_in_first_24h"] = (df["deathtime_dt"].notna()
                               & (death_to_in.fillna(99) <= 24)).astype(int)
    df["survived_24h"] = 1 - df["died_in_first_24h"]

    df["pre_icu_los_hours"] = ((df["intime_dt"] - df["admittime_dt"])
                               .dt.total_seconds() / 3600).clip(lower=0)
    df["elective"] = (df["admission_type"].fillna("")
                      .str.contains("ELECTIVE|SURGICAL SAME DAY",
                                    case=False, regex=True).astype(int))

    # Legacy physiology (min/max) — kept only for the side-by-side comparison.
    physio = pd.read_csv(require(args.mimic_physio, "MIMIC legacy physiology"))
    df = df.merge(physio, on="stay_id", how="left")
    vent = pd.read_csv(require(args.mimic_vent, "MIMIC mech vent"))
    df["mech_vent_24h"] = df["stay_id"].isin(vent["stay_id"]).astype(int)
    print(f"  mech_vent_24h positive: {df['mech_vent_24h'].mean():.3f}")

    # v2 physiology (median/IQR)
    p2 = pd.read_csv(require(args.physio_v2_dir / "mimic_physiology_v2.csv",
                             "MIMIC v2 physiology"))
    df = df.merge(p2, on="stay_id", how="left")
    if "n_hr_24h_rebuilt" in df.columns:
        both = df.dropna(subset=["n_hr_24h", "n_hr_24h_rebuilt"])
        agree = (both["n_hr_24h"] == both["n_hr_24h_rebuilt"]).mean()
        print(f"  n_hr_24h vs rebuilt exact agreement: {agree:.3f} "
              f"(corr {both['n_hr_24h'].corr(both['n_hr_24h_rebuilt']):.4f})")
        if agree < 0.95:
            print("  *** WARNING: exposure definitions differ between the "
                  "per-stay file and the v2 build. Reconcile before "
                  "interpreting anything below.")

    for c in ("hr_median_24h", "hr_iqr_24h", "map_median_24h", "map_iqr_24h",
              "hr_min_24h", "hr_max_24h", "map_min_24h", "map_max_24h",
              "gcs_min_24h", "pre_icu_los_hours"):
        if c not in df.columns:
            df[c] = np.nan
        df[f"{c}_z"] = standardise(df[c])
    for c in ("age_z", "n_comorbid_z", "n_chapters_z"):
        if c not in df.columns:
            base = c.replace("_z", "")
            if base in df.columns:
                df[c] = standardise(pd.to_numeric(df[base], errors="coerce"))

    df["log_n_hr"] = np.log(df["n_hr_24h"].fillna(0) + 1)
    print(f"  prepared: {len(df):,}")
    return df


def prep_eicu(args):
    print("\neICU cohort prep ...")
    df = pd.read_csv(args.eicu_per_stay)
    df = ensure_columns(df,
                        needed=["unitdischargestatus", "hospitaldischargestatus",
                                "unitdischargeoffset"],
                        source_path=args.eicu_root / "patient.csv.gz",
                        key_col="patientunitstayid", label="eICU patient")

    df["hospital_mortality"] = (df["hospitaldischargestatus"] == "Expired").astype(int)
    df["icu_mortality"] = (df["unitdischargestatus"] == "Expired").astype(int)
    df["unitdischargeoffset"] = pd.to_numeric(df["unitdischargeoffset"],
                                              errors="coerce")
    df["died_in_first_24h"] = ((df["icu_mortality"] == 1)
                               & (df["unitdischargeoffset"] <= 1440)).astype(int)
    df["survived_24h"] = 1 - df["died_in_first_24h"]

    df["age_z"] = standardise(pd.to_numeric(df["age_int"], errors="coerce"))
    df["log_apache"] = pd.to_numeric(df["log_apache"], errors="coerce")
    df["log_apache"] = df["log_apache"].fillna(df["log_apache"].median())

    p2 = pd.read_csv(require(args.physio_v2_dir / "eicu_physiology_v2.csv",
                             "eICU v2 physiology"))
    df = df.merge(p2, on="patientunitstayid", how="left")
    for c in ("hr_median_24h", "hr_iqr_24h", "map_median_24h", "map_iqr_24h"):
        if c not in df.columns:
            df[c] = np.nan
        df[f"{c}_z"] = standardise(df[c])

    df["log_n_hr"] = np.log(df["n_hr_24h"].fillna(0) + 1)
    print(f"  prepared: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Residualisation
# ---------------------------------------------------------------------------

def residualize(df, key, predictor, covariates, out_col, label):
    """OLS predictor ~ covariates; write residuals to out_col. Returns R^2."""
    terms = [t.strip() for t in covariates.split("+")]
    numeric = [t for t in terms if not t.startswith("C(")]
    factors = [t[2:-1] for t in terms if t.startswith("C(")]
    sub = df.dropna(subset=[predictor] + numeric + factors).copy()
    print(f"\n  [{label}]")
    print(f"    n = {len(sub):,} (dropped {len(df) - len(sub):,} for missing; "
          f"NO imputation)")
    if len(sub) == 0:
        raise ValueError(f"{label}: no complete cases")
    fit = smf.ols(f"{predictor} ~ {covariates}", data=sub).fit()
    print(f"    R^2 = {fit.rsquared:.4f}")
    sub[out_col] = fit.resid
    df = df.merge(sub[[key, out_col]], on=key, how="left")
    return df, float(fit.rsquared), int(len(sub))


# ---------------------------------------------------------------------------
# Mortality models
# ---------------------------------------------------------------------------

def continuous_effect(df, casemix, outcome, resid_col, label):
    sub = df[df["survived_24h"] == 1].dropna(subset=[outcome, resid_col]).copy()
    sub["resid_z"] = standardise(sub[resid_col])
    fit = fit_logit(sub, f"{outcome} ~ {casemix} + resid_z", label)
    if fit is None:
        return None
    e = get_effect(fit, "resid_z")
    e.update({"spec": "continuous_per_SD", "n": int(len(sub)),
              "harm": e["OR"] < 1.0})
    return e


def binary_effects(df, casemix, outcome, resid_col, label, pcts=PCTS):
    sub_all = df[df["survived_24h"] == 1].dropna(subset=[outcome, resid_col])
    rows = []
    for pct in pcts:
        cutoff = sub_all[resid_col].quantile(pct / 100)
        sub = sub_all.copy()
        sub["low_resid"] = (sub[resid_col] <= cutoff).astype(int)
        fit = fit_logit(sub, f"{outcome} ~ {casemix} + low_resid",
                        f"{label} bottom {pct}%")
        if fit is None:
            continue
        e = get_effect(fit, "low_resid")
        e.update({"spec": f"bottom_{pct}pct", "percentile": pct,
                  "cutoff": float(cutoff), "rate": float(sub["low_resid"].mean()),
                  "n": int(len(sub)), "harm": e["OR"] > 1.0})
        rows.append(e)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-root", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-per-stay", required=True, type=Path)
    ap.add_argument("--mimic-physio", required=True, type=Path)
    ap.add_argument("--mimic-vent", required=True, type=Path)
    ap.add_argument("--physio-v2-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./residualization_v2"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("CONDITIONAL RESIDUALISATION v2")
    print("=" * 78)

    mimic = prep_mimic(args)
    eicu = prep_eicu(args)

    # ---- Residualisation: harmonised and legacy, side by side --------------
    print("\n" + "=" * 78)
    print("RESIDUALISATION REGRESSIONS")
    print("=" * 78)

    r2 = {}
    mimic, r2["MIMIC_harmonised"], n_mh = residualize(
        mimic, "stay_id", "log_n_hr", HARMONISED_ACUITY,
        "resid_harm", "MIMIC harmonised (median/IQR)")
    eicu, r2["eICU_harmonised"], n_eh = residualize(
        eicu, "patientunitstayid", "log_n_hr", HARMONISED_ACUITY,
        "resid_harm", "eICU harmonised (median/IQR)")
    mimic, r2["MIMIC_legacy"], n_ml = residualize(
        mimic, "stay_id", "log_n_hr", LEGACY_MIMIC_ACUITY,
        "resid_legacy", "MIMIC legacy (min/max + admin)")
    eicu, r2["eICU_legacy"], n_el = residualize(
        eicu, "patientunitstayid", "log_n_hr", LEGACY_EICU_ACUITY,
        "resid_legacy", "eICU legacy (log_apache)")

    print("\n  ACUITY-COUPLING CONTRAST")
    print(f"    legacy      MIMIC {r2['MIMIC_legacy']:.4f} vs eICU "
          f"{r2['eICU_legacy']:.4f}  -> ratio "
          f"{r2['MIMIC_legacy'] / max(r2['eICU_legacy'], 1e-12):.2f}x")
    print(f"    harmonised  MIMIC {r2['MIMIC_harmonised']:.4f} vs eICU "
          f"{r2['eICU_harmonised']:.4f}  -> ratio "
          f"{r2['MIMIC_harmonised'] / max(r2['eICU_harmonised'], 1e-12):.2f}x")
    print("    The harmonised ratio is the defensible one. If it collapses "
          "toward 1, the fourfold claim was a specification artifact.")
    pd.DataFrame([{"model": k, "r2": v} for k, v in r2.items()]).to_csv(
        args.out_dir / "acuity_coupling_r2.csv", index=False)

    # ---- Mortality models ---------------------------------------------------
    MIMIC_CASEMIX = ("age_z + C(gender) + n_comorbid_z + n_chapters_z "
                     "+ C(anchor_year_group) + C(careunit) + C(admission_source) "
                     "+ hr_median_24h_z + hr_iqr_24h_z + map_median_24h_z "
                     "+ map_iqr_24h_z + mech_vent_24h + elective")
    EICU_CASEMIX = ("age_z + C(gender) + hr_median_24h_z + hr_iqr_24h_z "
                    "+ map_median_24h_z + map_iqr_24h_z + C(unittype)")

    print("\n" + "=" * 78)
    print("MORTALITY MODELS  (harm: binary OR>1, continuous OR<1)")
    print("=" * 78)

    arms = [
        ("MIMIC", "hospital_mortality", "PRIMARY (matched)", mimic,
         MIMIC_CASEMIX, "stay_id"),
        ("eICU", "hospital_mortality", "PRIMARY (matched)", eicu,
         EICU_CASEMIX, "patientunitstayid"),
        ("MIMIC", "mortality_30d", "secondary", mimic, MIMIC_CASEMIX, "stay_id"),
    ]

    rows = []
    for cohort, outcome, role, df, casemix, _ in arms:
        for resid_col, spec_label in (("resid_harm", "harmonised"),
                                      ("resid_legacy", "legacy")):
            tag = f"{cohort} {outcome} [{spec_label}]"
            e = continuous_effect(df, casemix, outcome, resid_col, tag)
            if e:
                print(f"  {tag:46s} continuous  {fmt(e)}  harm={e['harm']}")
                rows.append({"cohort": cohort, "outcome": outcome, "role": role,
                             "residual": spec_label, **e})
            for b in binary_effects(df, casemix, outcome, resid_col, tag):
                if b["percentile"] == 10:
                    print(f"  {tag:46s} bottom10%   {fmt(b)}  harm={b['harm']}")
                rows.append({"cohort": cohort, "outcome": outcome, "role": role,
                             "residual": spec_label, **b})

    res = pd.DataFrame(rows)
    res.to_csv(args.out_dir / "residual_mortality_v2.csv", index=False)

    # ---- Verdict ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT (harmonised residual, hospital mortality, both cohorts)")
    print("=" * 78)

    key = res[(res["residual"] == "harmonised")
              & (res["outcome"] == "hospital_mortality")
              & (res["spec"].isin(["continuous_per_SD", "bottom_10pct"]))]

    for cohort in ("MIMIC", "eICU"):
        sub = key[key["cohort"] == cohort]
        if sub.empty:
            continue
        flags = dict(zip(sub["spec"], sub["harm"]))
        internal = len(set(flags.values())) == 1
        print(f"\n  {cohort}:")
        for _, r in sub.iterrows():
            print(f"    {r['spec']:20s} {fmt(r)}  harm={r['harm']}")
        print(f"    within-cohort specifications concordant: "
              f"{'YES' if internal else 'NO'}")

    m = key[(key["cohort"] == "MIMIC") & (key["spec"] == "bottom_10pct")]
    e = key[(key["cohort"] == "eICU") & (key["spec"] == "bottom_10pct")]
    if not m.empty and not e.empty:
        mh, eh = bool(m.iloc[0]["harm"]), bool(e.iloc[0]["harm"])
        print(f"\n  cross-cohort harm direction: MIMIC={mh}, eICU={eh} -> "
              f"{'AGREE' if mh == eh else 'REVERSAL'}")
        if mh != eh:
            print("    A directional reversal on the matched outcome under a "
                  "harmonised residualisation is the paper's finding.")
            print("    It is NOT 'specification instability in MIMIC' — check "
                  "the within-cohort concordance lines above before writing.")

    print(f"\nOutputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
