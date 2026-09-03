#!/usr/bin/env bash
# Paper 17 — full reproduction, in dependency order.
#
# Every command below is the script's own documented usage. Steps that create a
# parquet cache are marked CACHE; everything downstream reads those caches, so
# the order matters. Total wall time is roughly 3 hours on first run, most of it
# in the four full-table passes, and under 40 minutes on any rerun.
#
#   ./run_all.sh
#
# Set MIMIC, EICU and OUT below, or export them before running.

set -euo pipefail

MIMIC="${MIMIC:-$HOME/physionet.org/files/mimiciv/3.1}"
EICU="${EICU:-$HOME/physionet.org/files/eicu-crd/2.0}"
OUT="${OUT:-$HOME/bcst}"
S="$(dirname "$0")/scripts"
R="uv run --offline"

# ---------------------------------------------------------------------------
# PREREQUISITE, not produced by this repository
# ---------------------------------------------------------------------------
# per_stay_multi_outcomes.csv is the MIMIC-IV stay-level cohort table: stay_id,
# careunit, los, adm_hour, age_z, gender, n_comorbid_z, n_chapters_z,
# anchor_year_group, hospital_expire_flag, and the pooled-item count n_hr_24h.
# It comes from the earlier BCST cohort build, not from any script here. Nothing
# below runs without it.
PER_STAY="${PER_STAY:-$OUT/multi_outcome_results/per_stay_multi_outcomes.csv}"
test -f "$PER_STAY" || { echo "missing prerequisite: $PER_STAY"; exit 1; }

# paper17_reconcile_counts.py additionally wants physiology_24h.csv from the
# earlier OASIS adjustment run; it is optional and only affects step 5.
PHYSIO="${PHYSIO:-$OUT/mimic_oasis_adjustment_results/physiology_24h.csv}"

# ---------------------------------------------------------------------------
# 1. Exposure composition and the source caches
# ---------------------------------------------------------------------------
$R "$S/paper17_diagnose_exposure.py" \
    --mimic-root "$MIMIC" --mimic-per-stay "$PER_STAY" \
    --out-dir "$OUT/exposure_diagnosis"

# CACHE: unit_profile/hr_timestamps.parquet
$R "$S/paper17_unit_documentation_profile.py" \
    --mimic-root "$MIMIC" --per-stay "$PER_STAY" \
    --out-dir "$OUT/unit_profile"

# CACHE: unit_profile_eicu/{nursecharting,vitalperiodic}_offsets.parquet
$R "$S/paper17_eicu_documentation_profile.py" \
    --eicu-root "$EICU" --source both \
    --out-dir "$OUT/unit_profile_eicu"

MC="$OUT/unit_profile/hr_timestamps.parquet"
NC="$OUT/unit_profile_eicu/nursecharting_offsets.parquet"
VP="$OUT/unit_profile_eicu/vitalperiodic_offsets.parquet"

$R "$S/paper17_build_physiology_v2.py" \
    --mimic-root "$MIMIC" --eicu-root "$EICU" \
    --out-dir "$OUT/physiology_v2"

$R "$S/paper17_reconcile_counts.py" \
    --per-stay "$PER_STAY" --physio "$PHYSIO" \
    --physio-v2 "$OUT/physiology_v2/mimic_physiology_v2.csv" \
    --out-dir "$OUT/reconcile"

# ---------------------------------------------------------------------------
# 2. Replication variables
# ---------------------------------------------------------------------------
$R "$S/paper17_probe_vitals.py" \
    --mimic-root "$MIMIC" --eicu-root "$EICU" \
    --out-dir "$OUT/vitals_probe"

# CACHE: cross_vitals/{mimic,eicu_nc,eicu_vp}.parquet
$R "$S/paper17_cross_vitals.py" \
    --mimic-root "$MIMIC" --eicu-root "$EICU" --mimic-per-stay "$PER_STAY" \
    --out-dir "$OUT/cross_vitals"

# ---------------------------------------------------------------------------
# 3. Original analysis, all from cache
# ---------------------------------------------------------------------------
$R "$S/paper17_pooled_unit_comparison.py" \
    --mimic-cache "$MC" --mimic-per-stay "$PER_STAY" --eicu-cache "$NC" \
    --eicu-root "$EICU" --out-dir "$OUT/pooled_comparison"

$R "$S/paper17_decomposition_ci.py" \
    --mimic-cache "$MC" --mimic-per-stay "$PER_STAY" --eicu-cache "$NC" \
    --eicu-vp-cache "$VP" --eicu-root "$EICU" \
    --out-dir "$OUT/decomposition_ci"

$R "$S/paper17_stream_selection.py" \
    --mimic-cache "$MC" --mimic-per-stay "$PER_STAY" --eicu-nc-cache "$NC" \
    --eicu-vp-cache "$VP" --eicu-root "$EICU" \
    --out-dir "$OUT/stream_selection"

$R "$S/paper17_chance_comparator.py" \
    --hr-nc "$NC" --hr-vp "$VP" --cross-dir "$OUT/cross_vitals" \
    --eicu-root "$EICU" --out-dir "$OUT/chance_comparator"

$R "$S/paper17_threshold_consequence.py" \
    --eicu-cache "$NC" --eicu-root "$EICU" --mimic-cache "$MC" \
    --mimic-per-stay "$PER_STAY" --out-dir "$OUT/threshold_consequence"

$R "$S/paper17_sensitivity.py" \
    --mimic-cache "$MC" --mimic-per-stay "$PER_STAY" --eicu-cache "$NC" \
    --eicu-root "$EICU" --out-dir "$OUT/sensitivity"

$R "$S/paper17_temporal_rerun.py" \
    --mimic-cache "$MC" --mimic-per-stay "$PER_STAY" --eicu-nc-cache "$NC" \
    --eicu-vp-cache "$VP" --eicu-root "$EICU" \
    --out-dir "$OUT/temporal_rerun"

$R "$S/paper17_table5_harmonise.py" \
    --hr-cache "$NC" --cross-dir "$OUT/cross_vitals" --eicu-root "$EICU" \
    --out-dir "$OUT/table5"

# ---------------------------------------------------------------------------
# 4. Severity harmonisation, reported in the manuscript as unsuccessful
# ---------------------------------------------------------------------------
$R "$S/paper17_probe_sofa_coverage.py" \
    --mimic-root "$MIMIC" --eicu-root "$EICU" --out-dir "$OUT/sofa_probe"

$R "$S/paper17_build_severity_v3.py" \
    --mimic-root "$MIMIC" --eicu-root "$EICU" --out-dir "$OUT/severity_v3"

# bcst_residualization_v2.py: see its own header for arguments.

# ---------------------------------------------------------------------------
# 5. Revision analyses
# ---------------------------------------------------------------------------
$R "$S/paper17_stage1_checks.py" \
    --mimic-cache "$MC" --mimic-per-stay "$PER_STAY" --eicu-nc-cache "$NC" \
    --eicu-vp-cache "$VP" --eicu-root "$EICU" \
    --out-dir "$OUT/revision_stage1"

# paper17_stage2_glmm.py: confirm the flag names against its own header before
# running; it needs the MIMIC and eICU caches, per-stay table and eICU root.
$R "$S/paper17_stage2_glmm.py" \
    --mimic-cache "$MC" --mimic-per-stay "$PER_STAY" --eicu-nc-cache "$NC" \
    --eicu-root "$EICU" --out-dir "$OUT/revision_stage2"

$R "$S/paper17_stage3_exclusions.py" \
    --mimic-cache "$MC" --mimic-per-stay "$PER_STAY" --eicu-nc-cache "$NC" \
    --eicu-vp-cache "$VP" --cross-dir "$OUT/cross_vitals" \
    --eicu-root "$EICU" --mimic-root "$MIMIC" \
    --out-dir "$OUT/revision_stage3"

$R "$S/paper17_stage4_restricted.py" \
    --mimic-cache "$MC" --mimic-per-stay "$PER_STAY" --eicu-nc-cache "$NC" \
    --eicu-vp-cache "$VP" --eicu-root "$EICU" \
    --out-dir "$OUT/revision_stage4"

$R "$S/paper17_stage5_remaining.py" \
    --mimic-cache "$MC" --mimic-per-stay "$PER_STAY" --eicu-nc-cache "$NC" \
    --eicu-vp-cache "$VP" --cross-dir "$OUT/cross_vitals" \
    --eicu-root "$EICU" --out-dir "$OUT/revision_stage5"

# Authoritative for every Table 3 and Table 5 cell. 15-20 min.
$R "$S/paper17_tables_final.py" \
    --mimic-cache "$MC" --mimic-per-stay "$PER_STAY" --eicu-nc-cache "$NC" \
    --cross-dir "$OUT/cross_vitals" --eicu-root "$EICU" \
    --out-dir "$OUT/revision_tables"

# Verify the fitter recovers known variances, then fit the cohort. 10-20 min.
$R "$S/paper17_nb_glmm.py" --simulate
$R "$S/paper17_nb_glmm.py" \
    --eicu-nc-cache "$NC" --eicu-root "$EICU" --out-dir "$OUT/nb_glmm"

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
# Figure 1 only; Figures 2-4 from this script are the superseded unrestricted
# versions and are not used in the manuscript.
$R "$S/paper17_figures.py" \
    --unit-profile "$OUT/unit_profile" \
    --unit-profile-eicu "$OUT/unit_profile_eicu" \
    --decomposition "$OUT/revision_tables" \
    --threshold "$OUT/revision_stage4" \
    --exposure-diagnosis "$OUT/exposure_diagnosis" \
    --eicu-root "$EICU" --mimic-per-stay "$PER_STAY" \
    --out-dir "$OUT/figures"

# Figures 2, 3, 4 on the restricted cohort, and Figure 5. 184 dpi.
$R "$S/paper17_revision_figures.py" \
    --tables-dir "$OUT/revision_tables" \
    --stage4-dir "$OUT/revision_stage4" \
    --stage3-dir "$OUT/revision_stage3" \
    --eicu-nc-cache "$NC" --mimic-cache "$MC" --mimic-per-stay "$PER_STAY" \
    --eicu-root "$EICU" --out-dir "$OUT/revision_figures"

echo
echo "Done. Manuscript values come from:"
echo "  Tables 3 and 5      $OUT/revision_tables/table{3,5}_cells.csv"
echo "  Threshold, restricted   $OUT/revision_stage4/threshold_restricted.csv"
echo "  Floor sweep         $OUT/revision_stage3/plausibility_thresholds.csv"
echo "  NB GLMM             $OUT/nb_glmm/nb_glmm_vpc.csv"
echo "  Figures 2-5         $OUT/revision_figures/"
