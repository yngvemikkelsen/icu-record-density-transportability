# Aggregate analysis outputs

Populate this directory from your local runs before cutting the release. Nothing
here contains patient-level data; all files are aggregate statistics.

Expected contents, by producing script:

| File | From |
|---|---|
| `per_item_volume.csv`, `charting_hour_by_item.csv`, `charting_hour_concentration.csv`, `subset_reconstruction.csv` | `paper17_diagnose_exposure.py` |
| `count_reconciliation.csv` | `paper17_reconcile_counts.py` |
| `unit_profiles.csv`, `unit_transport_eta2.csv`, `charting_hour_by_unit.csv` | `paper17_unit_documentation_profile.py` |
| `vitalperiodic_transport_eta2.csv`, `nursecharting_transport_eta2.csv` | `paper17_eicu_documentation_profile.py` |
| `pooled_decomposition.csv`, `unit_by_unit_percentiles.csv`, `matched_type_comparison.csv` | `paper17_pooled_unit_comparison.py` |
| `eicu_decomposition_ci.csv`, `mimic_unit_ci.csv`, `check_gap30.csv`, `check_hospital_coverage.csv` | `paper17_decomposition_ci.py` |
| `stream_distributions.csv`, `bottom_decile_agreement.csv`, `database_component_by_stream.csv`, `mimic_position_by_stream.csv` | `paper17_stream_selection.py` |
| `chance_comparator.csv` | `paper17_chance_comparator.py` |
| `eicu_hospital_agreement.csv`, `mimic_unit_agreement.csv`, `*_group_shares.csv`, `eicu_top_contributors.csv` | `paper17_threshold_consequence.py` |
| `eligibility_shares.csv`, `eligibility_by_hospital.csv`, `eligibility_count_eta2.csv`, `mixed_vpc.csv` | `paper17_sensitivity.py` |
| `mimic_hourly_or.csv`, `eicu_hourly_or.csv`, `mimic_by_careunit.csv`, `eicu_hospital_peaks.csv` | `paper17_temporal_rerun.py` |
| `cross_vitals_summary.csv` | `paper17_cross_vitals.py` |
| `table5_components.csv` | `paper17_table5_harmonise.py` |
| `mimic_sofa_coverage.csv`, `eicu_sofa_coverage.csv` | `paper17_probe_sofa_coverage.py` |
| `acuity_coupling_r2.csv`, `residual_mortality_v2.csv` | `bcst_residualization_v2.py` |
| `duplicate_timestamps.csv`, `covariate_missingness.csv` | `paper17_stage1_checks.py` |
| `glmm_count_vpc.csv` | `paper17_stage2_glmm.py` |
| `excluded_stay_profile.csv`, `plausibility_thresholds.csv`, `cross_variable_hospital_medians.csv` | `paper17_stage3_exclusions.py` |
| `eta2_restricted.csv`, `vpc_restricted.csv`, `threshold_restricted.csv` | `paper17_stage4_restricted.py` |
| `hospital_medians.csv`, `ss_order_sensitivity.csv` | `paper17_stage5_remaining.py` |
| `table3_cells.csv`, `table5_cells.csv` | `paper17_tables_final.py` |
| `nb_glmm_vpc.csv` | `paper17_nb_glmm.py` |
| `attribute_coverage.csv`, `attribute_marginal_eta2.csv`, `attribute_conditional_eta2.csv`, `attribute_permutation.csv`, `attribute_profiles.csv` | `paper17_hospital_attributes.py` |
| `vpc_ci.csv`, `vpc_validation.csv` | `paper17_vpc_ci.py` |
| `count_models.csv`, `metric_correlations.csv` | `paper17_count_models.py` |

`covariate_missingness.csv` carries an `n_entering` column giving the number of stays entering each model, so the rates can be checked against the manuscript denominators without re-deriving them. The eICU-CRD rows are computed on the 177,198 unit stays present in both source streams, which is the frame the admission-hour models are fitted on, not the 181,731-stay nurse-stream frame.

`glmm_count_vpc.csv` is superseded in full, and so are the two rows of `vpc_restricted.csv` whose `model` is `Negative binomial`. Their Poisson and negative binomial coefficients come from `paper17_stage2_glmm.py` and `paper17_stage4_restricted.py`, earlier routines that fitted the count models by a different method: they give a hospital coefficient of 0.776 unrestricted and 0.437 restricted, against the 0.728 and 0.348 reported in the manuscript. The reported values are those in `count_models.csv`, fitted by `paper17_count_models.py`, which `nb_glmm_vpc.csv` reproduces independently to five decimal places. Both files are retained for provenance and neither should be read as the reported estimates. The Gaussian rows of `vpc_restricted.csv` are current, as are all rows of `mixed_vpc.csv`, which is the source of the Gaussian coefficients quoted in the manuscript.

`table3_cells.csv` reproduces columns 2 to 5 of Table 3 exactly: the database-first, hospital and unit increments and the eICU-CRD hospital eta-squared with its interval. Its `vpc_hospital` column does not reproduce the `Hospital VPC (95% CI)` column of the printed table, and it carries no interval for that column. The printed column draws on two sources. For Record count and Gaps >30 min it is the negative binomial latent-scale coefficient with its profile-likelihood interval from `count_models.csv`, 0.348 (0.252-0.457) and 0.376 (0.264-0.494). For the six remaining metrics it is the Gaussian coefficient with its delete-one-hospital BCa interval from `vpc_ci.csv`. The `vpc_hospital` column of `table3_cells.csv` is the Gaussian coefficient throughout, from an earlier routine, and for the two count metrics it is superseded.

Do not commit the parquet caches (`hr_timestamps.parquet`,
`nursecharting_offsets.parquet`, `vitalperiodic_offsets.parquet`,
`mimic.parquet`, `eicu_nc.parquet`, `eicu_vp.parquet`). They are derived from
credentialed data and are excluded by `.gitignore`.
