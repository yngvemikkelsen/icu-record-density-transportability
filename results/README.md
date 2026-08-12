# Aggregate analysis outputs

Populate this directory from your local runs before cutting the release. Nothing
here contains patient-level data; all files are aggregate statistics.

Expected contents, by producing script:

| File | From |
|---|---|
| `per_item_volume.csv`, `charting_hour_by_item.csv`, `subset_reconstruction.csv` | `paper17_diagnose_exposure.py` |
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

Do not commit the parquet caches (`hr_timestamps.parquet`,
`nursecharting_offsets.parquet`, `vitalperiodic_offsets.parquet`,
`mimic.parquet`, `eicu_nc.parquet`, `eicu_vp.parquet`). They are derived from
credentialed data and are excluded by `.gitignore`.
