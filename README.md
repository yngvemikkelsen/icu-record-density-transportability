# ICU vital-sign record density: source-stream and site-level variation

Analysis code for:

> Mikkelsen Y. Data-Stream and Hospital-Level Variation in ICU Vital-Sign Record
> Density Across MIMIC-IV and eICU-CRD: Retrospective Data Quality Study.
> *JMIR Medical Informatics* (under revision, 2026).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22295700.svg)](https://doi.org/10.5281/zenodo.22295700)

---

## What this study does

Vital-sign record density — the count of charted observations per unit time — is
used as a process measure in clinical informatics. This study asks at which level
it varies: data stream, database, contributing site, or care unit.

The answer, across MIMIC-IV and eICU-CRD, is the **contributing site**. Two
further results follow from that. A difference that looks architectural is
reproduced by the choice of source table within one database: pairing MIMIC-IV
against eICU-CRD's monitor stream places 96.0% of the variance in record count at
the database level, and pairing the same MIMIC-IV data against eICU-CRD's nurse
stream reduces it to 2.6%. And a pooled low-density threshold designates
hospitals rather than patients: six of 60 hospitals supply half of everything it
flags in the restricted primary analysis, and five of 68 before restriction.

No patient-outcome association is interpreted as a study finding. The reason is
given in the manuscript: no severity instrument can be harmonised across the two
cohorts without either sharing the exposure's data stream or discarding a quarter
of one cohort in a manner correlated with the exposure.

---

## The restricted primary cohort

The revision distinguishes documentation practice from incomplete contribution of
the nurse-charted stream to the research database. Eight of the 68 eICU-CRD
hospitals contributing at least 500 eligible stays record a median of fewer than
12 heart-rate observations per 24 hours, against 26 for the cohort as a whole.
They are excluded from the primary analysis, leaving **60 hospitals and 88,560
stays**; the pooled decomposition then covers **155,700 stays across 61
hospitals** including MIMIC-IV.

The floor is empirical rather than a clinical standard: the median charting
interval is 60 minutes in both MIMIC-IV and the eICU-CRD nurse stream, so 12
observations per 24 hours is deliberately half the observed cadence. The result
holds across floors:

| Floor | Hospitals | eta2 count | eta2 median interval | eta2 gap fraction |
|---|---|---|---|---|
| none | 68 | 0.441 | 0.763 | 0.621 |
| >=6 (four-hourly) | 63 | 0.386 | 0.459 | 0.442 |
| >=12 (two-hourly, primary) | 60 | 0.354 | 0.251 | 0.208 |
| >=24 (hourly) | 55 | 0.321 | 0.188 | 0.149 |

Unrestricted estimates are retained throughout as sensitivity analyses.
Estimates from the two cohorts are not interchangeable.

---

## Data

Neither dataset is redistributed here. Both require credentialed access through
PhysioNet under their respective data use agreements:

- **MIMIC-IV v3.1** — https://physionet.org/content/mimiciv/3.1/
- **eICU-CRD v2.0** — https://physionet.org/content/eicu-crd/2.0/

Scripts take `--mimic-root` and `--eicu-root` pointing at local extractions.
Nothing in `results/` contains patient-level data; all outputs are aggregate.

Operational stream provenance, source-field selectors, the definition of one
record, duplicate-timestamp handling, plausible value ranges and every derived
metric are documented in Multimedia Appendix 1 of the manuscript.

---

## Scripts, in the order they were run

Each maps to the manuscript section it produces. Most cache intermediate
extractions to parquet, so reruns after the first are fast.

### Original analysis

| # | Script | Produces |
|---|---|---|
| 1 | `paper17_diagnose_exposure.py` | Exposure composition: which itemids reconstruct the pooled-item count, and their charting-hour concentration (Table 1, Figure 1) |
| 2 | `paper17_reconcile_counts.py` | Reconciles the three competing definitions of "heart-rate records in the first 24 h" |
| 3 | `paper17_build_physiology_v2.py` | First-24 h median/IQR physiology summaries, both cohorts, avoiding order statistics |
| 4 | `paper17_unit_documentation_profile.py` | MIMIC-IV documentation profile by care unit |
| 5 | `paper17_eicu_documentation_profile.py` | eICU-CRD profile by unit type and by hospital, both source streams |
| 6 | `paper17_pooled_unit_comparison.py` | Pooled decomposition and unit-by-unit positioning of MIMIC-IV against eICU-CRD |
| 7 | `paper17_decomposition_ci.py` | Hospital-clustered bootstrap on the decomposition; mixed-effects variance partition |
| 8 | `paper17_stream_selection.py` | Nurse-versus-monitor stream comparison (Table 2) |
| 9 | `paper17_chance_comparator.py` | Correct chance comparator for stream agreement under integer ties |
| 10 | `paper17_threshold_consequence.py` | Pooled versus site-specific low-density thresholds, unrestricted |
| 11 | `paper17_sensitivity.py` | Eligibility sensitivity and mixed-model variance partition, unrestricted |
| 12 | `paper17_temporal_rerun.py` | Admission-hour association under four specifications (Table 4) |
| 13 | `paper17_probe_vitals.py` | Verifies itemids and value-name strings for the replication variables |
| 14 | `paper17_cross_vitals.py` | Replication in oxygen saturation and respiratory rate |
| 15 | `paper17_table5_harmonise.py` | Puts the replication variance components on one cohort basis |
| 16 | `paper17_figures.py` | Figure 1, and the unrestricted versions of Figures 2-4 |

### Revision analysis

| # | Script | Produces |
|---|---|---|
| 17 | `paper17_stage1_checks.py` | Duplicate-timestamp burden per stream; per-covariate missingness and complete-case loss for the admission-hour models |
| 18 | `paper17_stage2_glmm.py` | Latent-scale Poisson and negative binomial variance partitions; sequential eta-squared under all six factor orderings with an order-free last-in contribution |
| 19 | `paper17_stage3_exclusions.py` | Characteristics of stays excluded by the three-record minimum; hospital variance component across plausibility floors; cross-variable sparseness pattern |
| 20 | `paper17_stage4_restricted.py` | Variance components with hospital-clustered bootstrap, mixed-model variance partition, and threshold consequence, restricted and unrestricted side by side |
| 21 | `paper17_stage5_remaining.py` | Cardiac vascular ICU quantities; participant-flow figure; per-hospital interval spread and stream contrast on the restricted cohort |
| 22 | `paper17_tables_final.py` | Every Table 3 and Table 5 cell recomputed in one run so each has a single provenance. **Supersedes the table outputs of 7, 11 and 15.** |
| 23 | `paper17_nb_glmm.py` | Fitted negative binomial GLMM, log link, random intercepts for hospital and unit within hospital, dispersion estimated jointly, maximum likelihood by Laplace approximation |
| 24 | `paper17_revision_figures.py` | Figures 2, 3, 4 on the restricted cohort and Figure 5, at 184 dpi |
| 25 | `paper17_hospital_attributes.py` | Whether the hospital attributes eICU-CRD records (bed-capacity category, teaching status, region) account for the site component, with a hospital-level permutation null |

Three further scripts document the severity-harmonisation attempt reported in the
manuscript as unsuccessful. They are included because the negative result is part
of the argument:

| Script | Purpose |
|---|---|
| `paper17_probe_sofa_coverage.py` | Coverage of candidate severity components in both cohorts |
| `paper17_build_severity_v3.py` | Harmonised severity components, worst-value rule stated in the header |
| `bcst_residualization_v2.py` | Residualisation and outcome models under three adjustment regimes |

---

## Reproducing the analysis

`run_all.sh` runs every step in dependency order, with each command taken from
the script's own documented usage:

```bash
export MIMIC=~/physionet.org/files/mimiciv/3.1
export EICU=~/physionet.org/files/eicu-crd/2.0
export OUT=~/bcst
./run_all.sh
```

**One prerequisite is not in this repository.** `per_stay_multi_outcomes.csv` is
the MIMIC-IV stay-level cohort table (`stay_id`, `careunit`, `los`, `adm_hour`,
`age_z`, `gender`, `n_comorbid_z`, `n_chapters_z`, `anchor_year_group`,
`hospital_expire_flag`, and the pooled-item count `n_hr_24h`). It comes from the
earlier cohort build, not from any script here, and nothing runs without it.
`paper17_reconcile_counts.py` also takes an optional `physiology_24h.csv` from
the same source. Set `PER_STAY` and `PHYSIO` if they are not at the default
paths; `run_all.sh` stops immediately if the first is missing.

Scripts use [PEP 723](https://peps.python.org/pep-0723/) inline dependency
metadata and run under [uv](https://docs.astral.sh/uv/); each also prints its own
usage. Dependencies are pandas, numpy, statsmodels, scipy, pyarrow and
matplotlib. No GPU is required.

Four steps read the source tables in full and create the parquet caches that
everything else uses: `paper17_unit_documentation_profile.py`,
`paper17_eicu_documentation_profile.py`, `paper17_build_physiology_v2.py` and
`paper17_cross_vitals.py`. Those dominate the first run at roughly two hours
together. Reruns read the caches and finish in well under an hour, of which
`paper17_tables_final.py` (eight metrics by 500 bootstrap replicates by two
cohorts, plus eight mixed models each side) takes 15 to 20 minutes and
`paper17_nb_glmm.py` 10 to 20.

`paper17_nb_glmm.py --simulate` fits the model to simulated overdispersed counts
with a known random-effect variance, matched to the cohort in size and cluster
structure, and reports the recovery error. `run_all.sh` runs it before the real
fit.

Where the same quantity is produced by more than one script,
`paper17_tables_final.py` is authoritative for Tables 3 and 5 and
`paper17_stage4_restricted.py` for the threshold consequence. `run_all.sh` prints
the file behind each manuscript table when it finishes.

---

## Reproducibility notes

Five points that matter for anyone re-running or extending this:

**Exposure composition is not obvious from the itemid.** The pooled-item MIMIC-IV
heart-rate count reconstructs as 220045 + 220046 + 220047 at 99.3% exact
agreement. The latter two are alarm limits, charted at shift boundaries, and are
documentation events rather than observations. The same pattern holds for oxygen
saturation (223769, 223770, 226253) and respiratory rate (224161, 224162).
`paper17_diagnose_exposure.py` and `paper17_probe_vitals.py` establish this rather
than assuming it.

**Percentile rules on integer counts do not flag the nominal percentile.** Record
counts are discrete, and a rule that includes all stays at or below a cutoff
flags more than the nominal share wherever the distribution is narrow. This
affects both the stream-agreement comparator and the threshold analysis;
`paper17_chance_comparator.py` computes the correct comparator from observed
marginals rather than assuming the nominal value.

**Records are rows, and that is equivalent to distinct timestamps here.** The
counting code groups by stay and calls `.size()`. Among 2,444,918 MIMIC-IV
heart-rate records no timestamp is duplicated within a stay; in eICU-CRD, 2 of
5,223,657 nurse-charted and 17 of 45,164,349 monitor records are. Zero-length
intervals account for under 0.001% of intervals in every stream and no metric
changes on de-duplication. `paper17_stage1_checks.py` establishes this.

**The sequential decomposition is order-dependent by construction.** The three
factors are perfectly nested, so whichever is entered last contributes exactly
0.000, and across orderings the hospital component for median charting interval
spans 0.000 to 0.759. The pooled columns are a descriptive summary under a stated
ordering; the variance partition coefficients carry the inferential statement.
`paper17_stage2_glmm.py` computes both.

**Conditioning on categorical attributes removes variance by construction.** The
eICU-CRD `hospital` table records bed-capacity category, teaching status and
region. Conditioning a metric on all three costs six indicator coefficients from
a between-hospital structure with only as many degrees of freedom as there are
hospitals, so part of the hospital component is removed whether or not the
attributes are associated with it. `paper17_hospital_attributes.py` therefore
refers each reduction to a null obtained by permuting the joint attribute vector
across hospitals; on these data the null median reduction is 9-11%, against
observed reductions of 14-29%.

**Analysis sets differ by design and are named in the code.** The restricted
primary cohort is 88,560 eICU-CRD stays across 60 hospitals, and 155,700 pooled
stays across 61 hospitals. Unrestricted sensitivity analyses use 119,317 eICU-CRD
stays across 193 hospitals for clustered inference, 96,518 across the 68
hospitals contributing at least 500 eligible stays for mixed models and
per-hospital summaries, and 186,457 pooled stays across 194 hospitals. Unit-level
profiles use hospital-by-unit-type cells with at least 50 stays. The admission-hour
models are reported both for all stays and for stays completing the 24-hour
window. Estimates from different sets are not interchangeable.

---

## Repository contents

```
.
├── README.md
├── LICENSE                                MIT
├── CITATION.cff                           v1.1.0, Zenodo DOI, paper reference
├── .zenodo.json                           Zenodo deposition metadata
├── .gitignore                             excludes the parquet caches
├── requirements.txt
├── setup.sh
├── run_all.sh                             full pipeline, dependency order
│
├── scripts/                               28 files
│   ├── paper17_diagnose_exposure.py           1  exposure composition
│   ├── paper17_reconcile_counts.py            2  competing count definitions
│   ├── paper17_build_physiology_v2.py         3  first-24 h physiology
│   ├── paper17_unit_documentation_profile.py  4  MIMIC-IV profile by unit
│   ├── paper17_eicu_documentation_profile.py  5  eICU-CRD profile, both streams
│   ├── paper17_pooled_unit_comparison.py      6  pooled decomposition
│   ├── paper17_decomposition_ci.py            7  clustered bootstrap
│   ├── paper17_stream_selection.py            8  nurse vs monitor
│   ├── paper17_chance_comparator.py           9  chance comparator under ties
│   ├── paper17_threshold_consequence.py      10  pooled vs site thresholds
│   ├── paper17_sensitivity.py                11  eligibility, mixed-model VPC
│   ├── paper17_temporal_rerun.py             12  admission-hour models
│   ├── paper17_probe_vitals.py               13  itemid and valname probe
│   ├── paper17_cross_vitals.py               14  SpO2 and respiratory rate
│   ├── paper17_table5_harmonise.py           15  replication on one basis
│   ├── paper17_figures.py                    16  Figure 1, unrestricted 2-4
│   ├── paper17_stage1_checks.py              17  duplicates, missingness
│   ├── paper17_stage2_glmm.py                18  latent-scale VPC, entry order
│   ├── paper17_stage3_exclusions.py          19  excluded stays, floor sweep
│   ├── paper17_stage4_restricted.py          20  restricted primary analyses
│   ├── paper17_stage5_remaining.py           21  CVICU, flow figure, spread
│   ├── paper17_tables_final.py               22  every Table 3 and 5 cell
│   ├── paper17_nb_glmm.py                    23  fitted NB GLMM
│   ├── paper17_revision_figures.py           24  Figures 2-5, restricted
│   ├── paper17_hospital_attributes.py        25  recorded site attributes
│   ├── paper17_probe_sofa_coverage.py            severity component coverage
│   ├── paper17_build_severity_v3.py              harmonised severity attempt
│   └── bcst_residualization_v2.py                residualisation, outcome models
│
├── figures/                               PNG and PDF
│   ├── Figure1_charting_hour.*                300 dpi, unaffected by restriction
│   ├── Figure2_variance_decomposition.*       184 dpi, restricted cohort
│   ├── Figure3_hospital_intervals.*           184 dpi, restricted cohort
│   ├── Figure4_pooled_threshold.*             184 dpi, restricted cohort
│   └── Figure5_flow.*                         184 dpi, participant flow
│
└── results/                               60 CSVs, aggregate only
    ├── README.md                              maps every file to its script
    │
    │   exposure composition (1, 2, 13)
    ├── per_item_volume.csv
    ├── charting_hour_by_item.csv
    ├── subset_reconstruction.csv
    ├── count_reconciliation.csv
    │
    │   documentation profiles (4, 5)
    ├── unit_profiles.csv
    ├── unit_transport_eta2.csv
    ├── charting_hour_by_unit.csv
    ├── nursecharting_transport_eta2.csv
    ├── vitalperiodic_transport_eta2.csv
    │
    │   decomposition (6, 7)
    ├── pooled_decomposition.csv
    ├── unit_by_unit_percentiles.csv
    ├── matched_type_comparison.csv
    ├── unit_profiles_pooled.csv
    ├── eicu_decomposition_ci.csv
    ├── mimic_unit_ci.csv
    ├── check_gap30.csv
    ├── check_hospital_coverage.csv
    │
    │   stream selection (8, 9)
    ├── stream_distributions.csv
    ├── bottom_decile_agreement.csv
    ├── database_component_by_stream.csv
    ├── mimic_position_by_stream.csv
    ├── chance_comparator.csv
    │
    │   thresholds and eligibility (10, 11)
    ├── eicu_hospital_agreement.csv
    ├── mimic_unit_agreement.csv
    ├── eicu_hospital_group_shares.csv
    ├── mimic_unit_group_shares.csv
    ├── eicu_top_contributors.csv
    ├── eligibility_shares.csv
    ├── eligibility_by_hospital.csv
    ├── eligibility_count_eta2.csv
    ├── mixed_vpc.csv
    │
    │   admission hour (12)
    ├── mimic_hourly_or.csv
    ├── eicu_hourly_or.csv
    ├── mimic_by_careunit.csv
    ├── eicu_hospital_peaks.csv
    │
    │   replication (14, 15)
    ├── cross_vitals_summary.csv
    ├── table5_components.csv
    │
    │   severity harmonisation, reported as unsuccessful
    ├── mimic_sofa_coverage.csv
    ├── eicu_sofa_coverage.csv
    ├── acuity_coupling_r2.csv
    ├── residual_mortality_v2.csv
    │
    │   revision (17-24)
    ├── duplicate_timestamps.csv
    ├── covariate_missingness.csv
    ├── glmm_count_vpc.csv
    ├── ss_order_sensitivity.csv
    ├── excluded_stay_profile.csv
    ├── plausibility_thresholds.csv
    ├── cross_variable_hospital_medians.csv
    ├── eta2_restricted.csv
    ├── vpc_restricted.csv
    ├── threshold_restricted.csv
    ├── hospital_medians.csv
    ├── table3_cells.csv
    ├── table5_cells.csv
    ├── nb_glmm_vpc.csv
    │
    │   recorded hospital attributes (25)
    ├── attribute_coverage.csv
    ├── attribute_marginal_eta2.csv
    ├── attribute_conditional_eta2.csv
    ├── attribute_permutation.csv
    └── attribute_profiles.csv
```

Numbers in the tree refer to the script table above. Where a quantity is produced
by more than one script, `paper17_tables_final.py` (22) is authoritative for
Tables 3 and 5, and `paper17_stage4_restricted.py` (20) for the threshold
consequence.

The parquet caches (`hr_timestamps.parquet`, `nursecharting_offsets.parquet`,
`vitalperiodic_offsets.parquet`, `mimic.parquet`, `eicu_nc.parquet`,
`eicu_vp.parquet`) are derived from credentialed data and are excluded by
`.gitignore`.

---

## Citation

If you use this code, cite the manuscript. If you use the archived snapshot,
cite the version DOI from the Zenodo badge above.

## Licence

Code released under the MIT Licence (see `LICENSE`). The manuscript and figures
are separately licensed; see the journal version for terms.
