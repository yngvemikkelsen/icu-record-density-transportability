# ICU vital-sign record density: source-stream and site-level variation

Analysis code for:

> Mikkelsen Y. Data-Stream and Hospital-Level Variation in ICU Vital-Sign Record
> Density Across MIMIC-IV and eICU-CRD: Retrospective Data Quality Study.
> *JMIR Medical Informatics* (submitted 2026).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21901638.svg)](https://doi.org/10.5281/zenodo.21901638)

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
hospitals rather than patients: five of 68 hospitals supply half of everything it
flags.

No patient-outcome association is interpreted as a study finding. The reason is
given in the manuscript: no severity instrument can be harmonised across the two
cohorts without either sharing the exposure's data stream or discarding a quarter
of one cohort in a manner correlated with the exposure.

---

## Data

Neither dataset is redistributed here. Both require credentialed access through
PhysioNet under their respective data use agreements:

- **MIMIC-IV v3.1** — https://physionet.org/content/mimiciv/3.1/
- **eICU-CRD v2.0** — https://physionet.org/content/eicu-crd/2.0/

Scripts take `--mimic-root` and `--eicu-root` pointing at local extractions.
Nothing in `results/` contains patient-level data; all outputs are aggregate.

---

## Scripts, in the order they were run

Each maps to the manuscript section it produces. Most cache intermediate
extractions to parquet, so reruns after the first are fast.

| # | Script | Produces |
|---|---|---|
| 1 | `paper17_diagnose_exposure.py` | Exposure composition: which itemids reconstruct the conventional count, and their charting-hour concentration (Table 1, Figure 1) |
| 2 | `paper17_reconcile_counts.py` | Reconciles the three competing definitions of "heart-rate records in the first 24 h" |
| 3 | `paper17_build_physiology_v2.py` | First-24 h median/IQR physiology summaries, both cohorts, avoiding order statistics |
| 4 | `paper17_unit_documentation_profile.py` | MIMIC-IV documentation profile by care unit |
| 5 | `paper17_eicu_documentation_profile.py` | eICU-CRD profile by unit type and by hospital, both source streams |
| 6 | `paper17_pooled_unit_comparison.py` | Pooled decomposition and unit-by-unit positioning of MIMIC-IV against eICU-CRD |
| 7 | `paper17_decomposition_ci.py` | Hospital-clustered bootstrap on the decomposition; mixed-effects variance partition (Table 3, Figure 2) |
| 8 | `paper17_stream_selection.py` | Nurse-versus-monitor stream comparison (Table 2) |
| 9 | `paper17_chance_comparator.py` | Correct chance comparator for stream agreement under integer ties |
| 10 | `paper17_threshold_consequence.py` | Pooled versus site-specific low-density thresholds (Figure 4) |
| 11 | `paper17_sensitivity.py` | Eligibility sensitivity and mixed-model variance partition |
| 12 | `paper17_temporal_rerun.py` | Admission-hour association under four specifications (Table 4) |
| 13 | `paper17_probe_vitals.py` | Verifies itemids and value-name strings for the replication variables |
| 14 | `paper17_cross_vitals.py` | Replication in oxygen saturation and respiratory rate |
| 15 | `paper17_table5_harmonise.py` | Puts the replication variance components on one cohort basis (Table 5) |
| 16 | `paper17_figures.py` | Figures 1–4 |

Three further scripts document the severity-harmonisation attempt reported in the
manuscript as unsuccessful. They are included because the negative result is part
of the argument:

| Script | Purpose |
|---|---|
| `paper17_probe_sofa_coverage.py` | Coverage of candidate severity components in both cohorts |
| `paper17_build_severity_v3.py` | Harmonised severity components, worst-value rule stated in the header |
| `bcst_residualization_v2.py` | Residualisation and outcome models under three adjustment regimes |

---

## Running

Scripts use [PEP 723](https://peps.python.org/pep-0723/) inline dependency
metadata and are intended to be run with [uv](https://docs.astral.sh/uv/):

```bash
uv run scripts/paper17_diagnose_exposure.py \
    --mimic-root ~/physionet.org/files/mimiciv/3.1 \
    --mimic-per-stay path/to/per_stay.csv \
    --out-dir out/exposure
```

Each script prints its own usage. Dependencies are pandas, numpy, statsmodels,
pyarrow and matplotlib; no script requires a GPU.

The heavy passes are `paper17_build_physiology_v2.py` (one chartevents pass, one
vitalPeriodic pass) and `paper17_cross_vitals.py` (three passes). Everything else
runs from cached parquet in minutes.

---

## Reproducibility notes

Three points that matter for anyone re-running or extending this:

**Exposure composition is not obvious from the itemid.** The conventional MIMIC-IV
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

**Analysis sets differ by design and are named in the code.** The pooled
decomposition uses all eligible stays from both cohorts. The eICU-CRD
hospital-clustered bootstrap resamples every hospital contributing eligible
stays. The mixed model and per-hospital summaries use hospitals contributing at
least 500 nurse-stream stays. Unit-level profiles use hospital-by-unit-type cells
with at least 50 stays. Estimates from different sets are not interchangeable.

---

## Repository contents

```
scripts/    analysis code, numbered above
figures/    Figures 1-4 as PNG (300 dpi) and PDF
results/    aggregate outputs: decomposition estimates, bootstrap intervals,
            threshold agreement tables, replication summaries
```

---

## Citation

If you use this code, cite the manuscript. If you use the archived snapshot,
cite the version DOI from the Zenodo badge above.

## Licence

Code released under the MIT Licence (see `LICENSE`). The manuscript and figures
are separately licensed; see the journal version for terms.
