# Statistics supplied at copyediting

At copyediting, JMIR asked for interquartile ranges, standard deviations and absolute
numerators and denominators (n/N) beside values that the accepted manuscript reported as
point estimates. Most of them could not be read from the files in `results/`, which hold
point estimates, so they were computed after acceptance, on 21 October 2026, by the scripts
in `scripts/copyedit/`. This folder holds what those runs wrote and the console log of each
run. Release 1.1.4, which the published article cites, predates this folder; release 1.1.5
adds it and changes no earlier script or output.

Each script rebuilds the analysis set it needs and prints that set's size beside the
published size before computing anything on it, and prints every statistic beside the value
the manuscript then gave: the `manuscript_says` column in the CSVs, `[manuscript: ...]` in
the logs. A statistic was supplied only where its analysis set reproduced.

## Runs

| Run | Script | Output | Status |
|---|---|---|---|
| v2 | `paper17_copyedit_stats_v2.py` | `copyedit_statistics_v2.csv`, `mimic_unit_dispersion.csv`, `attribute_dispersion_numbedscategory.csv`, `attribute_dispersion_region.csv`, `attribute_dispersion_teachingstatus.csv` | All six sections (A to F) reproduced their analysis sets and point estimates; used |
| v3 | `paper17_copyedit_stats_v3.py` | `copyedit_statistics_v3.csv` | The eICU-CRD rows and the MIMIC-IV retained row of section 1, and section 4, used. Superseded by v5: the MIMIC-IV excluded row of section 1, whose set was built from the heart-rate cache and so held 12 stays instead of 84, and section 3, whose item scan was not restricted to the first 24 hours. Section 2 reports medians over stays, which is not the quantity the comparator turned out to be |
| v5 | `paper17_copyedit_stats_v5.py` | `copyedit_statistics_v5.csv` | Used. Section 1 reproduces the 72, 12 and 84 split of excluded MIMIC-IV stays; section 3 reproduces 2,970,732 in-window records and the per-item stay counts in `../per_item_volume.csv`. Its reconstruction set has 85,138 stays against the 85,133 the manuscript states |
| v6 | `paper17_copyedit_stats_v6.py` | `copyedit_statistics_v6.csv` | Section A computes the comparator on nine candidate analysis sets; none returns the 26 the accepted manuscript gave. Section B only lists local severity-build and coverage-probe outputs; nothing in it was supplied |
| v7 | `paper17_copyedit_stats_v7.py` | none | Failed: it read `electivesurgery` from the patient table, which does not carry that column. Retained with its log |
| v8 | `paper17_copyedit_stats_v8.py` | `copyedit_statistics_v8.csv` | Numerators for urine output and elective surgery used. The shares in the CSV are on the 200,859-stay patient table and are not the published shares; see below |
| v9 | `paper17_copyedit_stats_v9.py` | `copyedit_statistics_v9.csv` | Share of stays without an `apacheApsVar` row, on each analysis set; the same counts as `../apache_enrichment.csv` |

`logs/` holds the console output of each run, v7 included. The first, unversioned run
computed sections A to C on every cached stay instead of the analysis sets; v2 replaced it
and its output is not deposited. There is no v4 run.

## Where each supplied value comes from

| Values supplied | Source |
|---|---|
| Stream comparison: records per 24 hours as mean (SD) and median (IQR), and median charting interval (IQR), for MIMIC-IV chartevents and the eICU-CRD nurseCharting and vitalPeriodic streams | `copyedit_statistics_v2.csv`, section A |
| MIMIC-IV cohort construction: age and ICU length of stay, removed against retained stays | `copyedit_statistics_v2.csv`, section B |
| MIMIC-IV care units: median charting interval, longest interval, gaps beyond 2 hours, record count and hours to first record, each with IQR | `mimic_unit_dispersion.csv`; pooled rows in `copyedit_statistics_v2.csv`, section C |
| The eight low-density eICU-CRD hospitals: median record count (IQR) and below-three-record share (n/N); the cohort median of that share | `copyedit_statistics_v2.csv`, section D |
| Restricted cohort mean record count, 39.01 (SD 23.3) | `copyedit_statistics_v2.csv`, section D |
| Recorded hospital attributes: record count and median interval, median (IQR), by bed-capacity category, teaching status and region | `attribute_dispersion_*.csv`; `copyedit_statistics_v2.csv`, section E |
| eICU-CRD denominators: the three-record minimum, 13,583/132,900, of which 8,934 with no nurse-charted record and 4,649 with one or two; single-unit-type hospitals, 26/60, 30/68 and 151/193 | `copyedit_statistics_v2.csv`, section F |
| Excluded against retained ICU length of stay: MIMIC-IV 6.53 (IQR 3.42-13.11) against 2.41 (IQR 1.57-4.48) | `copyedit_statistics_v5.csv`, section 1 |
| Excluded against retained ICU length of stay: eICU-CRD 2.20 (IQR 1.56-3.87) against 2.33 (IQR 1.59-4.08) | `copyedit_statistics_v3.csv`, section 1 |
| MIMIC-IV item profile in the first 24 hours: mean (SD) records per stay for the observation item, each alarm-limit item and the pooled count, on the analysis set and the reconstruction set | `copyedit_statistics_v5.csv`, section 3 |
| Gaps beyond 2 hours: 65,301/88,560 restricted-cohort stays (73.7%) have none | `copyedit_statistics_v3.csv`, section 4 |
| The comparator for hospital median record counts: the median of per-hospital median counts across the 68 hospitals contributing at least 500 eligible stays, 32 (IQR 25-45) | `copyedit_statistics_v5.csv`, section 2, and `copyedit_statistics_v6.csv`, section A; also `../hospital_medians.csv`, where the median is 31.5 before rounding and the quartiles 24.75 and 45 |
| Severity coverage: urine output 84,062/171,177 (49.1%) and elective-surgery status 33,256/171,177 (19.4%) of eICU-CRD stays | Numerators from `copyedit_statistics_v8.csv`. The denominator is the 171,177 stays with an `apacheApsVar` row, equal to the number of rows in the severity build, as printed in `logs/copyedit_stats_v8.log` |
| APACHE physiology table unavailable for 8,627/88,560 restricted-cohort stays (9.7%) | `copyedit_statistics_v9.csv`; `../apache_enrichment.csv` |

The remaining n/N values supplied at copyediting come from files that were already in
`results/` before this release:

- `chance_comparator.csv`, heart rate at the 10th percentile: 12,017 monitor-flagged and
  12,367 nurse-flagged of 117,572 stays, 1,274 flagged by both.
- `check_gap30.csv`: intervals of 55 to 65 minutes among gaps exceeding 30 minutes,
  1,268,105/1,379,521 in MIMIC-IV and 1,168,014/1,574,690 in eICU-CRD.
- `mimic_unit_agreement.csv`, 10th percentile: 3,789/66,911 stays reclassified.
- `eicu_hospital_peaks.csv`: 49/71 hospitals with their peak admission hour before 14:00.
- `mimic_sofa_coverage.csv` and `eicu_sofa_coverage.csv`: each coverage share times its
  denominator, 94,458 MIMIC-IV ICU stays and 200,859 eICU-CRD stays, gives an exact count,
  for example bilirubin 41,126/94,458 and 74,584/200,859. The two denominators are printed
  in `logs/copyedit_stats_v6.log`.

## One published value that this output does not reproduce

The published article gives the cohort median of the below-three-record share as 0.5%
(IQR 0.1%-7.3%). The source row, `all 193 eligible-stay hospitals` in section D of
`copyedit_statistics_v2.csv`, gives 0.51 (IQR 0.00-7.32). The lower quartile as printed
does not reproduce from this output.
