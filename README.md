# mimic-eicu-record-density

Analysis code for:

**Mikkelsen Y.** Transportability of ICU vital-sign record density as an EHR-derived process measure across MIMIC-IV and eICU-CRD. *International Journal of Medical Informatics* (under review, 2026).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## What this is

This repository contains the analysis code used to produce the results reported in the manuscript above. The study tests whether ICU vital-sign record density — a candidate EHR-derived process measure — retains consistent clinical meaning when measured across recording systems with different data-generating architectures (MIMIC-IV hybrid nurse-plus-monitor stream vs eICU-CRD monitor-only stream).

Three cross-cohort questions are addressed:

1. Does cyclic temporal structure in vital-sign record density replicate across cohorts?
2. Is low record density associated with mortality consistently across cohorts at corresponding operationalisations?
3. Do the two recording streams index the same underlying clinical construct, as measured by acuity-coupling strength?

The principal finding: cyclic temporal structure replicates reproducibly, but the mortality-relevant interpretation of the measure does not transport across recording streams.

## Data access

**This repository contains analysis code only. It does not contain patient data.**

The underlying datasets are subject to PhysioNet credentialed access and **must not be redistributed**:

- MIMIC-IV v3.1: <https://physionet.org/content/mimiciv/3.1/>
- eICU Collaborative Research Database v2.0: <https://physionet.org/content/eicu-crd/2.0/>

To reproduce the analyses you must obtain your own PhysioNet credentialed access (which requires completion of CITI training in human subjects research) and place the downloaded files at the paths configured in `cohort/config.yaml`.

## Repository structure

```
mimic-eicu-record-density/
├── cohort/                    Cohort construction scripts
│   ├── build_mimic_cohort.py
│   ├── build_eicu_cohort.py
│   ├── config.yaml            Data paths and exclusion thresholds
│   └── README.md              Exclusion criteria, expected row counts
├── analysis/                  Statistical analyses
│   ├── temporal_pattern.py    Cyclic admission-hour structure (Table 2, Figure 2)
│   ├── mortality_models.py    Three exposure specifications (Table 3, Figure 3)
│   ├── acuity_coupling.py     R² diagnostic (Table 4)
│   ├── negative_controls.py   CVICU, weekend, recent-era falsification (Supp S1)
│   └── threshold_sensitivity.py  Per-hospital threshold sweep (Supp S2)
├── figures/                   PNG figure generation
│   └── generate_figures.py
├── docs/
│   └── reproducibility.md     Step-by-step replication guide
├── outputs/                   Generated locally; not tracked in git
├── requirements.txt           Pinned Python dependencies
├── LICENSE                    MIT
├── CITATION.cff               Machine-readable citation metadata
└── README.md                  This file
```

## Software requirements

- Python ≥ 3.10
- See `requirements.txt` for pinned versions

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The analyses were developed and run on macOS 26 (Tahoe) on an Apple Silicon M2 Max with 64 GB RAM. They do not require a GPU. Full reproduction takes approximately 30–45 minutes wall-clock on the reference hardware.

## Reproducing the manuscript results

1. Obtain PhysioNet credentialed access and download MIMIC-IV v3.1 and eICU-CRD v2.0
2. Edit `cohort/config.yaml` to point at your data paths
3. Build cohorts: `python3 cohort/build_mimic_cohort.py && python3 cohort/build_eicu_cohort.py`
4. Run analyses: `python3 analysis/temporal_pattern.py && python3 analysis/mortality_models.py && python3 analysis/acuity_coupling.py`
5. Run negative controls and sensitivity sweep: `python3 analysis/negative_controls.py && python3 analysis/threshold_sensitivity.py`
6. Generate figures: `python3 figures/generate_figures.py`

See `docs/reproducibility.md` for the full step-by-step guide including expected output row counts at each stage.

## Versioning

The version archived at Zenodo under the DOI above corresponds to the manuscript-of-record. Subsequent commits to `main` reflect post-publication maintenance and may differ from the published-version code. Use the Zenodo DOI to cite the exact version associated with the paper.

## Citation

If you use this code or build on the methods, please cite the paper:

> Mikkelsen Y. Transportability of ICU vital-sign record density as an EHR-derived process measure across MIMIC-IV and eICU-CRD. *International Journal of Medical Informatics*. 2026 (in press). doi:[insert at acceptance]

You may also cite the archived code directly via the Zenodo DOI above. A machine-readable citation is available in [`CITATION.cff`](CITATION.cff).

## License

This code is released under the MIT License. See [`LICENSE`](LICENSE).

The data licenses for MIMIC-IV and eICU-CRD are governed by their respective PhysioNet credentialed-access terms and are not affected by this code license.

## Acknowledgments

The author thanks the PhysioNet team and the original contributors to MIMIC-IV (Beth Israel Deaconess Medical Center; MIT Laboratory for Computational Physiology) and eICU-CRD (Philips Healthcare; MIT Laboratory for Computational Physiology) for making these datasets publicly available for secondary research use.

This work was carried out as part of the author's Postgraduate Diploma in Artificial Intelligence for Business at Saïd Business School, University of Oxford.

## Contact

Yngve Mikkelsen, MD, MSc, DBA
Saïd Business School, University of Oxford
[insert email]

For questions about reproducing the analyses, please open a GitHub issue.
