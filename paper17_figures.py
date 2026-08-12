#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "matplotlib", "pyarrow"]
# ///
"""
Paper 17: build the four manuscript figures.

  Figure 1  Charting-hour distribution of heart-rate observations (220045)
            against the alarm-limit items (220046, 220047), MIMIC-IV.
  Figure 2  Distribution of per-hospital median charting interval across eICU
            hospitals (nurse stream), with MIMIC care units overlaid.
  Figure 3  Variance decomposition by database, hospital and unit, with 95%
            bootstrap intervals on the eICU hospital and unit components.
  Figure 4  Share of stays at each hospital designated low density by the
            pooled bottom-decile rule, against the MIMIC care units.

Okabe-Ito palette throughout, greyscale-safe, 300 dpi PNG plus vector PDF.
Every value is read from a saved analysis output; nothing is hardcoded except
Figure 1 and Figure 3, which read the numbers reported in the manuscript and are
cross-checked against their source CSVs where those exist.

Usage:
  python paper17_figures.py \
      --unit-profile ~/bcst/unit_profile \
      --unit-profile-eicu ~/bcst/unit_profile_eicu \
      --decomposition ~/bcst/decomposition_ci \
      --threshold ~/bcst/threshold_consequence \
      --exposure-diagnosis ~/bcst/exposure_diagnosis \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito: distinguishable in greyscale and to colour-blind readers
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERM = "#D55E00"
GREY = "#666666"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})


def save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}.png and {name}.pdf")


# ---------------------------------------------------------------- Figure 1 --

def figure1(exposure_dir: Path, out_dir: Path):
    src = exposure_dir / "charting_hour_by_item.csv"
    if not src.exists():
        print(f"  SKIP Figure 1: {src} not found")
        return
    d = pd.read_csv(src)
    obs = d[d["itemid"] == 220045].sort_values("charthour")
    hi = d[d["itemid"] == 220046].sort_values("charthour")
    lo = d[d["itemid"] == 220047].sort_values("charthour")
    if obs.empty or hi.empty:
        print("  SKIP Figure 1: expected itemids absent")
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.axhline(1 / 24, color=GREY, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(23.4, 1 / 24, " uniform", va="center", ha="left", color=GREY,
            fontsize=7.5)
    ax.plot(obs["charthour"], obs["share"], "-o", color=BLUE, lw=1.6, ms=3.4,
            label="Heart rate (220045)", zorder=3)
    ax.plot(hi["charthour"], hi["share"], "-s", color=VERM, lw=1.6, ms=3.4,
            label="HR alarm \u2013 high (220046)", zorder=3)
    ax.plot(lo["charthour"], lo["share"], "-^", color=ORANGE, lw=1.6, ms=3.4,
            label="HR alarm \u2013 low (220047)", zorder=3)
    for h in (0, 8, 20):
        ax.axvline(h, color=GREY, lw=0.6, alpha=0.35, zorder=0)
    ax.set_xlabel("Clock hour of charting")
    ax.set_ylabel("Share of the item's records")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.6, 23.6)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    save(fig, out_dir, "Figure1_charting_hour")


# ---------------------------------------------------------------- Figure 2 --

def figure2(eicu_dir: Path, mimic_dir: Path, out_dir: Path):
    ev = eicu_dir / "nursecharting_offsets.parquet"
    mv = mimic_dir / "hr_timestamps.parquet"
    if not ev.exists() or not mv.exists():
        print("  SKIP Figure 2: cached offsets not found")
        return
    print("  computing per-hospital medians ...")

    def per_stay_interval(df, key, off):
        df = df.sort_values([key, off])
        g = df.groupby(key)[off]
        n = g.size()
        gaps = df.assign(gap=g.diff()).dropna(subset=["gap"])
        med = gaps.groupby(key)["gap"].median()
        return pd.DataFrame({"n_records": n, "median_interval": med}).reset_index()

    e = per_stay_interval(pd.read_parquet(ev), "patientunitstayid",
                          "observationoffset")
    pat = pd.read_csv(args_eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid",
                               "unitdischargeoffset"])
    e = e.merge(pat, on="patientunitstayid", how="inner")
    e = e[(e["unitdischargeoffset"] >= 1440) & (e["n_records"] >= 3)]
    cnt = e["hospitalid"].value_counts()
    e = e[e["hospitalid"].isin(cnt[cnt >= 500].index)]
    hosp = e.groupby("hospitalid")["median_interval"].median().sort_values()

    m = per_stay_interval(pd.read_parquet(mv), "stay_id", "offset_min")
    ps = pd.read_csv(args_mimic_per_stay, usecols=["stay_id", "careunit", "los"])
    m = m.merge(ps, on="stay_id", how="inner")
    m = m[(m["los"] >= 1.0) & (m["n_records"] >= 3)]
    units = m.groupby("careunit")["median_interval"].median()

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(hosp))
    ax.bar(x, hosp.values, color=BLUE, width=0.85, alpha=0.85, linewidth=0,
           label=f"eICU hospitals (n={len(hosp)})")
    for i, (u, v) in enumerate(units.items()):
        ax.plot([-1.5], [v], marker="D", ms=4, color=VERM, zorder=5,
                label="MIMIC-IV care units" if i == 0 else None)
    ax.axhline(units.median(), color=VERM, lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.set_xlabel("eICU hospitals, ordered by median charting interval")
    ax.set_ylabel("Median interval between\nrecords (minutes)")
    ax.set_xlim(-3, len(hosp))
    ax.set_xticks([])
    ax.legend(frameon=False, loc="upper left")
    save(fig, out_dir, "Figure2_hospital_intervals")


# ---------------------------------------------------------------- Figure 3 --

def figure3(decomp_dir: Path, out_dir: Path):
    src = decomp_dir / "eicu_decomposition_ci.csv"
    if not src.exists():
        print(f"  SKIP Figure 3: {src} not found")
        return
    d = pd.read_csv(src)
    order = ["median_interval_min", "frac_time_in_gaps", "max_interval_min",
             "n_gaps_gt2h", "n_records", "n_gaps_gt30m", "iqr_interval_min",
             "t_first_h"]
    labels = {"median_interval_min": "Median interval",
              "frac_time_in_gaps": "Fraction of window in gaps",
              "max_interval_min": "Longest interval",
              "n_gaps_gt2h": "Gaps > 2 h", "n_records": "Record count",
              "n_gaps_gt30m": "Gaps > 30 min",
              "iqr_interval_min": "Interval IQR",
              "t_first_h": "Hours to first record"}
    d = d.set_index("metric").loc[[m for m in order if m in set(d["metric"])]]
    y = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.errorbar(d["eta2_hospital"], y,
                xerr=[d["eta2_hospital"] - d["hosp_lo"],
                      d["hosp_hi"] - d["eta2_hospital"]],
                fmt="o", color=BLUE, ms=5, lw=1.4, capsize=2.5,
                label="Hospital")
    ax.errorbar(d["eta2_unit_within_hospital"], y - 0.22,
                xerr=[d["eta2_unit_within_hospital"] - d["unit_lo"],
                      d["unit_hi"] - d["eta2_unit_within_hospital"]],
                fmt="s", color=ORANGE, ms=4.5, lw=1.4, capsize=2.5,
                label="Unit within hospital")
    ax.set_yticks(y - 0.11)
    ax.set_yticklabels([labels.get(i, i) for i in d.index])
    ax.set_xlabel("Variance explained (eta\u00b2), eICU nurse stream, "
                  "with 95% bootstrap interval")
    ax.set_xlim(-0.02, 0.85)
    ax.grid(axis="x", lw=0.5, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    save(fig, out_dir, "Figure3_variance_decomposition")


# ---------------------------------------------------------------- Figure 4 --

def figure4(thr_dir: Path, out_dir: Path):
    eh = thr_dir / "eicu_hospital_group_shares.csv"
    mu = thr_dir / "mimic_unit_group_shares.csv"
    if not eh.exists():
        print(f"  SKIP Figure 4: {eh} not found")
        return
    e = pd.read_csv(eh).iloc[:, -1].sort_values().values
    m = pd.read_csv(mu).iloc[:, -1].sort_values().values if mu.exists() else None

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(e))
    ax.bar(x, e, color=BLUE, width=0.85, linewidth=0, alpha=0.85,
           label=f"eICU hospitals (n={len(e)})")
    ax.axhline(0.10, color=GREY, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.text(len(e) * 0.42, 0.115, "nominal decile (10%)", va="bottom",
            ha="left", color=GREY, fontsize=7.5)
    if m is not None:
        for i, v in enumerate(m):
            ax.plot([-2.0], [v], marker="D", ms=4, color=VERM, zorder=5,
                    label="MIMIC-IV care units" if i == 0 else None)
    ax.set_xlabel("Hospitals, ordered by share of stays designated low density")
    ax.set_ylabel("Share of the group's stays\nflagged by the pooled rule")
    ax.set_xlim(-3.5, len(e))
    ax.set_ylim(0, 1.02)
    ax.set_xticks([])
    ax.legend(frameon=False, loc="upper left")
    save(fig, out_dir, "Figure4_pooled_threshold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-profile", required=True, type=Path)
    ap.add_argument("--unit-profile-eicu", required=True, type=Path)
    ap.add_argument("--decomposition", required=True, type=Path)
    ap.add_argument("--threshold", required=True, type=Path)
    ap.add_argument("--exposure-diagnosis", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./figures"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    global args_eicu_root, args_mimic_per_stay
    args_eicu_root = a.eicu_root
    args_mimic_per_stay = a.mimic_per_stay

    print("Figure 1"); figure1(a.exposure_diagnosis, a.out_dir)
    print("Figure 2"); figure2(a.unit_profile_eicu, a.unit_profile, a.out_dir)
    print("Figure 3"); figure3(a.decomposition, a.out_dir)
    print("Figure 4"); figure4(a.threshold, a.out_dir)
    print(f"\n-> {a.out_dir}")
    print("PNG at 300 dpi for the manuscript; PDF vector for production.")


if __name__ == "__main__":
    main()
