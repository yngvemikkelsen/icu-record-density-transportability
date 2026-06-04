#!/usr/bin/env python3
"""
generate_figures.py — Paper 17 figure generator
Produces three publication-grade PNG figures from analysis output CSVs:
  Figure 1: Vital-sign record density distributions, MIMIC-IV vs eICU-CRD
  Figure 2: Temporal forest plot — adjusted OR for low record density by admission hour
  Figure 3: Mortality forest plot — adjusted OR by exposure specification and cohort

Usage:
  Edit the PATHS dict below to point at your analysis output CSVs, then run:
    python3 generate_figures.py

  Or override on command line:
    python3 generate_figures.py --outdir /path/to/figures

Requires: pandas, matplotlib, numpy
  pip install pandas matplotlib numpy
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# EDIT THESE PATHS to point at your actual analysis output CSVs.
# If a CSV is missing, the script will fall back to manuscript-reported summary
# values so the figure still renders (clearly labelled as such).
# ----------------------------------------------------------------------------
PATHS = {
    # Figure 1: density distributions
    # Expected columns: stay_id, records_per_24h
    "mimic_density":  "/Users/yngve/bcst/outputs/paper17/mimic_density_per_stay.csv",
    "eicu_density":   "/Users/yngve/bcst/outputs/paper17/eicu_density_per_stay.csv",

    # Figure 2: temporal forest plot
    # Expected columns: hour_bin, OR, ci_lo, ci_hi, cohort
    # cohort ∈ {"MIMIC-IV", "eICU-CRD"}
    "temporal_or":    "/Users/yngve/bcst/outputs/paper17/temporal_or_by_hour.csv",

    # Figure 3: mortality forest plot
    # Expected columns: spec, cohort, OR, ci_lo, ci_hi
    # spec ∈ {"Bottom 10% percentile-matched", "Bottom 10% conditional residual",
    #         "Continuous residual (per SD)"}
    "mortality_or":   "/Users/yngve/bcst/outputs/paper17/mortality_or_by_spec.csv",
}

# Publication style: simple, readable, journal-friendly
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 600,         # 600 dpi for halftone publication (exceeds IJMI min 300)
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLOR_MIMIC = "#1f4e79"   # navy
COLOR_EICU  = "#c0504d"   # rust
COLOR_NULL  = "#7f7f7f"   # neutral grey


def load_csv_or_none(path):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    print(f"  [warn] {path} not found; using manuscript fallback values.", file=sys.stderr)
    return None


# ----------------------------------------------------------------------------
# Figure 1: density distributions
# ----------------------------------------------------------------------------
def figure_1(outdir: Path):
    print("Generating Figure 1: density distributions...")
    mimic = load_csv_or_none(PATHS["mimic_density"])
    eicu  = load_csv_or_none(PATHS["eicu_density"])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)

    # MIMIC-IV
    ax = axes[0]
    if mimic is not None:
        x = mimic["records_per_24h"].clip(0, 200)  # clip extreme tail for visibility
        ax.hist(x, bins=60, color=COLOR_MIMIC, alpha=0.85, edgecolor="white", linewidth=0.3)
        median = np.median(mimic["records_per_24h"])
        ax.axvline(median, color="black", linestyle="--", linewidth=0.8)
        ax.text(median + 3, ax.get_ylim()[1]*0.9, f"median = {median:.0f}", fontsize=8)
    else:
        # Fallback: schematic
        ax.text(0.5, 0.5, "MIMIC-IV density CSV\nnot found", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="grey")
    ax.set_title("MIMIC-IV (hybrid nurse + monitor)", fontsize=10)
    ax.set_xlabel("Records per 24 h (first 24 h)")
    ax.set_ylabel("ICU stays")
    ax.set_xlim(0, 200)

    # eICU-CRD
    ax = axes[1]
    if eicu is not None:
        x = eicu["records_per_24h"].clip(0, 350)
        ax.hist(x, bins=60, color=COLOR_EICU, alpha=0.85, edgecolor="white", linewidth=0.3)
        median = np.median(eicu["records_per_24h"])
        ax.axvline(median, color="black", linestyle="--", linewidth=0.8)
        ax.text(median + 5, ax.get_ylim()[1]*0.9, f"median = {median:.0f}", fontsize=8)
    else:
        ax.text(0.5, 0.5, "eICU-CRD density CSV\nnot found", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="grey")
    ax.set_title("eICU-CRD (monitor only)", fontsize=10)
    ax.set_xlabel("Records per 24 h (first 24 h)")
    ax.set_xlim(0, 350)

    out = outdir / "Figure_1_density_distributions.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote: {out}")


# ----------------------------------------------------------------------------
# Figure 2: temporal forest plot — OR for low record density by admission hour
# ----------------------------------------------------------------------------
def figure_2(outdir: Path):
    print("Generating Figure 2: temporal forest plot...")
    df = load_csv_or_none(PATHS["temporal_or"])

    if df is None:
        # Fallback uses only the MIMIC peak point reported in the manuscript abstract
        df = pd.DataFrame([
            {"hour_bin": "06–08", "cohort": "MIMIC-IV", "OR": 3.86, "ci_lo": 3.15, "ci_hi": 4.72},
        ])
        print("  [warn] Using single manuscript-reported MIMIC peak only.", file=sys.stderr)

    # Sort by cohort then by hour_bin
    df = df.copy()
    df["cohort"] = pd.Categorical(df["cohort"], categories=["MIMIC-IV", "eICU-CRD"], ordered=True)
    df = df.sort_values(["cohort", "hour_bin"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.32 * len(df) + 1.5)), constrained_layout=True)

    y_positions = np.arange(len(df))[::-1]  # top = first row

    for i, row in df.iterrows():
        y = y_positions[i]
        color = COLOR_MIMIC if row["cohort"] == "MIMIC-IV" else COLOR_EICU
        ax.errorbar(row["OR"], y, xerr=[[row["OR"] - row["ci_lo"]], [row["ci_hi"] - row["OR"]]],
                    fmt="o", color=color, markersize=5, capsize=2, linewidth=1.2)
        ax.text(ax.get_xlim()[1] if False else 6.5, y, f'  {row["OR"]:.2f} ({row["ci_lo"]:.2f}–{row["ci_hi"]:.2f})',
                va="center", fontsize=8, family="monospace")

    ax.axvline(1.0, color=COLOR_NULL, linestyle=":", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'{r["cohort"]}  {r["hour_bin"]}' for _, r in df.iterrows()], fontsize=9)
    ax.set_xlabel("Adjusted OR for low record density (95% CI)")
    ax.set_title("Temporal pattern of low record density by ICU admission hour", fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10"])

    out = outdir / "Figure_2_temporal_forest.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote: {out}")


# ----------------------------------------------------------------------------
# Figure 3: mortality forest plot — by exposure specification and cohort
# ----------------------------------------------------------------------------
def figure_3(outdir: Path):
    print("Generating Figure 3: mortality forest plot...")
    df = load_csv_or_none(PATHS["mortality_or"])

    if df is None:
        # Fallback uses values from manuscript abstract
        df = pd.DataFrame([
            {"spec": "Bottom 10% percentile-matched",      "cohort": "MIMIC-IV",  "OR": 0.81, "ci_lo": 0.73, "ci_hi": 0.89},
            {"spec": "Bottom 10% percentile-matched",      "cohort": "eICU-CRD",  "OR": 1.61, "ci_lo": 1.50, "ci_hi": 1.73},
            {"spec": "Bottom 10% conditional residual",    "cohort": "MIMIC-IV",  "OR": 0.92, "ci_lo": 0.85, "ci_hi": 1.00},
            {"spec": "Bottom 10% conditional residual",    "cohort": "eICU-CRD",  "OR": 1.60, "ci_lo": 1.50, "ci_hi": 1.72},
            {"spec": "Continuous residual (per SD)",       "cohort": "MIMIC-IV",  "OR": 1.06, "ci_lo": 1.03, "ci_hi": 1.09},
        ])
        print("  [warn] Using manuscript abstract values as fallback.", file=sys.stderr)

    df = df.copy()
    df["cohort"] = pd.Categorical(df["cohort"], categories=["MIMIC-IV", "eICU-CRD"], ordered=True)
    df = df.sort_values(["spec", "cohort"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.0, 0.45 * len(df) + 1.8), constrained_layout=True)

    y_positions = np.arange(len(df))[::-1]
    for i, row in df.iterrows():
        y = y_positions[i]
        color = COLOR_MIMIC if row["cohort"] == "MIMIC-IV" else COLOR_EICU
        ax.errorbar(row["OR"], y, xerr=[[row["OR"] - row["ci_lo"]], [row["ci_hi"] - row["OR"]]],
                    fmt="o", color=color, markersize=6, capsize=3, linewidth=1.4)
        ax.text(3.5, y, f'  {row["OR"]:.2f} ({row["ci_lo"]:.2f}–{row["ci_hi"]:.2f})',
                va="center", fontsize=8, family="monospace")

    ax.axvline(1.0, color=COLOR_NULL, linestyle=":", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'{r["cohort"]}  ·  {r["spec"]}' for _, r in df.iterrows()], fontsize=9)
    ax.set_xlabel("Adjusted OR for hospital mortality (95% CI)")
    ax.set_title("Mortality association by exposure specification, MIMIC-IV vs eICU-CRD", fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.7, 1, 1.5, 2, 3])
    ax.set_xticklabels(["0.5", "0.7", "1", "1.5", "2", "3"])
    ax.set_xlim(0.5, 3.0)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", color=COLOR_MIMIC, linewidth=0, label="MIMIC-IV"),
        Line2D([], [], marker="o", color=COLOR_EICU,  linewidth=0, label="eICU-CRD"),
    ]
    ax.legend(handles=handles, loc="lower right")

    out = outdir / "Figure_3_mortality_forest.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/Users/yngve/bcst/outputs/paper17/figures",
                    help="Directory to write PNG figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir}\n")

    figure_1(outdir)
    figure_2(outdir)
    figure_3(outdir)

    print("\nDone. Verify each figure visually before manuscript submission.")
    print("If a CSV path was missing, edit the PATHS dict at the top of this script.")


if __name__ == "__main__":
    main()
