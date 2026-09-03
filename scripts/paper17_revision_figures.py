#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "matplotlib", "pyarrow"]
# ///
"""
Paper 17 revision figures.

Figure 2 and Figure 4 were produced on the unrestricted eICU-CRD cohort and are
regenerated here on the restricted primary cohort. Figure 5 is the participant
flow diagram requested by the editor.

Figures 1 and 3 are unchanged: Figure 1 is the MIMIC-IV charting-hour
distribution, which the restriction does not touch, and Figure 3 is the
per-hospital interval distribution, which is regenerated here as well because
its range changes when the eight hospitals are removed.

All output is written at 184 dpi so that a 6.5-inch figure lands within the
1200-pixel limit, with a vector PDF alongside.

Usage:
  python paper17_revision_figures.py \
      --tables-dir ~/bcst/revision_tables \
      --stage4-dir ~/bcst/revision_stage4 \
      --stage3-dir ~/bcst/revision_stage3 \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --mimic-cache ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/revision_figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREEN, VERM, GREY = ("#0072B2", "#E69F00", "#009E73",
                                   "#D55E00", "#666666")
DPI = 184
WINDOW_MIN = 24 * 60
MIN_RECORDS = 3
MIN_HOSP = 500
FLOOR = 12

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 300,
})

ORDER = ["Record count", "Median interval", "Interval IQR",
         "Longest interval", "Gaps >30 min", "Gaps >2 h",
         "Fraction of window in gaps", "Hours to first record"]


def save(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    try:
        from PIL import Image
        w, h = Image.open(out_dir / f"{name}.png").size
        flag = "" if max(w, h) <= 1200 else "  *** OVER 1200 px"
        print(f"  {name}: {w}x{h}{flag}")
    except ImportError:
        print(f"  {name} written")


def fig2(a, out_dir):
    """Variance components, restricted primary cohort."""
    t = pd.read_csv(a.tables_dir / "table3_cells.csv")
    r = t[t["cohort"] == "restricted"].set_index("metric")
    s4 = a.stage4_dir / "eta2_restricted.csv"
    unit = None
    if s4.exists():
        u = pd.read_csv(s4)
        unit = u[u["cohort"] == "restricted"].set_index("metric")
    keep = [m for m in ORDER if m in r.index]
    y = np.arange(len(keep))[::-1]

    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.errorbar(r.loc[keep, "eicu_eta2"], y,
                xerr=[r.loc[keep, "eicu_eta2"] - r.loc[keep, "eta2_lo"],
                      r.loc[keep, "eta2_hi"] - r.loc[keep, "eicu_eta2"]],
                fmt="o", color=BLUE, ms=5, lw=1.4, capsize=2.5,
                label="Hospital")
    if unit is not None:
        k2 = [m for m in keep if m in unit.index]
        y2 = [y[keep.index(m)] - 0.24 for m in k2]
        ax.errorbar(unit.loc[k2, "eta2_unit"], y2,
                    xerr=[unit.loc[k2, "eta2_unit"] - unit.loc[k2, "unit_lo"],
                          unit.loc[k2, "unit_hi"] - unit.loc[k2, "eta2_unit"]],
                    fmt="s", color=ORANGE, ms=4.5, lw=1.4, capsize=2.5,
                    label="Unit within hospital")
    ax.set_yticks(y - 0.12)
    ax.set_yticklabels(keep)
    ax.set_xlabel("Variance explained (η²), eICU-CRD nurse stream,\n"
                  "restricted cohort, with 95% bootstrap interval")
    ax.set_xlim(-0.02, max(0.5, float(r.loc[keep, "eta2_hi"].max()) + 0.05))
    ax.grid(axis="x", lw=0.5, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    save(fig, out_dir, "Figure2_variance_decomposition")


def hospital_medians(a):
    ev = pd.read_parquet(a.eicu_nc_cache).sort_values(
        ["patientunitstayid", "observationoffset"])
    g = ev.groupby("patientunitstayid")["observationoffset"]
    d = pd.DataFrame({"n_records": g.size()})
    d["median_interval_min"] = (ev.assign(gap=g.diff())
                                  .dropna(subset=["gap"])
                                  .groupby("patientunitstayid")["gap"].median())
    d = d.reset_index()
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid",
                               "unitdischargeoffset"])
    d = d.merge(pat, on="patientunitstayid", how="inner")
    d = d[(d["unitdischargeoffset"] >= WINDOW_MIN)
          & (d["n_records"] >= MIN_RECORDS)]
    c = d["hospitalid"].value_counts()
    d = d[d["hospitalid"].isin(c[c >= MIN_HOSP].index)]
    med_c = d.groupby("hospitalid")["n_records"].median()
    med_i = d.groupby("hospitalid")["median_interval_min"].median()
    keep = med_c[med_c >= FLOOR].index
    return med_i, keep, d


def fig3(a, out_dir):
    """Per-hospital interval distribution, restricted, with MIMIC overlaid."""
    med_i, keep, _ = hospital_medians(a)
    excl = med_i.drop(index=keep, errors="ignore").sort_values()
    inc = med_i.loc[keep].sort_values()

    ev = pd.read_parquet(a.mimic_cache).sort_values(["stay_id", "offset_min"])
    g = ev.groupby("stay_id")["offset_min"]
    m = pd.DataFrame({"n": g.size()})
    m["mi"] = (ev.assign(gap=g.diff()).dropna(subset=["gap"])
                 .groupby("stay_id")["gap"].median())
    m = m.reset_index().merge(
        pd.read_csv(a.mimic_per_stay, usecols=["stay_id", "careunit", "los"]),
        on="stay_id", how="inner")
    m = m[(m["los"] >= 1.0) & (m["n"] >= MIN_RECORDS)]
    units = m.groupby("careunit")["mi"].median()

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(inc))
    ax.bar(x, inc.values, color=BLUE, width=0.85, lw=0, alpha=0.9,
           label=f"eICU-CRD hospitals, restricted (n={len(inc)})")
    xe = np.arange(len(inc), len(inc) + len(excl))
    ax.bar(xe, excl.values, color=GREY, width=0.85, lw=0, alpha=0.55,
           label=f"excluded, below the plausibility floor (n={len(excl)})")
    for i, v in enumerate(units.values):
        ax.plot([-2.2], [v], marker="D", ms=4, color=VERM, zorder=5,
                label="MIMIC-IV care units" if i == 0 else None)
    ax.axhline(float(units.median()), color=VERM, lw=1.1, ls=(0, (4, 3)),
               zorder=4)
    ax.set_xlabel("eICU-CRD hospitals, ordered by median charting interval")
    ax.set_ylabel("Median interval between\nrecords (minutes)")
    ax.set_xlim(-4, len(med_i))
    ax.set_xticks([])
    ax.legend(frameon=False, loc="upper left")
    save(fig, out_dir, "Figure3_hospital_intervals")


def fig4(a, out_dir):
    """Share of each hospital's stays flagged by the pooled rule, restricted."""
    _, keep, d = hospital_medians(a)
    d = d[d["hospitalid"].isin(keep)].copy()
    cut = d["n_records"].quantile(0.10)
    d["low"] = (d["n_records"] <= cut).astype(int)
    per = d.groupby("hospitalid")["low"].mean().sort_values()

    ev = pd.read_parquet(a.mimic_cache)
    mc = ev.groupby("stay_id").size().rename("n").reset_index()
    m = mc.merge(pd.read_csv(a.mimic_per_stay,
                             usecols=["stay_id", "careunit", "los"]),
                 on="stay_id", how="inner")
    m = m[(m["los"] >= 1.0) & (m["n"] >= MIN_RECORDS)].copy()
    m["low"] = (m["n"] <= m["n"].quantile(0.10)).astype(int)
    mu = m.groupby("careunit")["low"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(per))
    ax.bar(x, per.values, color=BLUE, width=0.85, lw=0, alpha=0.9,
           label=f"eICU-CRD hospitals, restricted (n={len(per)})")
    ax.axhline(0.10, color=GREY, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.text(len(per) * 0.42, 0.115, "nominal decile (10%)", va="bottom",
            ha="left", color=GREY, fontsize=7.5)
    for i, v in enumerate(mu.values):
        ax.plot([-2.2], [v], marker="D", ms=4, color=VERM, zorder=5,
                label="MIMIC-IV care units" if i == 0 else None)
    ax.set_xlabel("Hospitals, ordered by share of stays designated low density")
    ax.set_ylabel("Share of the group's stays\nflagged by the pooled rule")
    ax.set_xlim(-4, len(per))
    ax.set_ylim(0, max(0.35, float(per.max()) * 1.08))
    ax.set_xticks([])
    ax.legend(frameon=False, loc="upper left")
    save(fig, out_dir, "Figure4_pooled_threshold")


def fig5(a, out_dir):
    """Participant flow, both cohorts."""
    ps = pd.read_csv(a.mimic_per_stay, usecols=["stay_id", "los"])
    cnt = (pd.read_parquet(a.mimic_cache).groupby("stay_id").size()
             .rename("n").reset_index())
    mm = ps.merge(cnt, on="stay_id", how="left")
    mm["n"] = mm["n"].fillna(0)
    n0 = len(mm)
    n1 = int((mm["los"] >= 1.0).sum())
    n2 = int(((mm["los"] >= 1.0) & (mm["n"] >= MIN_RECORDS)).sum())

    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid",
                               "unitdischargeoffset"])
    ec = (pd.read_parquet(a.eicu_nc_cache).groupby("patientunitstayid").size()
            .rename("n").reset_index())
    ee = pat.merge(ec, on="patientunitstayid", how="left")
    ee["n"] = ee["n"].fillna(0)
    e0 = len(ee)
    e1 = int((ee["unitdischargeoffset"] >= WINDOW_MIN).sum())
    sub = ee[(ee["unitdischargeoffset"] >= WINDOW_MIN)
             & (ee["n"] >= MIN_RECORDS)]
    e2 = len(sub)
    hc = sub["hospitalid"].value_counts()
    b = sub[sub["hospitalid"].isin(hc[hc >= MIN_HOSP].index)]
    e3 = len(b)
    med = b.groupby("hospitalid")["n"].median()
    e4 = int(b["hospitalid"].isin(med[med >= FLOOR].index).sum())

    steps_m = [(f"ICU stays\n{n0:,}", None),
               (f"Completing the 24-hour window\n{n1:,}",
                f"stay <24 h\n{n0-n1:,}"),
               (f"Analyzed\n{n2:,}", f"<{MIN_RECORDS} records\n{n1-n2:,}")]
    steps_e = [(f"Unit stays\n{e0:,}", None),
               (f"Completing the 24-hour window\n{e1:,}",
                f"stay <24 h\n{e0-e1:,}"),
               (f"Meeting the record minimum\n{e2:,}",
                f"<{MIN_RECORDS} records\n{e1-e2:,}"),
               (f"Hospitals with \u2265{MIN_HOSP} stays\n{e3:,}",
                f"hospital <{MIN_HOSP} stays\n{e2-e3:,}"),
               (f"Restricted primary analysis\n{e4:,}",
                f"hospital median\n<{FLOOR} records\n{e3-e4:,}")]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.8))
    for ax, steps, title in ((axes[0], steps_m, "MIMIC-IV"),
                             (axes[1], steps_e, "eICU-CRD")):
        ax.set_xlim(0, 11); ax.set_ylim(0, len(steps) * 2 + 0.5)
        ax.axis("off"); ax.set_title(title, fontsize=10, fontweight="bold")
        for i, (main, excl) in enumerate(steps):
            y = (len(steps) - i) * 2 - 1
            ax.text(2.6, y, main, ha="center", va="center", fontsize=7.4,
                    bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=BLUE,
                              lw=1.1))
            if i < len(steps) - 1:
                ax.annotate("", xy=(2.6, y - 1.5), xytext=(2.6, y - 0.55),
                            arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
            if excl:
                ax.text(7.4, y + 1, f"excluded\n{excl}", ha="center",
                        va="center", fontsize=6.7, color=VERM,
                        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                  ec=VERM, lw=0.8))
                ax.annotate("", xy=(5.6, y + 1), xytext=(2.6, y + 1),
                            arrowprops=dict(arrowstyle="->", color=VERM,
                                            lw=0.8))
    save(fig, out_dir, "Figure5_flow")
    print(f"    MIMIC-IV {n0:,} -> {n1:,} -> {n2:,}")
    print(f"    eICU-CRD {e0:,} -> {e1:,} -> {e2:,} -> {e3:,} -> {e4:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", required=True, type=Path)
    ap.add_argument("--stage4-dir", required=True, type=Path)
    ap.add_argument("--stage3-dir", type=Path)
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--mimic-cache", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./revision_figures"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    print("Figures on the restricted primary cohort")
    fig2(a, a.out_dir)
    fig3(a, a.out_dir)
    fig4(a, a.out_dir)
    fig5(a, a.out_dir)
    print(f"\n  Figure 1 is unchanged and is not regenerated.")
    print(f"-> {a.out_dir}")


if __name__ == "__main__":
    main()
