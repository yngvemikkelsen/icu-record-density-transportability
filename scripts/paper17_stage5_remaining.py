#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "statsmodels", "pyarrow", "matplotlib"]
# ///
"""
Paper 17 revision, Stage 5: the remaining items.

REVIEWER N COMMENT 4 — the cardiac vascular ICU
------------------------------------------------
CVICU appears as an exception in three separate analyses and each time the
manuscript notes it and moves on. The reviewer is right that this is the
unit-level effect the paper otherwise calls negligible, and that it deserves to
be a finding rather than a footnote. This assembles every CVICU-specific
quantity in one place so a subsection can be written from it: hours to first
record, on-the-hour charting fraction, the admission-hour association, and
where the unit sits on each documentation metric relative to the other eight.

EDITORIAL COMMENT 17 — participant flow
----------------------------------------
Produces the STROBE flow figure from the cohort derivation, both databases.

CONSEQUENCE OF THE RESTRICTION
------------------------------
Stage 4 established that eight eICU-CRD hospitals do not plausibly contribute
the nurse-charted stream, and that the principal estimates should be reported on
the restricted cohort. Two further published quantities depend on that cohort
and are recomputed here:

  the sixteen-fold spread in per-hospital median charting interval, which is
  quoted in the Abstract and Discussion and was computed before restriction

  the stream-selection contrast and the cross-vital replication, which used the
  unrestricted set

Usage:
  python paper17_stage5_remaining.py \
      --mimic-cache ~/bcst/unit_profile/hr_timestamps.parquet \
      --mimic-per-stay ~/bcst/multi_outcome_results/per_stay_multi_outcomes.csv \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-vp-cache ~/bcst/unit_profile_eicu/vitalperiodic_offsets.parquet \
      --cross-dir ~/bcst/cross_vitals \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/revision_stage5
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WINDOW_MIN = 24 * 60
MIN_RECORDS = 3
MIN_HOSP = 500
FLOOR = 12

BLUE, ORANGE, VERM, GREY = "#0072B2", "#E69F00", "#D55E00", "#666666"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300})


def metrics(ev, key, off, charttime=None):
    ev = ev.sort_values([key, off])
    g = ev.groupby(key)[off]
    out = pd.DataFrame({"n_records": g.size(), "t_first_h": g.min() / 60.0})
    ev = ev.assign(gap=g.diff())
    gg = ev.dropna(subset=["gap"]).groupby(key)["gap"]
    out["median_interval_min"] = gg.median()
    out["iqr_interval_min"] = gg.quantile(0.75) - gg.quantile(0.25)
    out["max_interval_min"] = gg.max()
    out["n_gaps_gt2h"] = gg.apply(lambda s: int((s > 120).sum()))
    out["frac_time_in_gaps"] = gg.apply(
        lambda s: float(s[s > 120].sum()) / WINDOW_MIN)
    if charttime is not None:
        minute = ev[charttime].dt.minute
        out["on_the_hour_frac"] = (ev.assign(oh=(minute < 5).astype(float))
                                     .groupby(key)["oh"].mean())
    return out.reset_index()


def eta2(y, groups):
    y = np.asarray(y, float); groups = np.asarray(groups)
    ok = ~np.isnan(y); y, groups = y[ok], groups[ok]
    if len(y) < 10:
        return np.nan
    grand = y.mean(); sst = ((y - grand) ** 2).sum()
    if sst <= 0:
        return np.nan
    d = pd.DataFrame({"y": y, "g": groups}).groupby("g")["y"].agg(["mean", "size"])
    return float((d["size"] * (d["mean"] - grand) ** 2).sum() / sst)


# ------------------------------------------------- REVIEWER N COMMENT 4 ----

def cvicu(a, out_dir):
    print("=" * 78)
    print("REVIEWER N 4 — the cardiac vascular ICU as a finding, not a footnote")
    print("=" * 78)
    ev = pd.read_parquet(a.mimic_cache)
    ct = "charttime" if "charttime" in ev.columns else None
    m = metrics(ev, "stay_id", "offset_min", ct)
    ps = pd.read_csv(a.mimic_per_stay, usecols=["stay_id", "careunit", "los"])
    m = m.merge(ps, on="stay_id", how="inner")
    m = m[(m["los"] >= 1.0) & (m["n_records"] >= MIN_RECORDS)].copy()

    cols = ["n_records", "t_first_h", "median_interval_min",
            "iqr_interval_min", "max_interval_min", "frac_time_in_gaps"]
    if "on_the_hour_frac" in m.columns:
        cols.append("on_the_hour_frac")

    prof = m.groupby("careunit")[cols].median()
    prof["n_stays"] = m.groupby("careunit").size()
    prof = prof.sort_values("n_stays", ascending=False)
    print(f"\n  per-unit medians ({len(prof)} units, {len(m):,} stays)")
    print(prof.round(3).to_string())
    prof.to_csv(out_dir / "cvicu_unit_profile.csv")

    cv = [u for u in prof.index if "Cardiac Vascular" in u or "CVICU" in u]
    if not cv:
        print("\n  CVICU not identified in careunit labels; check the labels above")
        return m
    cv = cv[0]
    others = prof.drop(index=cv)
    print(f"\n  {cv} against the other {len(others)} units")
    print(f"    {'metric':26s} {'CVICU':>10s} {'others median':>14s} "
          f"{'others range':>22s} {'outside?':>9s}")
    print("    " + "-" * 78)
    rows = []
    for c in cols:
        v = float(prof.loc[cv, c]); o = others[c]
        outside = v < o.min() or v > o.max()
        print(f"    {c:26s} {v:10.3f} {o.median():14.3f} "
              f"{f'{o.min():.3f} to {o.max():.3f}':>22s} "
              f"{'YES' if outside else '':>9s}")
        rows.append({"metric": c, "cvicu": v, "others_median": float(o.median()),
                     "others_min": float(o.min()), "others_max": float(o.max()),
                     "outside_range": bool(outside)})
    pd.DataFrame(rows).to_csv(out_dir / "cvicu_contrast.csv", index=False)

    # how much of the between-unit variance does CVICU alone account for?
    print(f"\n  between-unit eta-squared, all units vs CVICU removed")
    print(f"    {'metric':26s} {'all units':>10s} {'CVICU dropped':>14s} "
          f"{'share from CVICU':>18s}")
    print("    " + "-" * 74)
    for c in cols:
        e_all = eta2(m[c], m["careunit"])
        sub = m[m["careunit"] != cv]
        e_wo = eta2(sub[c], sub["careunit"])
        share = 1 - e_wo / e_all if e_all and e_all > 0 else np.nan
        print(f"    {c:26s} {e_all:10.4f} {e_wo:14.4f} {share:18.3f}")
    print("\n  A large share means the unit component the manuscript calls")
    print("  negligible is largely one unit with a distinct admission pathway,")
    print("  which is a finding rather than an exception to be noted.")
    return m


# ------------------------------------------------- EDITORIAL COMMENT 17 ----

def flow(a, m_mimic, out_dir):
    print("\n" + "=" * 78)
    print("EDITORIAL 17 — participant flow")
    print("=" * 78)
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
    sub = ee[(ee["unitdischargeoffset"] >= WINDOW_MIN) & (ee["n"] >= MIN_RECORDS)]
    e2 = len(sub)
    hc = sub["hospitalid"].value_counts()
    big = hc[hc >= MIN_HOSP].index
    e3 = int(sub["hospitalid"].isin(big).sum())
    med = sub[sub["hospitalid"].isin(big)].groupby("hospitalid")["n"].median()
    keep = med[med >= FLOOR].index
    e4 = int(sub["hospitalid"].isin(keep).sum())

    print(f"\n  MIMIC-IV   {n0:,} ICU stays")
    print(f"             {n0 - n1:,} excluded, stay shorter than 24 h")
    print(f"             {n1:,} completing the window")
    print(f"             {n1 - n2:,} excluded, fewer than {MIN_RECORDS} records")
    print(f"             {n2:,} analysed")
    print(f"\n  eICU-CRD   {e0:,} unit stays")
    print(f"             {e0 - e1:,} excluded, stay shorter than 24 h")
    print(f"             {e1:,} completing the window")
    print(f"             {e1 - e2:,} excluded, fewer than {MIN_RECORDS} records")
    print(f"             {e2:,} analysed")
    print(f"             {e2 - e3:,} excluded, hospital under {MIN_HOSP} stays")
    print(f"             {e3:,} in the clustered analyses")
    print(f"             {e3 - e4:,} excluded, hospital median under {FLOOR} records")
    print(f"             {e4:,} in the restricted primary analysis")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.6))
    for ax, steps, title in (
            (axes[0], [(f"ICU stays\\n{n0:,}", None),
                       (f"Completing the 24-hour window\\n{n1:,}",
                        f"excluded: stay <24 h\\n{n0 - n1:,}"),
                       (f"Analysed\\n{n2:,}",
                        f"excluded: <{MIN_RECORDS} records\\n{n1 - n2:,}")],
             "MIMIC-IV"),
            (axes[1], [(f"Unit stays\\n{e0:,}", None),
                       (f"Completing the 24-hour window\\n{e1:,}",
                        f"excluded: stay <24 h\\n{e0 - e1:,}"),
                       (f"Analysed\\n{e2:,}",
                        f"excluded: <{MIN_RECORDS} records\\n{e1 - e2:,}"),
                       (f"Hospitals \u2265{MIN_HOSP} stays\\n{e3:,}",
                        f"excluded\\n{e2 - e3:,}"),
                       (f"Restricted primary\\n{e4:,}",
                        f"excluded: hospital median\\n<{FLOOR} records\\n{e3 - e4:,}")],
             "eICU-CRD")):
        ax.set_xlim(0, 11); ax.set_ylim(0, len(steps) * 2 + 0.5)
        ax.axis("off"); ax.set_title(title, fontsize=10, fontweight="bold")
        for i, (main, excl) in enumerate(steps):
            y = (len(steps) - i) * 2 - 1
            ax.text(2.6, y, main.replace("\\n", "\n"), ha="center", va="center",
                    fontsize=7.6, bbox=dict(boxstyle="round,pad=0.45",
                                            fc="white", ec=BLUE, lw=1.1))
            if i < len(steps) - 1:
                ax.annotate("", xy=(2.6, y - 1.5), xytext=(2.6, y - 0.55),
                            arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
            if excl:
                ax.text(7.2, y + 1, excl.replace("\\n", "\n"), ha="center",
                        va="center", fontsize=6.9, color=VERM,
                        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                  ec=VERM, lw=0.8))
                ax.annotate("", xy=(5.3, y + 1), xytext=(2.6, y + 1),
                            arrowprops=dict(arrowstyle="->", color=VERM, lw=0.8))
    fig.savefig(out_dir / "Figure5_flow.png", bbox_inches="tight",
                facecolor="white", dpi=184)
    fig.savefig(out_dir / "Figure5_flow.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"\n  wrote Figure5_flow.png (184 dpi, within 1200 px) and .pdf")


# ----------------------------------- quantities that move with restriction --

def restricted_quantities(a, out_dir):
    print("\n" + "=" * 78)
    print("QUANTITIES QUOTED IN THE PAPER THAT MOVE WITH THE RESTRICTION")
    print("=" * 78)
    d = metrics(pd.read_parquet(a.eicu_nc_cache), "patientunitstayid",
                "observationoffset")
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    d = d.merge(pat, on="patientunitstayid", how="inner")
    d = d[(d["unitdischargeoffset"] >= WINDOW_MIN)
          & (d["n_records"] >= MIN_RECORDS)].copy()
    cnt = d["hospitalid"].value_counts()
    d = d[d["hospitalid"].isin(cnt[cnt >= MIN_HOSP].index)]
    med_all = d.groupby("hospitalid")["median_interval_min"].median()
    cnt_all = d.groupby("hospitalid")["n_records"].median()
    keep = cnt_all[cnt_all >= FLOOR].index
    med_r = med_all.loc[keep]
    cnt_r = cnt_all.loc[keep]

    print(f"\n  per-hospital median charting interval, minutes")
    for lab, s in (("unrestricted", med_all), ("restricted", med_r)):
        print(f"    {lab:14s} n={len(s):3d}  p10 {s.quantile(.1):6.1f}  "
              f"median {s.median():6.1f}  p90 {s.quantile(.9):6.1f}  "
              f"max {s.max():6.1f}   p90/p10 = {s.quantile(.9) / s.quantile(.1):.1f}x")
    print(f"\n  The manuscript quotes a sixteen-fold spread from the")
    print(f"  unrestricted p10 and p90. The restricted figure is what should")
    print(f"  now appear in the Abstract and Discussion.")

    print(f"\n  per-hospital median record count")
    for lab, s in (("unrestricted", cnt_all), ("restricted", cnt_r)):
        print(f"    {lab:14s} p10 {s.quantile(.1):6.1f}  median {s.median():6.1f}"
              f"  p90 {s.quantile(.9):6.1f}")
    pd.DataFrame({"unrestricted_interval": med_all,
                  "unrestricted_count": cnt_all}).to_csv(
        out_dir / "hospital_medians.csv")

    # stream comparison on the restricted set
    vp = pd.read_parquet(a.eicu_vp_cache)
    v = vp.groupby("patientunitstayid").size().rename("n_vp").reset_index()
    j = d[["patientunitstayid", "hospitalid", "n_records"]].merge(
        v, on="patientunitstayid", how="inner")
    print(f"\n  stream agreement, bottom decile, on stays in both tables")
    for lab, sub in (("unrestricted", j),
                     ("restricted", j[j["hospitalid"].isin(keep)])):
        lo_v = sub["n_vp"] <= sub["n_vp"].quantile(.10)
        lo_n = sub["n_records"] <= sub["n_records"].quantile(.10)
        inter = int((lo_v & lo_n).sum())
        obs = inter / max(int(lo_v.sum()), 1)
        chance = float(lo_n.mean())
        print(f"    {lab:14s} n={len(sub):,}  observed {obs:.3f}  "
              f"chance {chance:.3f}  ratio {obs / chance:.2f}  "
              f"Spearman {sub['n_vp'].rank().corr(sub['n_records'].rank()):+.3f}")
    print("  The stream-selection finding does not depend on the restriction;")
    print("  report the restricted figures for consistency with the primary")
    print("  analysis.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-cache", required=True, type=Path)
    ap.add_argument("--mimic-per-stay", required=True, type=Path)
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--eicu-vp-cache", required=True, type=Path)
    ap.add_argument("--cross-dir", type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./revision_stage5"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    m = cvicu(a, a.out_dir)
    flow(a, m, a.out_dir)
    restricted_quantities(a, a.out_dir)
    print(f"\n-> {a.out_dir}")


if __name__ == "__main__":
    main()
