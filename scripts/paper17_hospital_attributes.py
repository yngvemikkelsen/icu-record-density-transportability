#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "statsmodels", "pyarrow"]
# ///
"""
Paper 17, editorial comment 6: how much of the hospital component is explained
by the hospital attributes eICU-CRD actually records?

WHY THIS EXISTS
---------------
Comment 6 asks how variation in hospital tier and nurse-to-patient ratio might
independently drive the observed documentation differences. The manuscript has
been treating all four site attributes as unrecorded. Three of them are not:
the eICU-CRD `hospital` table carries numbedscategory, teachingstatus and
region, collected by self-reported survey. Nurse-to-patient ratio, bedside EHR
vendor and local tele-ICU configuration genuinely are absent.

So part of the comment is answerable. This script asks: of the hospital
component in record density, how much is attributable to bed capacity, teaching
status and region, and how much survives conditioning on them?

WHAT IS COMPUTED
----------------
  1. Coverage. How many of the analysed hospitals have each attribute
     populated. The table is survey-derived and incomplete; if coverage is poor
     the rest is uninterpretable and the script says so.

  2. Marginal association. One-way eta-squared of each documentation metric on
     each attribute, at stay level, so it is on the same scale as the hospital
     eta-squared already reported.

  3. Conditional decomposition. Hospital eta-squared before and after
     residualising the metric on the recorded attributes. The difference is the
     share of the site component the recorded attributes account for; the
     remainder is what vendor, staffing, configuration, export completeness and
     practice must share.

  5. Permutation null. Conditioning on six indicator coefficients removes some of
     the hospital component by construction. Attribute vectors are shuffled
     across hospitals and the explained share recomputed, so the observed value
     can be read against the share that arises from the parameter count alone.

  4. Per-attribute profile. Median record count and median charting interval by
     bed category and by teaching status, so the direction of any association
     is visible rather than only its magnitude.

All four are reported on the plausibility-restricted primary cohort and on the
unrestricted cohort.

INTERPRETATION, STATED IN ADVANCE
---------------------------------
A small conditional reduction means the recorded structural attributes do not
explain the site component, which strengthens the paper's position: the
component is not a proxy for hospital size or teaching status. A large
reduction would mean the site component is substantially structural, which is a
different and also publishable finding, but it would require rewriting the
Discussion. Either way this is reported as computed.

Coverage is the caveat that governs everything: attributes are self-reported and
may be missing, so a null result is evidence of no association only among the
hospitals for which the attribute is populated.

Usage:
  python paper17_hospital_attributes.py \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-root ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir ~/bcst/hospital_attributes
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WINDOW_MIN = 24 * 60
MIN_RECORDS = 3
MIN_HOSP = 500
FLOOR = 12

METRICS = ["n_records", "median_interval_min", "max_interval_min",
           "n_gaps_gt2h", "frac_time_in_gaps"]
LABEL = {"n_records": "Record count",
         "median_interval_min": "Median interval",
         "max_interval_min": "Longest interval",
         "n_gaps_gt2h": "Gaps >2 h",
         "frac_time_in_gaps": "Fraction of window in gaps"}
ATTRS = ["numbedscategory", "teachingstatus", "region"]
ATTR_LABEL = {"numbedscategory": "Bed capacity category",
              "teachingstatus": "Teaching status",
              "region": "Region"}


def eta2(y, groups):
    y = np.asarray(y, float); groups = np.asarray(groups)
    ok = ~pd.isna(y) & ~pd.isna(groups)
    y, groups = y[ok], groups[ok]
    if len(y) < 10:
        return np.nan
    grand = y.mean(); sst = ((y - grand) ** 2).sum()
    if sst <= 0:
        return np.nan
    d = pd.DataFrame({"y": y, "g": groups}).groupby("g")["y"].agg(["mean", "size"])
    return float((d["size"] * (d["mean"] - grand) ** 2).sum() / sst)


def residualise_on_dummies(y, frame, cols):
    """Least-squares residuals of y on indicator columns for cols. Rows with any
    attribute missing are returned as NaN and dropped downstream."""
    sub = frame[cols]
    ok = (~sub.isna().any(axis=1)).to_numpy() & ~pd.isna(y)
    out = np.full(len(y), np.nan)
    if ok.sum() < 100:
        return out
    X = pd.get_dummies(sub[ok].astype(str), drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(X)), X])
    yy = np.asarray(y, float)[ok]
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    out[ok] = yy - X @ beta
    return out


def per_stay_metrics(ev):
    ev = ev.sort_values(["patientunitstayid", "observationoffset"])
    g = ev.groupby("patientunitstayid")["observationoffset"]
    out = pd.DataFrame({"n_records": g.size()})
    ev = ev.assign(gap=g.diff())
    gg = ev.dropna(subset=["gap"]).groupby("patientunitstayid")["gap"]
    out["median_interval_min"] = gg.median()
    out["max_interval_min"] = gg.max()
    out["n_gaps_gt2h"] = gg.apply(lambda s: int((s > 120).sum()))
    out["frac_time_in_gaps"] = gg.apply(
        lambda s: float(s[s > 120].sum()) / WINDOW_MIN)
    return out.reset_index()


def load(a):
    d = per_stay_metrics(pd.read_parquet(a.eicu_nc_cache))
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    d = d.merge(pat, on="patientunitstayid", how="inner")
    d = d[(d["unitdischargeoffset"] >= WINDOW_MIN)
          & (d["n_records"] >= MIN_RECORDS)].copy()
    cnt = d["hospitalid"].value_counts()
    d = d[d["hospitalid"].isin(cnt[cnt >= MIN_HOSP].index)].copy()

    hosp = pd.read_csv(a.eicu_root / "hospital.csv.gz")
    keep = ["hospitalid"] + [c for c in ATTRS if c in hosp.columns]
    missing = set(ATTRS) - set(hosp.columns)
    if missing:
        print(f"  NOTE: hospital table lacks {sorted(missing)} — skipped")
    d = d.merge(hosp[keep], on="hospitalid", how="left")

    med = d.groupby("hospitalid")["n_records"].median()
    d["restricted"] = d["hospitalid"].isin(med[med >= a.floor].index)
    return d, [c for c in ATTRS if c in hosp.columns]


def coverage(d, attrs, out_dir):
    print("\n" + "=" * 78)
    print("1. COVERAGE OF THE RECORDED HOSPITAL ATTRIBUTES")
    print("=" * 78)
    rows = []
    for cohort, sub in (("unrestricted", d), ("restricted", d[d["restricted"]])):
        h = sub.drop_duplicates("hospitalid")
        for c in attrs:
            n_pop = int(h[c].notna().sum())
            rows.append({"cohort": cohort, "attribute": ATTR_LABEL[c],
                         "n_hospitals": len(h), "n_populated": n_pop,
                         "share_populated": n_pop / max(len(h), 1),
                         "n_levels": int(h[c].nunique(dropna=True))})
            print(f"  {cohort:13s} {ATTR_LABEL[c]:24s} "
                  f"{n_pop:3d}/{len(h):3d} hospitals populated "
                  f"({n_pop / max(len(h), 1):.1%}), "
                  f"{h[c].nunique(dropna=True)} levels")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "attribute_coverage.csv", index=False)
    worst = df["share_populated"].min()
    if worst < 0.5:
        print("\n  WARNING: at least one attribute is populated for fewer than")
        print("  half the analysed hospitals. Report coverage alongside any")
        print("  estimate below, and do not read a null as evidence of no")
        print("  association across all sites.")
    return df


def marginal(d, attrs, out_dir):
    print("\n" + "=" * 78)
    print("2. MARGINAL ASSOCIATION: eta-squared of each metric on each attribute")
    print("=" * 78)
    rows = []
    for cohort, sub in (("unrestricted", d), ("restricted", d[d["restricted"]])):
        print(f"\n  {cohort}")
        print(f"    {'metric':30s} " + " ".join(f"{ATTR_LABEL[c][:14]:>15s}"
                                                for c in attrs)
              + f"{'hospital':>11s}")
        for m in METRICS:
            vals = [eta2(sub[m], sub[c]) for c in attrs]
            eh = eta2(sub[m], sub["hospitalid"])
            print(f"    {LABEL[m]:30s} "
                  + " ".join(f"{v:15.4f}" for v in vals) + f"{eh:11.4f}")
            row = {"cohort": cohort, "metric": LABEL[m], "eta2_hospital": eh}
            row.update({f"eta2_{c}": v for c, v in zip(attrs, vals)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "attribute_marginal_eta2.csv",
                              index=False)


def conditional(d, attrs, out_dir):
    print("\n" + "=" * 78)
    print("3. HOSPITAL COMPONENT BEFORE AND AFTER CONDITIONING ON ATTRIBUTES")
    print("=" * 78)
    print("  Computed on stays with all three attributes populated, so the two")
    print("  columns are on the same rows and the difference is interpretable.")
    rows = []
    for cohort, sub in (("unrestricted", d), ("restricted", d[d["restricted"]])):
        complete = sub.dropna(subset=attrs)
        nh = complete["hospitalid"].nunique()
        print(f"\n  {cohort}: {len(complete):,} stays, {nh} hospitals with all "
              f"attributes populated")
        print(f"    {'metric':30s} {'hospital':>10s} {'| attributes':>13s} "
              f"{'explained':>11s}")
        for m in METRICS:
            before = eta2(complete[m], complete["hospitalid"])
            resid = residualise_on_dummies(complete[m].to_numpy(float),
                                           complete, attrs)
            after = eta2(resid, complete["hospitalid"].to_numpy())
            share = (before - after) / before if before and before > 0 else np.nan
            print(f"    {LABEL[m]:30s} {before:10.4f} {after:13.4f} "
                  f"{share:10.1%}")
            rows.append({"cohort": cohort, "metric": LABEL[m],
                         "eta2_hospital": before,
                         "eta2_hospital_given_attributes": after,
                         "share_explained_by_attributes": share,
                         "n_stays": len(complete), "n_hospitals": nh})
    pd.DataFrame(rows).to_csv(out_dir / "attribute_conditional_eta2.csv",
                              index=False)
    print("\n  'explained' is the proportion of the hospital component removed")
    print("  by conditioning on bed capacity, teaching status and region. What")
    print("  remains is shared by vendor, staffing, local configuration, export")
    print("  completeness and documentation practice, none of which is")
    print("  recorded.")


def permutation(d, attrs, out_dir, n_perm=500, seed=17):
    """Null distribution of the explained share.

    Conditioning removes one parameter per attribute level from a between-
    hospital structure with only as many degrees of freedom as there are
    hospitals, so some share is removed by construction. This shuffles the
    attribute vector ACROSS HOSPITALS, keeping each hospital's stays together
    and keeping the joint distribution of the three attributes intact, then
    recomputes the explained share. The observed value is reported against that
    null.
    """
    print("\n" + "=" * 78)
    print(f"5. PERMUTATION NULL FOR THE EXPLAINED SHARE ({n_perm} permutations)")
    print("=" * 78)
    print("  Attribute vectors are shuffled across hospitals, so any share")
    print("  removed under the null is capitalisation on the number of")
    print("  parameters rather than a real association.")
    rng = np.random.default_rng(seed)
    rows = []
    for cohort, sub in (("unrestricted", d), ("restricted", d[d["restricted"]])):
        complete = sub.dropna(subset=attrs).copy()
        hosp = (complete[["hospitalid"] + attrs]
                .drop_duplicates("hospitalid").reset_index(drop=True))
        nh = len(hosp)
        print(f"\n  {cohort}: {len(complete):,} stays, {nh} hospitals")
        print(f"    {'metric':30s} {'observed':>9s} {'null med':>9s} "
              f"{'null p95':>9s} {'p':>7s}")
        for m in METRICS:
            y = complete[m].to_numpy(float)
            hid = complete["hospitalid"].to_numpy()
            before = eta2(y, hid)
            obs = (before - eta2(residualise_on_dummies(y, complete, attrs),
                                 hid)) / before
            null = []
            for _ in range(n_perm):
                perm_map = hosp.copy()
                perm_map[attrs] = hosp[attrs].to_numpy()[
                    rng.permutation(nh)]
                shuffled = complete[["hospitalid"]].merge(
                    perm_map, on="hospitalid", how="left")
                r = residualise_on_dummies(y, shuffled, attrs)
                null.append((before - eta2(r, hid)) / before)
            null = np.array([x for x in null if np.isfinite(x)])
            med, p95 = np.median(null), np.percentile(null, 95)
            pval = float((null >= obs).mean())
            print(f"    {LABEL[m]:30s} {obs:9.1%} {med:9.1%} {p95:9.1%} "
                  f"{pval:7.3f}")
            rows.append({"cohort": cohort, "metric": LABEL[m],
                         "observed_share": obs, "null_median": med,
                         "null_p95": p95, "p_value": pval,
                         "excess_over_null": obs - med,
                         "n_hospitals": nh, "n_perm": len(null)})
    pd.DataFrame(rows).to_csv(out_dir / "attribute_permutation.csv",
                              index=False)
    print("\n  Report the observed share against the null median. The excess")
    print("  over the null, not the raw share, is what the recorded attributes")
    print("  actually account for.")


def profile(d, attrs, out_dir):
    print("\n" + "=" * 78)
    print("4. DOCUMENTATION PROFILE BY ATTRIBUTE, RESTRICTED COHORT")
    print("=" * 78)
    sub = d[d["restricted"]]
    frames = []
    for c in attrs:
        g = (sub.groupby(c)
                .agg(n_hospitals=("hospitalid", "nunique"),
                     n_stays=("n_records", "size"),
                     median_count=("n_records", "median"),
                     median_interval=("median_interval_min", "median")))
        g.index = g.index.astype(str)
        g.insert(0, "attribute", ATTR_LABEL[c])
        g.index.name = "level"
        print(f"\n  {ATTR_LABEL[c]}")
        print(g.to_string())
        frames.append(g.reset_index())
    pd.concat(frames).to_csv(out_dir / "attribute_profiles.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eicu-nc-cache", required=True, type=Path)
    ap.add_argument("--eicu-root", required=True, type=Path)
    ap.add_argument("--floor", type=int, default=FLOOR)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--out-dir", type=Path, default=Path("./hospital_attributes"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("HOSPITAL ATTRIBUTES AND THE SITE COMPONENT, eICU-CRD nurse stream")
    print("=" * 78)
    d, attrs = load(a)
    print(f"  {len(d):,} stays, {d['hospitalid'].nunique()} hospitals "
          f"(>= {MIN_HOSP} eligible stays)")
    print(f"  restricted cohort: {int(d['restricted'].sum()):,} stays, "
          f"{d.loc[d['restricted'], 'hospitalid'].nunique()} hospitals "
          f"(median >= {a.floor} records)")
    print(f"  attributes available: {[ATTR_LABEL[c] for c in attrs]}")
    print("  not recorded in eICU-CRD: bedside EHR vendor, nurse-to-patient "
          "ratio,\n  local tele-ICU configuration")

    coverage(d, attrs, a.out_dir)
    marginal(d, attrs, a.out_dir)
    conditional(d, attrs, a.out_dir)
    permutation(d, attrs, a.out_dir, n_perm=a.n_perm)
    profile(d, attrs, a.out_dir)

    print("\n" + "=" * 78)
    print("HOW TO USE THIS IN THE MANUSCRIPT")
    print("=" * 78)
    print("  If the explained share is small, editorial comment 6 is answered")
    print("  directly: hospital tier and size are recorded, are associated with")
    print("  documentation only weakly, and do not account for the site")
    print("  component. Report the share and the coverage, and confine the")
    print("  'not recorded' sentence to vendor, staffing ratio and tele-ICU")
    print("  configuration.")
    print("  If it is large, the site component is substantially structural and")
    print("  the Discussion needs rewriting around that. Report it either way.")
    print(f"\n-> {a.out_dir}")


if __name__ == "__main__":
    main()
