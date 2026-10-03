#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "pyarrow"]
# ///
"""
Paper 17 revision, editorial comments 2, 3 and the count part of 4.

COMMENT 2 — count models for the gap metrics
The first round asked for generalized linear mixed models for the count
outcomes.  Only record count was refitted.  Gaps exceeding 30 minutes and gaps
exceeding 2 hours are also non-negative integer counts, the latter heavily
zero-inflated, and remain in Gaussian models.  This fits:

    gaps >30 min   negative binomial GLMM, log link, random intercepts for
                   hospital and for unit within hospital
    gaps >2 h      hurdle negative binomial GLMM: a Bernoulli logit GLMM for
                   whether any gap occurs, and a zero-truncated negative
                   binomial GLMM for how many, given at least one

Hurdle rather than zero-inflated: a zero here is a definite statement about the
stay, that no interval exceeded two hours, not a mixture of structural and
sampling zeros that these data could separate.  A hurdle model has two variance
partitions and no single pooled one, so both parts are reported and neither is
presented as "the" VPC for the metric.

COMMENT 3 — latent versus observed scale
The negative binomial VPCs in Table 3 are on the latent (log) scale, where the
level-1 variance is ln(1 + 1/mu + alpha) and so depends on which mean is put in
for mu; the Gaussian VPCs are on the standardized observed scale.  The two are
not comparable, so setting 0.348 beside 0.306 does not establish that the count
model strengthens the site-dominant result.  For every negative binomial model
this reports:

  * the latent-scale VPC, printing the value of mu used;
  * the observed-scale VPC in closed form.  For this model it is exact rather
    than an approximation.  With A = exp(a), C = exp(c), m0 = exp(b0):
        Var_hospital = m0^2 E[C]^2 (E[A^2] - E[A]^2)
        Var_unit     = m0^2 E[A^2] (E[C^2] - E[C]^2)
        Var_residual = m0 E[A] E[C] + alpha m0^2 E[A^2] E[C^2]
    using E[A] = exp(s_h^2/2) and E[A^2] = exp(2 s_h^2), likewise for C;
  * the same quantity by Monte Carlo over the random effects, which is the
    route the comment suggests and a check on the algebra;
  * what the eta-squared estimator returns on data simulated from the fitted
    model.  That is NOT the observed-scale VPC: eta-squared is biased upward
    because it uses observed group means from finitely many stays, and the gap
    between the two columns is that bias at these cluster sizes.  It is printed
    because it bears directly on comment 1.

COMMENT 4, for the count models
--profile-ci adds profile-likelihood intervals, one of the two methods the
comment names.  A hospital-clustered bootstrap of a GLMM is not affordable at
this scale: one fit takes about a minute.  The VPC is made the free parameter
by solving s_h^2 = v (s_u^2 + level-1) / (1 - v), the remaining parameters are
re-maximised at each fixed v from a warm start, and the bound is where the
deviance crosses 3.841.  By default this is done for record count in the
restricted cohort, which is where the comment names 0.348 and 0.112;
--profile-all extends it to every model and costs several hours.

Run --selftest first.  It needs no data.

Usage
  python paper17_count_models.py --selftest

  python paper17_count_models.py \
      --eicu-nc-cache ~/bcst/unit_profile_eicu/nursecharting_offsets.parquet \
      --eicu-root     ~/physionet.org/files/eicu-crd/2.0 \
      --out-dir       ~/bcst/count_models \
      [--profile-ci] [--profile-all]
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import splu
from scipy.special import expit, gammaln

warnings.filterwarnings("ignore")

WINDOW_MIN = 24 * 60
MIN_RECORDS = 3
MIN_HOSP = 500
FLOOR = 12
CHI2_95 = 3.8414588206941245


# ============================================================ likelihoods ===
def nb_terms(y, eta, alpha):
    """NB2 log-likelihood with first and second derivatives wrt eta."""
    r = 1.0 / alpha
    mu = np.exp(eta)
    s = r + mu
    ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
          + r * (np.log(r) - np.log(s)) + y * (eta - np.log(s)))
    g = y - (y + r) * mu / s
    w = (y + r) * r * mu / (s * s)
    return ll.sum(), g, w


def ztnb_terms(y, eta, alpha):
    """Zero-truncated NB2: subtracts log(1 - P(0)) and its two derivatives.

    With t = alpha*mu and L0 = log P(0) = -log1p(t)/alpha,
        L0'  = -mu/(1+t)              L0'' = -mu/(1+t)^2
    and for q = log(1 - P0),
        q'   = -P0 L0' / (1-P0)
        q''  = -[ P0 (L0'^2 + L0'') / (1-P0) + (P0 L0' / (1-P0))^2 ]
    """
    ll, g, w = nb_terms(y, eta, alpha)
    mu = np.exp(eta)
    t = alpha * mu
    L0 = -np.log1p(t) / alpha
    P0 = np.exp(L0)
    one_m = np.maximum(-np.expm1(L0), 1e-300)
    L0p = -mu / (1.0 + t)
    L0pp = -mu / (1.0 + t) ** 2
    dq = -P0 * L0p / one_m
    d2q = -(P0 * (L0p ** 2 + L0pp) / one_m + (P0 * L0p / one_m) ** 2)
    return ll - np.log(one_m).sum(), g - dq, np.maximum(w + d2q, 1e-10)


def bern_terms(y, eta, alpha=None):
    p = expit(eta)
    return (float(np.sum(y * eta - np.logaddexp(0.0, eta))), y - p,
            np.maximum(p * (1.0 - p), 1e-10))


FAMILY = {"nb": nb_terms, "ztnb": ztnb_terms, "bernoulli": bern_terms}
NPAR = {"nb": 4, "ztnb": 4, "bernoulli": 3}


# ================================================================ fitting ===
def design(n, hosp_idx, unit_idx):
    q_h, q_u = int(hosp_idx.max()) + 1, int(unit_idx.max()) + 1
    rows = np.arange(n)
    Z = sparse.hstack([
        sparse.csr_matrix((np.ones(n), (rows, hosp_idx)), shape=(n, q_h)),
        sparse.csr_matrix((np.ones(n), (rows, unit_idx)), shape=(n, q_u)),
    ]).tocsr()
    return Z, q_h, q_u


def laplace(theta, y, Z, q_h, q_u, family, tol=1e-8, maxit=80):
    """Laplace-approximated log-likelihood at theta = (b0, ln s_h, ln s_u[, ln a])."""
    terms = FAMILY[family]
    if family == "bernoulli":
        b0, ls_h, ls_u = theta
        alpha = None
    else:
        b0, ls_h, ls_u, la = theta
        alpha = float(np.exp(la))
        if not np.isfinite(alpha) or not 1e-8 < alpha < 1e4:
            return -np.inf, None
    s_h2, s_u2 = float(np.exp(2 * ls_h)), float(np.exp(2 * ls_u))
    if min(s_h2, s_u2) < 1e-12 or max(s_h2, s_u2) > 1e4:
        return -np.inf, None
    q = q_h + q_u
    dinv = np.concatenate([np.full(q_h, 1.0 / s_h2), np.full(q_u, 1.0 / s_u2)])
    Dinv = sparse.diags(dinv)

    b = np.zeros(q)
    for _ in range(maxit):
        eta = b0 + Z @ b
        if not np.all(np.isfinite(eta)) or np.max(np.abs(eta)) > 50:
            return -np.inf, None
        _, g, w = terms(y, eta, alpha)
        H = (Z.T @ sparse.diags(w) @ Z + Dinv).tocsc()
        try:
            step = splu(H).solve(Z.T @ g - dinv * b)
        except (RuntimeError, ValueError):
            return -np.inf, None
        b = b + step
        if np.max(np.abs(step)) < tol:
            break

    eta = b0 + Z @ b
    ll, _, w = terms(y, eta, alpha)
    H = (Z.T @ sparse.diags(w) @ Z + Dinv).tocsc()
    try:
        lu = splu(H)
    except (RuntimeError, ValueError):
        return -np.inf, None
    val = (ll - 0.5 * float(np.sum(dinv * b * b))
           - 0.5 * (q_h * np.log(s_h2) + q_u * np.log(s_u2))
           - 0.5 * float(np.sum(np.log(np.abs(lu.U.diagonal())))))
    return (val if np.isfinite(val) else -np.inf), b


ALPHA_FLOOR = 1e-3


def fit(y, hosp_idx, unit_idx, family="nb", label="", x0=None, quiet=True,
        fixed_alpha=None):
    """Fit by Laplace-approximated maximum likelihood from several starts.

    Three failure modes cost us a first run and are now handled explicitly.

      *  Nelder-Mead can terminate on the plateau where the likelihood is
         non-finite and the objective returns the 1e12 sentinel.  scipy then
         reports success=True and the "estimates" are simply the start values.
         Any fit whose log-likelihood is at or near the sentinel is marked
         failed, whatever scipy says.
      *  A single start is not enough.  The moment start for alpha can sit in a
         region the simplex cannot leave, so several starts are tried and the
         highest likelihood kept.
      *  For a zero-truncated model the dispersion can be driven to the
         boundary, where log alpha runs to minus infinity and the simplex
         wanders without converging.  When alpha lands on the floor the model
         is refitted with alpha held at the floor, which is the zero-truncated
         Poisson, and the result is flagged.
    """
    n = len(y)
    Z, q_h, q_u = design(n, hosp_idx, unit_idx)
    bern = family == "bernoulli"
    t0 = time.time()

    def nll_factory(fa):
        def nll(t):
            if fa is None:
                th = t
            else:
                th = np.concatenate([t, [np.log(fa)]])
            v, _ = laplace(th, y, Z, q_h, q_u, family)
            return -v if np.isfinite(v) else 1e12
        return nll

    def starts():
        if bern:
            p = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
            b = np.log(p / (1 - p))
            return [np.array([b, np.log(sh), np.log(su)])
                    for sh, su in ((0.35, 0.20), (0.80, 0.40), (0.15, 0.10),
                                   (1.60, 0.60))]
        m0, v0 = float(y.mean()), float(y.var(ddof=1))
        a_mom = min(max((v0 - m0) / m0 ** 2, 1e-3), 50.0)
        b = np.log(max(m0, 0.5))
        grid = [(a, sh, su)
                for a in dict.fromkeys([a_mom, 0.15, 0.05, 0.5, 1.5])
                for sh, su in ((0.35, 0.20), (0.75, 0.35))]
        out = [np.array([b, np.log(sh), np.log(su), np.log(a)])
               for a, sh, su in grid]
        return [x[:3] for x in out] if fixed_alpha is not None else out

    nll = nll_factory(fixed_alpha)
    # Screen candidate starts by ONE likelihood evaluation each, then optimise
    # from the best.  A full optimisation per start costs several minutes at
    # cohort scale; screening costs seconds and fixes the same failure, which
    # was a single moment start landing where the simplex could not move.
    cand = starts() if x0 is None else [np.asarray(x0, float)]
    if len(cand) > 1:
        cand = sorted(cand, key=nll)
    best = None
    for x0i in cand[:3]:
        if nll(x0i) >= 1e11:
            continue
        r = minimize(nll, x0i, method="Nelder-Mead",
                     options={"maxiter": 6000, "maxfev": 9000,
                              "xatol": 1e-6, "fatol": 1e-6})
        r = minimize(nll, r.x, method="Nelder-Mead",
                     options={"maxiter": 3000, "xatol": 1e-8, "fatol": 1e-8})
        if r.fun < 1e11 and (best is None or r.fun < best.fun):
            best = r
        if best is not None and best.success:
            break
    if best is None:
        return {"label": label, "family": family, "n": n,
                "n_hospitals": int(q_h), "n_units": int(q_u),
                "intercept": np.nan, "var_hospital": np.nan,
                "var_unit": np.nan, "alpha": np.nan, "loglik": np.nan,
                "converged": False, "fit_failed": True, "n_iter": 0,
                "theta": None, "mean_outcome": float(y.mean()),
                "alpha_at_boundary": False,
                "fit_seconds": time.time() - t0,
                "level1_var": np.nan, "mu_used": np.nan,
                "vpc_hospital_latent": np.nan, "vpc_unit_latent": np.nan,
                "vpc_hospital_observed": np.nan, "vpc_unit_observed": np.nan}

    th = (best.x if fixed_alpha is None
          else np.concatenate([best.x, [np.log(fixed_alpha)]]))
    alpha = np.nan if bern else float(np.exp(th[3]))
    boundary = (not bern) and fixed_alpha is None and alpha < ALPHA_FLOOR
    if boundary:
        return fit(y, hosp_idx, unit_idx, family, label, quiet=quiet,
                   fixed_alpha=ALPHA_FLOOR) | {"alpha_at_boundary": True}

    out = {"label": label, "family": family, "n": n,
           "n_hospitals": int(q_h), "n_units": int(q_u),
           "intercept": float(th[0]),
           "var_hospital": float(np.exp(2 * th[1])),
           "var_unit": float(np.exp(2 * th[2])),
           "alpha": alpha,
           "loglik": float(-best.fun), "converged": bool(best.success),
           "fit_failed": False,
           "alpha_at_boundary": bool(fixed_alpha is not None),
           "n_iter": int(best.nit), "theta": th.copy(),
           "mean_outcome": float(y.mean()), "fit_seconds": time.time() - t0}
    out.update(vpcs(out))
    return out


# ============================================================ VPC on scales ==
def vpcs(o):
    """Latent-scale VPCs, and the exact observed-scale VPCs in closed form."""
    s_h2, s_u2 = o["var_hospital"], o["var_unit"]
    if o["family"] == "bernoulli":
        lvl1 = np.pi ** 2 / 3.0
        tot = s_h2 + s_u2 + lvl1
        return {"level1_var": lvl1, "mu_used": np.nan,
                "vpc_hospital_latent": s_h2 / tot,
                "vpc_unit_latent": s_u2 / tot,
                "vpc_hospital_observed": np.nan,
                "vpc_unit_observed": np.nan}
    alpha, mu = o["alpha"], o["mean_outcome"]
    lvl1 = float(np.log1p(1.0 / mu + alpha))
    tot = s_h2 + s_u2 + lvl1
    V_h, V_u, V_r = observed_components(o["intercept"], s_h2, s_u2, alpha)
    T = V_h + V_u + V_r
    return {"level1_var": lvl1, "mu_used": mu,
            "vpc_hospital_latent": s_h2 / tot,
            "vpc_unit_latent": s_u2 / tot,
            "vpc_hospital_observed": float(V_h / T),
            "vpc_unit_observed": float(V_u / T)}


def observed_components(b0, s_h2, s_u2, alpha):
    """Exact observed-scale variance components of the NB-lognormal model."""
    m0 = np.exp(b0)
    EA, EA2 = np.exp(s_h2 / 2.0), np.exp(2.0 * s_h2)
    EC, EC2 = np.exp(s_u2 / 2.0), np.exp(2.0 * s_u2)
    V_h = m0 ** 2 * EC ** 2 * (EA2 - EA ** 2)
    V_u = m0 ** 2 * EA2 * (EC2 - EC ** 2)
    V_r = m0 * EA * EC + alpha * m0 ** 2 * EA2 * EC2
    return float(V_h), float(V_u), float(V_r)


def observed_by_mc(b0, s_h2, s_u2, alpha, n=2_000_000, seed=17):
    """Observed-scale VPC by Monte Carlo over the random effects, using the
    analytic conditional moments.  Targets the population quantity, so it is
    comparable with the closed form; no group-mean estimator is involved."""
    rng = np.random.default_rng(seed)
    A = np.exp(rng.normal(0.0, np.sqrt(s_h2), n))
    C = np.exp(rng.normal(0.0, np.sqrt(s_u2), n))
    m0 = np.exp(b0)
    EC, varC = C.mean(), C.var()
    V_h = m0 ** 2 * EC ** 2 * A.var()
    V_u = m0 ** 2 * (A ** 2).mean() * varC
    m = m0 * A * C
    V_r = m.mean() + alpha * (m ** 2).mean()
    T = V_h + V_u + V_r
    return float(V_h / T), float(V_u / T)


def eta2_on_simulated(o, units_per=2, per_unit=700, n_rep=30, seed=17):
    """What eta-squared returns on data simulated from the fitted model, at the
    cluster sizes of the real cohort.  Upward biased by construction."""
    rng = np.random.default_rng(seed)
    r = 1.0 / o["alpha"]
    hs, us = [], []
    for _ in range(n_rep):
        y, hi, ui = [], [], []
        k = 0
        for h in range(o["n_hospitals"]):
            a = rng.normal(0, np.sqrt(o["var_hospital"]))
            for _ in range(units_per):
                c = rng.normal(0, np.sqrt(o["var_unit"]))
                mu = np.exp(o["intercept"] + a + c)
                y.append(rng.poisson(rng.gamma(r, mu / r, per_unit)))
                hi.append(np.full(per_unit, h))
                ui.append(np.full(per_unit, k))
                k += 1
        y = np.concatenate(y).astype(float)
        hi, ui = np.concatenate(hi), np.concatenate(ui)
        gr = y.mean()
        sst = ((y - gr) ** 2).sum()
        sh = _ss(y, hi, gr) / sst
        hs.append(sh)
        us.append(_ss(y, ui, gr) / sst - sh)
    return (float(np.mean(hs)), float(np.std(hs, ddof=1)),
            float(np.mean(us)), float(np.std(us, ddof=1)))


def _ss(y, g, grand):
    n = np.bincount(g).astype(float)
    s = np.bincount(g, weights=y)
    return float((n * (s / n - grand) ** 2).sum())


# ===================================================== profile likelihood ===
def profile_ci(y, hosp_idx, unit_idx, o, which="hospital", tol=2e-3,
               max_step=12, verbose=True):
    """Profile-likelihood 95% interval for a latent-scale VPC of an NB model.

    Each evaluation re-maximises the remaining parameters from the previous
    solution, so the inner optimisations are short.
    """
    family = o["family"]
    if family == "bernoulli":
        return np.nan, np.nan
    n = len(y)
    Z, q_h, q_u = design(n, hosp_idx, unit_idx)
    mu = o["mean_outcome"]
    ll_max = o["loglik"]
    v_hat = o["vpc_hospital_latent" if which == "hospital"
             else "vpc_unit_latent"]
    other = "var_unit" if which == "hospital" else "var_hospital"
    # If the point fit held the dispersion at the boundary, the profile must
    # hold it there too.  Otherwise ll_max is a constrained maximum while the
    # profile is unconstrained, the deviance is referenced to the wrong
    # optimum, and the interval comes out too wide.
    fix_alpha = ALPHA_FLOOR if o.get("alpha_at_boundary") else None
    mle = [o["intercept"], 0.5 * np.log(max(o[other], 1e-6))]
    if fix_alpha is None:
        mle.append(np.log(max(o["alpha"], ALPHA_FLOOR)))
    mle = np.array(mle)
    warm = {"x": mle.copy()}

    def prof(v):
        if not 1e-6 < v < 0.999:
            return -np.inf

        def nll(free):
            if fix_alpha is None:
                b0, ls_other, la = free
                alpha = float(np.exp(la))
            else:
                b0, ls_other = free
                alpha, la = fix_alpha, np.log(fix_alpha)
            if not np.isfinite(alpha) or not 1e-8 < alpha < 1e4:
                return 1e12
            lvl1 = np.log1p(1.0 / mu + alpha)
            s_other2 = float(np.exp(2 * ls_other))
            s_this2 = v * (s_other2 + lvl1) / (1.0 - v)
            if not np.isfinite(s_this2) or s_this2 <= 0:
                return 1e12
            lt = 0.5 * np.log(s_this2)
            th = ((b0, lt, ls_other, la) if which == "hospital"
                  else (b0, ls_other, lt, la))
            val, _ = laplace(np.array(th), y, Z, q_h, q_u, family)
            return -val if np.isfinite(val) else 1e12

        r = minimize(nll, warm["x"], method="Nelder-Mead",
                     options={"maxiter": 600, "maxfev": 800,
                              "xatol": 1e-5, "fatol": 1e-5})
        if np.isfinite(r.fun) and r.fun < 1e11:
            warm["x"] = r.x
        return -r.fun

    def dev(v):
        return 2.0 * (ll_max - prof(v))

    out = {}
    step0 = max(0.01, 0.12 * v_hat)
    for side, sign, limit in (("lo", -1.0, 1e-4), ("hi", +1.0, 0.995)):
        # Continuation march outward from the maximum, each inner fit warm
        # started from the previous one, then bisect inside the bracket that
        # straddles the threshold.  Jumping straight to a remote v and
        # optimising from the MLE is what failed: the inner fit could not
        # follow, the deviance came back spuriously small, and the bound
        # collapsed onto the point estimate.
        warm["x"] = mle.copy()
        v_prev, d_prev = v_hat, dev(v_hat)
        step = step0
        v_cur, d_cur, bracketed = v_prev, d_prev, False
        for _ in range(40):
            v_cur = v_prev + sign * step
            if (sign < 0 and v_cur <= limit) or (sign > 0 and v_cur >= limit):
                v_cur = limit
            d_cur = dev(v_cur)
            if not np.isfinite(d_cur):
                break
            if d_cur > CHI2_95:
                bracketed = True
                break
            if v_cur == limit:
                break
            if d_cur < d_prev - 1e-6:        # the march found a better optimum
                warm["x"] = mle.copy()
            v_prev, d_prev = v_cur, d_cur
            step = min(step * 1.35, 0.08)
        if not bracketed:
            out[side] = np.nan
            if verbose:
                print(f"        {which} {side}: deviance never crossed "
                      f"{CHI2_95:.2f} out to {v_cur:.3f} — bound unbounded "
                      f"within range, reported as missing")
            continue
        a, b = (min(v_prev, v_cur), max(v_prev, v_cur))
        for _ in range(max_step):
            m = 0.5 * (a + b)
            if dev(m) > CHI2_95:
                if sign > 0:
                    b = m
                else:
                    a = m
            else:
                if sign > 0:
                    a = m
                else:
                    b = m
            if b - a < tol:
                break
        out[side] = 0.5 * (a + b)
        if verbose:
            print(f"        {which} {side} bound {out[side]:.4f}")
    return out["lo"], out["hi"]


# ================================================================ data ======
def metrics(ev, key, off):
    ev = ev.sort_values([key, off])
    g = ev.groupby(key)[off]
    out = pd.DataFrame({"n_records": g.size()})
    ev = ev.assign(gap=g.diff())
    gg = ev.dropna(subset=["gap"]).groupby(key)["gap"]
    out["n_gaps_gt30m"] = gg.apply(lambda s: int((s > 30).sum()))
    out["n_gaps_gt2h"] = gg.apply(lambda s: int((s > 120).sum()))
    return out.reset_index()


def load(a):
    e = metrics(pd.read_parquet(a.eicu_nc_cache), "patientunitstayid",
                "observationoffset")
    pat = pd.read_csv(a.eicu_root / "patient.csv.gz",
                      usecols=["patientunitstayid", "hospitalid", "unittype",
                               "unitdischargeoffset"])
    e = e.merge(pat, on="patientunitstayid", how="inner")
    e = e[(e["unitdischargeoffset"] >= WINDOW_MIN)
          & (e["n_records"] >= MIN_RECORDS)].copy()
    c = e["hospitalid"].value_counts()
    e = e[e["hospitalid"].isin(c[c >= MIN_HOSP].index)].copy()
    med = e.groupby("hospitalid")["n_records"].median()
    er = e[e["hospitalid"].isin(med[med >= FLOOR].index)].copy()
    return e, er


def codes(d):
    hi = pd.factorize(d["hospitalid"])[0].astype(np.int64)
    ui = pd.factorize(d["hospitalid"].astype(str) + ":"
                      + d["unittype"].astype(str))[0].astype(np.int64)
    return hi, ui


def show(o, mc=None, e2=None):
    print(f"\n  {o['label']}  [{o['family']}]")
    print(f"    {o['n']:,} observations, {o['n_hospitals']} hospitals, "
          f"{o['n_units']} hospital-by-unit cells")
    if o.get("fit_failed"):
        print("    *** FIT FAILED from every start — no usable estimates. "
              "Do not report. ***")
        return
    if not o["converged"]:
        print("    *** DID NOT CONVERGE — estimates below are not usable. ***")
    if o.get("alpha_at_boundary"):
        print(f"    *** dispersion fell below {ALPHA_FLOOR:g} and was refitted "
              f"with alpha held there.")
        print("        The data are indistinguishable from the equidispersed "
              "(Poisson) case at")
        print("        this level; report it that way, not as a negative "
              "binomial with alpha = 0. ***")
    tail = (f"   alpha {o['alpha']:.4f}" if o["family"] != "bernoulli" else "")
    print(f"    hospital variance {o['var_hospital']:.4f}   "
          f"unit variance {o['var_unit']:.4f}{tail}")
    if o["family"] == "bernoulli":
        print(f"    level-1 variance  {o['level1_var']:.4f}  = pi^2/3")
    else:
        print(f"    level-1 variance  {o['level1_var']:.4f}  "
              f"= ln(1 + 1/mu + alpha), mu = {o['mu_used']:.4f}")
    print(f"    VPC latent scale       hospital "
          f"{o['vpc_hospital_latent']:.3f}   unit {o['vpc_unit_latent']:.3f}")
    if np.isfinite(o["vpc_hospital_observed"]):
        print(f"    VPC observed, exact    hospital "
              f"{o['vpc_hospital_observed']:.3f}   "
              f"unit {o['vpc_unit_observed']:.3f}")
    if mc:
        print(f"    VPC observed, MC       hospital {mc[0]:.3f}   "
              f"unit {mc[1]:.3f}")
    if e2:
        print(f"    eta-squared on simulated data   hospital {e2[0]:.3f} "
              f"(SD {e2[1]:.3f})   unit {e2[2]:.3f} (SD {e2[3]:.3f})")
        print(f"      the gap from the exact observed-scale value is the "
              f"upward bias of")
        print(f"      eta-squared at {o['n_hospitals']} hospitals, not a "
              f"difference of scale")
    print(f"    log-likelihood {o['loglik']:.1f}  converged={o['converged']}  "
          f"({o['n_iter']} iterations, {o['fit_seconds']:.1f} s)")


# =============================================================== selftest ===
def sim_nb(n_hosp, units_per, per_unit, b0, s_h, s_u, alpha, seed):
    """Returns the data and the REALISED variance of the drawn random effects,
    which is the quantity a single fit can be held to, rather than the
    generating parameter."""
    rng = np.random.default_rng(seed)
    r = 1.0 / alpha
    y, hi, ui, aa, cc = [], [], [], [], []
    k = 0
    for h in range(n_hosp):
        a = rng.normal(0, s_h)
        aa.append(a)
        for _ in range(units_per):
            c = rng.normal(0, s_u)
            cc.append(c)
            mu = np.exp(b0 + a + c)
            y.append(rng.poisson(rng.gamma(r, mu / r, per_unit)))
            hi.append(np.full(per_unit, h))
            ui.append(np.full(per_unit, k))
            k += 1
    return (np.concatenate(y).astype(float), np.concatenate(hi),
            np.concatenate(ui), float(np.var(aa)), float(np.var(cc)))


def sim_hurdle(n_hosp, units_per, per_unit, T2, zz, seed):
    """Simulate a hurdle process and return each part's data together with the
    realised variance of the drawn random effects for that part."""
    rng = np.random.default_rng(seed)
    r = 1.0 / T2["alpha"]
    logit0 = np.log(zz["p0"] / (1 - zz["p0"]))
    yz, hz, uz, yt, ht, ut = ([] for _ in range(6))
    ab_, cb_, az_, cz_ = [], [], [], []
    k = 0
    for h in range(n_hosp):
        ab, az = rng.normal(0, zz["s_hb"]), rng.normal(0, T2["s_h"])
        ab_.append(ab)
        az_.append(az)
        for _ in range(units_per):
            cb, cz = rng.normal(0, zz["s_ub"]), rng.normal(0, T2["s_u"])
            cb_.append(cb)
            cz_.append(cz)
            pos = rng.random(per_unit) < expit(logit0 + ab + cb)
            yz.append(pos.astype(float))
            hz.append(np.full(per_unit, h))
            uz.append(np.full(per_unit, k))
            npos = int(pos.sum())
            if npos:
                mu = np.exp(T2["b0"] + az + cz)
                v = np.zeros(0, int)
                while len(v) < npos:
                    d = rng.poisson(rng.gamma(r, mu / r, npos * 3))
                    v = np.concatenate([v, d[d > 0]])
                yt.append(v[:npos].astype(float))
                ht.append(np.full(npos, h))
                ut.append(np.full(npos, k))
            k += 1
    return {"bernoulli": (np.concatenate(yz), np.concatenate(hz),
                          np.concatenate(uz), float(np.var(ab_)),
                          float(np.var(cb_))),
            "ztnb": (np.concatenate(yt), np.concatenate(ht),
                     np.concatenate(ut), float(np.var(az_)),
                     float(np.var(cz_)))}


def selftest(profile=False):
    print("=" * 78)
    print("SELF-TEST 1 — the closed-form observed-scale VPC against Monte Carlo")
    print("=" * 78)
    print("  Both target the population quantity, so they must agree to MC error.")
    print(f"\n  {'s_h2':>6s} {'s_u2':>6s} {'alpha':>6s} "
          f"{'closed h':>9s} {'MC h':>9s} {'closed u':>9s} {'MC u':>9s} "
          f"{'max diff':>9s}")
    worst = 0.0
    for s_h2, s_u2, al in [(0.176, 0.040, 0.35), (0.040, 0.090, 0.15),
                           (0.360, 0.010, 0.80), (0.104, 0.034, 0.15),
                           (0.519, 0.031, 0.15)]:
        b0 = np.log(6.0)
        Vh, Vu, Vr = observed_components(b0, s_h2, s_u2, al)
        T = Vh + Vu + Vr
        mh, mu_ = observed_by_mc(b0, s_h2, s_u2, al, n=1_000_000)
        d = max(abs(Vh / T - mh), abs(Vu / T - mu_))
        worst = max(worst, d)
        print(f"  {s_h2:6.3f} {s_u2:6.3f} {al:6.2f} {Vh / T:9.4f} {mh:9.4f} "
              f"{Vu / T:9.4f} {mu_:9.4f} {d:9.5f}")
    print(f"\n  worst disagreement {worst:.5f} "
          f"{'— algebra confirmed' if worst < 2e-3 else '— CHECK THE ALGEBRA'}")

    print("\n" + "=" * 78)
    print("SELF-TEST 2 — negative binomial GLMM recovers the realised variances")
    print("=" * 78)
    T = dict(b0=np.log(6.0), s_h=0.42, s_u=0.20, alpha=0.35)
    print(f"  generating: hospital var {T['s_h']**2:.4f}, unit var "
          f"{T['s_u']**2:.4f}, alpha {T['alpha']:.4f}")
    print(f"\n  {'rep':>3s} {'realised h':>11s} {'fitted h':>9s} "
          f"{'realised u':>11s} {'fitted u':>9s} {'alpha':>7s} {'s':>6s}")
    eh, eu = [], []
    for rep in range(5):
        y, hi, ui, rh, ru = sim_nb(40, 2, 350, T["b0"], T["s_h"], T["s_u"],
                                   T["alpha"], 100 + rep)
        o = fit(y, hi, ui, "nb", "")
        eh.append(o["var_hospital"] - rh)
        eu.append(o["var_unit"] - ru)
        print(f"  {rep:3d} {rh:11.4f} {o['var_hospital']:9.4f} "
              f"{ru:11.4f} {o['var_unit']:9.4f} {o['alpha']:7.4f} "
              f"{o['fit_seconds']:6.1f}")
    print(f"\n  mean error against the realised variance: hospital "
          f"{np.mean(eh):+.4f}, unit {np.mean(eu):+.4f}")

    print("\n" + "=" * 78)
    print("SELF-TEST 3 — the two hurdle parts")
    print("=" * 78)
    print("  Both parts are checked over several replicates against the")
    print("  REALISED variance of the drawn effects.  A single replicate is not")
    print("  enough: with 30 hospitals the sampling spread of an estimated")
    print("  variance is wide, and one draw can sit 30% off either way without")
    print("  anything being wrong with the fitter.")
    zz = dict(s_hb=0.50, s_ub=0.25, p0=0.30)
    T2 = dict(b0=np.log(2.2), s_h=0.40, s_u=0.18, alpha=0.30)
    print(f"\n  {'part':14s} {'rep':>3s} {'n':>8s} {'real h':>8s} "
          f"{'fit h':>8s} {'real u':>8s} {'fit u':>8s} {'alpha':>7s}")
    agg = {"bernoulli": [[], []], "ztnb": [[], []]}
    detail = {}
    for rep in range(3):
        parts = sim_hurdle(30, 2, 400, T2, zz, 300 + rep)
        for fam in ("bernoulli", "ztnb"):
            y, h, u, rh, ru = parts[fam]
            o = fit(y, h, u, fam, f"simulated hurdle, {fam}")
            agg[fam][0].append(o["var_hospital"] - rh)
            agg[fam][1].append(o["var_unit"] - ru)
            detail.setdefault(fam, o)
            al = (f"{o['alpha']:7.4f}" if fam != "bernoulli" else "      -")
            print(f"  {fam:14s} {rep:3d} {len(y):8,} {rh:8.4f} "
                  f"{o['var_hospital']:8.4f} {ru:8.4f} {o['var_unit']:8.4f} "
                  f"{al}")
    for fam in ("bernoulli", "ztnb"):
        print(f"  {fam:14s} mean error vs realised: hospital "
              f"{np.mean(agg[fam][0]):+.4f}   unit {np.mean(agg[fam][1]):+.4f}")
    print("\n  One fitted model of each part, in full:")
    show(detail["bernoulli"])
    ot = detail["ztnb"]
    show(ot, mc=observed_by_mc(ot["intercept"], ot["var_hospital"],
                               ot["var_unit"], ot["alpha"], n=500_000))

    if profile:
        print("\n" + "=" * 78)
        print("SELF-TEST 4 — profile-likelihood interval and its cost")
        print("=" * 78)
        y, hi, ui, rh, ru = sim_nb(40, 2, 350, T["b0"], T["s_h"], T["s_u"],
                                   T["alpha"], 100)
        o = fit(y, hi, ui, "nb", "")
        truth = rh / (rh + ru + np.log1p(1 / y.mean() + T["alpha"]))
        t0 = time.time()
        lo, up = profile_ci(y, hi, ui, o, "hospital")
        print(f"  hospital VPC {o['vpc_hospital_latent']:.3f} "
              f"({lo:.3f}-{up:.3f})   realised-variance target {truth:.3f}   "
              f"covered={bool(lo <= truth <= up)}")
        print(f"  {time.time() - t0:.0f} s at n={len(y):,}; cohort scale is "
              f"about {88560 / len(y):.1f}x the data per fit")


# =================================================================== main ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eicu-nc-cache", type=Path)
    ap.add_argument("--eicu-root", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./count_models"))
    ap.add_argument("--profile-ci", action="store_true")
    ap.add_argument("--profile-all", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest(profile=a.profile_ci or a.profile_all)
        return
    for r in ("eicu_nc_cache", "eicu_root"):
        if getattr(a, r) is None:
            ap.error(f"--{r.replace('_', '-')} is required")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    eb, er = load(a)
    rows = []
    for cohort, d in (("restricted", er), ("unrestricted", eb)):
        hi, ui = codes(d)
        print("\n" + "=" * 78)
        print(f"{cohort.upper()} COHORT — {len(d):,} stays, "
              f"{d['hospitalid'].nunique()} hospitals")
        print("=" * 78)
        print("  Distributional grounds for the model choice:")
        for col in ("n_records", "n_gaps_gt30m", "n_gaps_gt2h"):
            v = d[col].to_numpy(float)
            print(f"    {col:14s} mean {v.mean():8.3f}  var/mean "
                  f"{v.var(ddof=1) / max(v.mean(), 1e-9):7.2f}  "
                  f"zeros {np.mean(v == 0):6.1%}  max {v.max():.0f}")

        want_ci = a.profile_all or (a.profile_ci and cohort == "restricted")
        for col, name in (("n_records", "Record count"),
                          ("n_gaps_gt30m", "Gaps >30 min")):
            y = d[col].to_numpy(float)
            o = fit(y, hi, ui, "nb", f"{name} ({cohort})")
            mc = observed_by_mc(o["intercept"], o["var_hospital"],
                                o["var_unit"], o["alpha"])
            e2 = eta2_on_simulated(o)
            show(o, mc, e2)
            rec = {k: v for k, v in o.items() if k != "theta"}
            rec.update({"cohort": cohort, "metric": name, "part": "single",
                        "vpc_hospital_mc": mc[0], "vpc_unit_mc": mc[1],
                        "eta2_sim_hospital": e2[0], "eta2_sim_unit": e2[2]})
            if want_ci and (a.profile_all or col == "n_records"):
                print("    profile-likelihood 95% intervals, latent scale:")
                for w in ("hospital", "unit"):
                    lo, up = profile_ci(y, hi, ui, o, w)
                    rec[f"vpc_{w}_lo"], rec[f"vpc_{w}_hi"] = lo, up
            rows.append(rec)

        y = d["n_gaps_gt2h"].to_numpy(float)
        print(f"\n  Gaps >2 h is {np.mean(y == 0):.1%} zeros: hurdle "
              f"specification.")
        pr = pd.DataFrame({"h": hi, "any": (y > 0).astype(float)}) \
            .groupby("h")["any"].mean()
        n01 = int(((pr < 0.01) | (pr > 0.99)).sum())
        print(f"    per-hospital share with any gap: min {pr.min():.3f}, "
              f"median {pr.median():.3f}, max {pr.max():.3f}; "
              f"{n01} of {len(pr)} hospitals at 0 or 1")
        if n01:
            print("    Hospitals at 0 or 1 drive the logit random intercept "
                  "toward separation,")
            print("    so a large hospital variance in the Bernoulli part "
                  "reflects those sites")
            print("    rather than a well-identified variance component.")
        ob = fit((y > 0).astype(float), hi, ui, "bernoulli",
                 f"Gaps >2 h, whether any ({cohort})")
        show(ob)
        pos = y > 0
        ot = fit(y[pos], hi[pos], ui[pos], "ztnb",
                 f"Gaps >2 h, how many given >=1 ({cohort})")
        show(ot, mc=observed_by_mc(ot["intercept"], ot["var_hospital"],
                                   ot["var_unit"], ot["alpha"]))
        print("    The hurdle model has one variance partition per part and no")
        print("    single pooled value; both are reported and neither is "
              "presented")
        print("    as the VPC for the metric.")
        for o2, part in ((ob, "hurdle: whether any"),
                         (ot, "hurdle: how many given >=1")):
            rec = {k: v for k, v in o2.items() if k != "theta"}
            rec.update({"cohort": cohort, "metric": "Gaps >2 h", "part": part})
            if a.profile_all and o2["family"] == "ztnb":
                print(f"    profile-likelihood 95% intervals, {part}:")
                for w in ("hospital", "unit"):
                    lo, up = profile_ci(y[pos], hi[pos], ui[pos], o2, w)
                    rec[f"vpc_{w}_lo"], rec[f"vpc_{w}_hi"] = lo, up
            rows.append(rec)

    pd.DataFrame(rows).to_csv(a.out_dir / "count_models.csv", index=False)
    print(f"\n-> {a.out_dir}")


if __name__ == "__main__":
    main()
