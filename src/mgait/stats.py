"""Statistics used across the analysis.

- Linear mixed models with a random intercept per subject (repeated measures
  across the three speeds).
- Holm correction for families of tests.
- A paired, cluster-based permutation test on time-normalised curves (the
  non-parametric counterpart of statistical parametric mapping).
- Agreement statistics for method comparison (Bland-Altman, ICC(A,1)).
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


class _Fit:
    """Minimal fitted-model view with coefficients back in the outcome's units."""

    def __init__(self, params, ci, pvalues):
        self.params, self._ci, self.pvalues = params, ci, pvalues

    def conf_int(self):
        return self._ci


def mixed_model(df: pd.DataFrame, formula: str, group: str = "subject"):
    """Random-intercept mixed model. The outcome (left of ~) is standardised
    before fitting for numerical stability; coefficients and intervals are
    returned in the original units."""
    yname = formula.split("~")[0].strip()
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", formula))
    used = [c for c in df.columns if c in tokens] + [group]
    d = df[list(dict.fromkeys(used))].replace([np.inf, -np.inf], np.nan).dropna().copy()
    mu, sd = d[yname].mean(), d[yname].std()
    d[yname] = (d[yname] - mu) / sd
    last = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("lbfgs", "powell", "nm"):
            try:
                r = smf.mixedlm(formula, d, groups=d[group]).fit(reml=True, method=method)
                if np.all(np.isfinite(r.params)):
                    break
            except Exception as e:  # singular fits: try the next optimiser
                last = e
        else:
            raise last
    fe = r.fe_params.index
    params = r.params[fe] * sd
    params.iloc[list(fe).index("Intercept")] += mu if "Intercept" in fe else 0
    ci = r.conf_int().loc[fe] * sd
    return _Fit(params, ci, r.pvalues[fe])


def speed_effects(T: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """For each feature: mixed model feature ~ speed (continuous, m/s) with a
    random intercept per subject. Reports the change per +0.5 m/s, its 95 % CI,
    the standardised effect (per +0.5 m/s, in between-trial SD units) and the
    Holm-adjusted p-value."""
    rows = []
    for f in features:
        d = T[["subject", "speed_ms", f]].dropna().rename(columns={f: "y"})
        if d["subject"].nunique() < 10:
            continue
        r = mixed_model(d, "y ~ speed_ms")
        b, (lo, hi) = r.params["speed_ms"], r.conf_int().loc["speed_ms"]
        sd = d["y"].std()
        rows.append({"feature": f, "n_subjects": d["subject"].nunique(), "n_trials": len(d),
                     "mean_0.5": d.loc[d.speed_ms == 0.5, "y"].mean(), "mean_1.0": d.loc[d.speed_ms == 1.0, "y"].mean(),
                     "per_0.5ms": 0.5 * b, "ci_lo": 0.5 * lo, "ci_hi": 0.5 * hi,
                     "std_effect": 0.5 * b / sd if sd else np.nan, "p": r.pvalues["speed_ms"]})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy())
    return out


def paired_cluster_test(A: np.ndarray, B: np.ndarray, n_perm: int = 5000, alpha: float = 0.05,
                        seed: int = 0) -> dict:
    """Paired comparison of curves (subjects x points) with sign-flip
    permutations and cluster-mass inference over the curve domain."""
    rng = np.random.default_rng(seed)
    D = A - B
    n = D.shape[0]

    def tcurve(X):
        return X.mean(0) / (X.std(0, ddof=1) / np.sqrt(n) + 1e-12)

    def clusters(t, thr):
        above = np.abs(t) > thr
        out, i = [], 0
        while i < len(t):
            if above[i]:
                j = i
                while j + 1 < len(t) and above[j + 1] and np.sign(t[j + 1]) == np.sign(t[i]):
                    j += 1
                out.append((i, j, float(np.abs(t[i:j + 1]).sum())))
                i = j + 1
            else:
                i += 1
        return out

    from scipy import stats as st
    thr = st.t.ppf(1 - alpha / 2, n - 1)
    t_obs = tcurve(D)
    obs = clusters(t_obs, thr)
    null = np.zeros(n_perm)
    for k in range(n_perm):
        s = rng.choice([-1, 1], size=(n, 1))
        cl = clusters(tcurve(D * s), thr)
        null[k] = max((c[2] for c in cl), default=0.0)
    res = [{"start": a, "end": b, "mass": m, "p": float((null >= m).mean())} for a, b, m in obs]
    return {"t": t_obs, "threshold": thr, "clusters": res, "mean_diff": D.mean(0)}


def bland_altman(ref: np.ndarray, test: np.ndarray) -> dict:
    d = np.asarray(test, float) - np.asarray(ref, float)
    bias, sd = np.nanmean(d), np.nanstd(d, ddof=1)
    return {"bias": bias, "loa_lo": bias - 1.96 * sd, "loa_hi": bias + 1.96 * sd, "sd": sd, "n": int(np.isfinite(d).sum())}


def icc_a1(ref: np.ndarray, test: np.ndarray) -> float:
    """ICC(A,1): two-way, absolute agreement, single measurement (McGraw & Wong)."""
    Y = np.column_stack([ref, test]).astype(float)
    Y = Y[np.isfinite(Y).all(1)]
    n, k = Y.shape
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def bootstrap_ci(values: np.ndarray, groups: np.ndarray, stat=np.mean, n_boot: int = 2000, seed: int = 0):
    """Percentile CI resampling whole groups (subjects)."""
    rng = np.random.default_rng(seed)
    values, groups = np.asarray(values), np.asarray(groups)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        boots.append(stat(values[np.concatenate([idx[g] for g in pick])]))
    return np.percentile(boots, [2.5, 97.5])
