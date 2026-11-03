"""Model evaluation with subject-wise validation.

Every model is scored with participants held out as whole groups, so no
person contributes data to both training and test folds. A record-wise split
is computed alongside where it illustrates how much a naive split inflates
results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


def classifiers() -> dict:
    return {
        "Logistic regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(C=1.0, max_iter=5000)),
        "Random forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                                              random_state=C.SEED, n_jobs=-1)),
    }


def regressors() -> dict:
    return {
        "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "Random forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                                             random_state=C.SEED, n_jobs=-1)),
    }


def cv_predict(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None, n_splits: int = 10,
               proba: bool = False, seed: int = C.SEED) -> np.ndarray:
    """Out-of-fold predictions. groups=None gives a record-wise (shuffled) split."""
    if groups is None:
        cv = (StratifiedKFold(n_splits, shuffle=True, random_state=seed) if not np.issubdtype(np.asarray(y).dtype, np.floating)
              else KFold(n_splits, shuffle=True, random_state=seed))
        splits = cv.split(X, y)
    else:
        splits = GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(X, y, groups)
    pred = None
    for tr, te in splits:
        m = clone(model).fit(X[tr], y[tr])
        p = m.predict_proba(X[te]) if proba else m.predict(X[te])
        if pred is None:
            pred = np.empty((len(y),) + p.shape[1:], dtype=p.dtype if not proba else float)
            if not proba and pred.dtype.kind in "US":
                pred = np.empty(len(y), dtype=object)
        pred[te] = p
    return pred


def score_classification(y, yhat, groups) -> dict:
    out = {"accuracy": float(np.mean(yhat == y)), "balanced_accuracy": balanced_accuracy_score(y, yhat),
           "macro_f1": f1_score(y, yhat, average="macro")}
    # subject-level bootstrap CI of accuracy
    from .stats import bootstrap_ci
    lo, hi = bootstrap_ci((yhat == y).astype(float), groups)
    out.update({"acc_ci_lo": lo, "acc_ci_hi": hi})
    return out


def score_regression(y, yhat, groups) -> dict:
    from .stats import bootstrap_ci
    err = np.abs(yhat - y)
    lo, hi = bootstrap_ci(err, groups)
    return {"r2": r2_score(y, yhat), "mae": mean_absolute_error(y, yhat), "mae_ci_lo": lo, "mae_ci_hi": hi}


def permutation_null(model, X, y, groups, n_perm: int = 50, seed: int = C.SEED) -> np.ndarray:
    """Subject-wise CV accuracy with labels shuffled within each subject, which
    keeps the label distribution per person but breaks any feature-label link."""
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_perm):
        yp = y.copy()
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            yp[idx] = rng.permutation(yp[idx])
        accs.append(np.mean(cv_predict(model, X, yp, groups) == yp))
    return np.array(accs)
