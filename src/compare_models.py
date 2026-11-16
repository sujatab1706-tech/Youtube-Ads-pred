"""Compare linear regression with other models on the same train/test split.

Models compared (all scored on the original clicks scale):
  linear_log        - Step 2's chosen model: linear regression on log(1 + clicks)
  ridge_log         - linear regression with L2 regularisation (shrinks correlated coefficients)
  lasso_log         - linear regression with L1 regularisation (can set useless features to zero)
  poisson_glm       - Poisson regression: a linear model built for count data
  logistic_ctr      - logistic regression predicting the chance that one impression becomes a
                      click; clicks = predicted chance x impressions
  gradient_boosting - histogram gradient boosted trees (non-linear, learns interactions)

Usage:
    python src/compare_models.py --data data/youtube_ads.csv --out models/comparison.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, PoissonRegressor, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features import build_preprocessor
from train import SEED, evaluate, load_dataset


def log_target(regressor):
    """Fit the regressor on log(1 + clicks) and convert predictions back to clicks."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", build_preprocessor()), ("model", regressor)]),
        func=np.log1p,
        inverse_func=np.expm1,
    )


class LogisticCTR(BaseEstimator, RegressorMixin):
    """Logistic regression on click-through rate.

    Each ad is treated as `impressions` yes/no trials. We give the model two weighted rows per
    ad: one labelled "click" weighted by clicks, one labelled "no click" weighted by
    impressions - clicks. The model learns P(click | ad features); clicks = P x impressions.
    """

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        impressions = np.expm1(X["log_impressions"]).round().to_numpy()
        y = np.asarray(y, dtype=float)
        X2 = pd.concat([X, X], ignore_index=True)
        labels = np.r_[np.ones(len(X)), np.zeros(len(X))]
        weights = np.r_[y, impressions - y]
        weights = weights / weights.mean()  # keep the optimiser well scaled
        self.pipe_ = Pipeline([
            ("prep", build_preprocessor()),
            ("model", LogisticRegression(C=self.C, max_iter=2000)),
        ])
        self.pipe_.fit(X2, labels, model__sample_weight=weights)
        return self

    def predict_ctr(self, X):
        return self.pipe_.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.predict_ctr(X) * np.expm1(X["log_impressions"]).to_numpy()


def build_models() -> dict:
    alphas = np.logspace(-4, 2, 30)
    return {
        "linear_log": log_target(LinearRegression()),
        "ridge_log": log_target(RidgeCV(alphas=alphas)),
        "lasso_log": log_target(LassoCV(alphas=alphas, cv=5, random_state=SEED)),
        "poisson_glm": Pipeline([("prep", build_preprocessor()), ("model", PoissonRegressor(alpha=1e-6, max_iter=1000))]),
        "logistic_ctr": LogisticCTR(C=1e4),
        "gradient_boosting": log_target(HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=SEED)),
    }


def compare(X_train, X_test, y_train, y_test) -> tuple[pd.DataFrame, dict]:
    rows, fitted = [], {}
    for name, model in build_models().items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        seconds = time.perf_counter() - start
        pred = np.clip(model.predict(X_test), 0, None)
        rows.append({"model": name, **evaluate(y_test, pred), "train_seconds": seconds})
        fitted[name] = model
    return pd.DataFrame(rows).set_index("model"), fitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/youtube_ads.csv"))
    parser.add_argument("--out", type=Path, default=Path("models/comparison.json"))
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    results, _ = compare(X_train, X_test, y_train, y_test)
    results = results.sort_values("mae")
    print(results.round(3).to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results.round(4).to_dict(orient="index"), indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
