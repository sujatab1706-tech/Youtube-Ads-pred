"""Train and evaluate linear regression models that predict ad clicks.

Three models are compared on the same held-out test set:
  1. baseline        - always predicts the average clicks (any real model must beat this)
  2. linear_raw      - linear regression predicting clicks directly
  3. linear_log      - linear regression predicting log(1 + clicks), converted back to clicks

All metrics are reported on the original clicks scale so the models are comparable.
The best model is retrained on all the data and saved to models/model.joblib.

Usage:
    python src/train.py --data data/youtube_ads.csv --out models
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from features import CATEGORICAL, NUMERIC, TARGET, add_features, build_preprocessor, clean

SEED = 42


def build_models() -> dict:
    return {
        "baseline": Pipeline([("prep", build_preprocessor()), ("model", DummyRegressor(strategy="mean"))]),
        "linear_raw": Pipeline([("prep", build_preprocessor()), ("model", LinearRegression())]),
        "linear_log": TransformedTargetRegressor(
            regressor=Pipeline([("prep", build_preprocessor()), ("model", LinearRegression())]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "median_abs_pct_error": float(np.median(np.abs(y_true - y_pred) / np.maximum(y_true, 1)) * 100),
    }


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = add_features(clean(pd.read_csv(path)))
    return df[NUMERIC + CATEGORICAL], df[TARGET]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/youtube_ads.csv"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"Train rows: {len(X_train):,}   Test rows: {len(X_test):,}\n")

    results = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_test), 0, None)  # clicks can't be negative
        results[name] = evaluate(y_test, pred)

    print(f"{'model':<12}{'RMSE':>10}{'MAE':>10}{'R2':>8}{'MdAPE %':>10}")
    for name, m in results.items():
        print(f"{name:<12}{m['rmse']:>10.1f}{m['mae']:>10.1f}{m['r2']:>8.3f}{m['median_abs_pct_error']:>10.1f}")

    best = min((n for n in results if n != "baseline"), key=lambda n: results[n]["mae"])
    print(f"\nBest model by MAE: {best}")

    # 5-fold cross-validation checks the score isn't a lucky split.
    cv = cross_val_score(build_models()[best], X, y, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2")
    print(f"5-fold CV R2: {cv.mean():.3f} ± {cv.std():.3f}")

    final = build_models()[best].fit(X, y)
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, args.out / "model.joblib")
    metrics = {
        "best_model": best,
        "test_metrics": results,
        "cv_r2_mean": float(cv.mean()),
        "cv_r2_std": float(cv.std()),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": NUMERIC + CATEGORICAL,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved {args.out / 'model.joblib'} and {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
