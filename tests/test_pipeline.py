"""Fast checks that the data, features and model behave as expected."""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from features import CATEGORICAL, NUMERIC, add_features, clean  # noqa: E402
from generate_data import generate  # noqa: E402
from train import build_models, evaluate  # noqa: E402


def small_dataset(rows=2000):
    df = add_features(clean(generate(rows, seed=0)))
    return df[NUMERIC + CATEGORICAL], df["clicks"]


def test_generator_is_reproducible():
    assert generate(100, seed=1).equals(generate(100, seed=1))


def test_clean_removes_duplicates_and_bad_rows():
    df = clean(generate(1000, seed=0))
    assert df["ad_id"].is_unique
    assert (df["clicks"] <= df["impressions"]).all()


def test_features_have_no_missing_categories():
    X, _ = small_dataset()
    assert X[CATEGORICAL].notna().all().all()
    assert set(X["time_of_day"]) <= {"night", "morning", "afternoon", "evening"}


def test_log_model_beats_baseline():
    X, y = small_dataset()
    models = build_models()
    scores = {}
    for name in ["baseline", "linear_log"]:
        models[name].fit(X, y)
        scores[name] = evaluate(y, models[name].predict(X))["mae"]
    assert scores["linear_log"] < 0.5 * scores["baseline"]


def test_model_handles_missing_values_and_new_categories():
    X, y = small_dataset()
    model = build_models()["linear_log"].fit(X, y)
    row = X.iloc[[0]].copy()
    row["video_watch_pct"] = np.nan
    row["category"] = "Music"  # never seen in training
    pred = model.predict(row)
    assert np.isfinite(pred).all() and pred[0] >= 0


def test_logistic_ctr_predicts_valid_click_rates():
    from compare_models import LogisticCTR

    X, y = small_dataset()
    model = LogisticCTR(C=1e4).fit(X, y)
    ctr = model.predict_ctr(X)
    assert ((ctr > 0) & (ctr < 1)).all()
    assert abs(model.predict(X).sum() / y.sum() - 1) < 0.1  # total clicks roughly right
