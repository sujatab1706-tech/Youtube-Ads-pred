"""Check the live model for data drift and falling accuracy, and decide whether to retrain.

Compares recent API traffic (logs/predictions.jsonl + logs/feedback.jsonl) with the training data:
  1. Data drift      - do incoming ads look different from the training ads? (Evidently tests)
  2. Prediction drift - are the model's predictions shifting?
  3. Performance      - for ads whose real clicks are known, is the error worse than at training time?

Writes an interactive HTML report and a JSON summary. The summary's `retrain_needed`
flag is what the GitHub Actions workflow uses to decide whether to retrain.

Usage:
    python src/monitor.py --log-dir logs --out reports
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import DataDriftPreset, RegressionPreset
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent))
from features import CATEGORICAL, NUMERIC, add_features, clean  # noqa: E402

RAW_NUMERIC = ["hour", "ad_length_sec", "daily_budget_usd", "impressions", "video_watch_pct", "thumbnail_quality"]
RAW_CATEGORICAL = ["category", "device", "ad_format", "age_group"]
MONITORED = RAW_NUMERIC + RAW_CATEGORICAL

# Retraining rules. Tune these to how much risk the business accepts.
DRIFT_SHARE_LIMIT = 0.30  # retrain if 30%+ of monitored features have drifted
MAE_RATIO_LIMIT = 1.25    # retrain if live error is 25%+ worse than at training time
MIN_FEEDBACK_ROWS = 200   # don't judge accuracy on too few examples

warnings.filterwarnings("ignore", message="R\\^2 score is not well-defined")


def load_reference(data_path: Path, model) -> pd.DataFrame:
    df = add_features(clean(pd.read_csv(data_path)))
    df["prediction"] = np.clip(model.predict(df[NUMERIC + CATEGORICAL]), 0, None)
    return df.rename(columns={"clicks": "target"})


def load_current(log_dir: Path) -> pd.DataFrame:
    preds = pd.read_json(log_dir / "predictions.jsonl", lines=True)
    preds = preds.rename(columns={"predicted_clicks": "prediction"})
    fb_path = log_dir / "feedback.jsonl"
    if fb_path.exists():
        fb = pd.read_json(fb_path, lines=True)[["prediction_id", "actual_clicks"]].drop_duplicates("prediction_id", keep="last")
        preds = preds.merge(fb, on="prediction_id", how="left")
    else:
        preds["actual_clicks"] = np.nan
    return preds.rename(columns={"actual_clicks": "target"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/youtube_ads.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/model.joblib"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    model = joblib.load(args.model)
    train_metrics = json.loads(args.model.with_name("metrics.json").read_text())
    ref_mae = train_metrics["test_metrics"][train_metrics["best_model"]]["mae"]

    reference = load_reference(args.data, model)
    current = load_current(args.log_dir)
    labelled = current.dropna(subset=["target"])
    print(f"Reference rows: {len(reference):,}   Live predictions: {len(current):,}   With real clicks: {len(labelled):,}")

    # 1 + 2: data and prediction drift on everything the API received.
    drift_cols = MONITORED + ["prediction"]
    definition = DataDefinition(numerical_columns=RAW_NUMERIC + ["prediction"], categorical_columns=RAW_CATEGORICAL)
    drift_snap = Report([DataDriftPreset(columns=drift_cols)]).run(
        reference_data=Dataset.from_pandas(reference[drift_cols], data_definition=definition),
        current_data=Dataset.from_pandas(current[drift_cols], data_definition=definition),
    )
    drifted = []
    for m in drift_snap.dict()["metrics"]:
        name = m["metric_name"]
        if name.startswith("ValueDrift(column="):
            col = name.split("column=")[1].split(",")[0]
            method = name.split("method=")[1].split(",")[0]
            threshold = float(name.split("threshold=")[1].rstrip(")"))
            p_or_dist = float(m["value"])
            # p-value tests drift when small; distance tests (Wasserstein, Jensen-Shannon) drift when large
            is_drift = p_or_dist < threshold if "p_value" in method else p_or_dist > threshold
            if is_drift:
                drifted.append(col)
    feature_drift = [c for c in drifted if c != "prediction"]
    drift_share = len(feature_drift) / len(MONITORED)

    # 3: performance on ads with known outcomes.
    perf = {}
    if len(labelled) >= MIN_FEEDBACK_ROWS:
        perf = {
            "live_mae": float(mean_absolute_error(labelled["target"], labelled["prediction"])),
            "live_r2": float(r2_score(labelled["target"], labelled["prediction"])),
            "mean_actual_clicks": float(labelled["target"].mean()),
            "mean_predicted_clicks": float(labelled["prediction"].mean()),
        }
        perf["mae_ratio_vs_training"] = perf["live_mae"] / ref_mae

    reasons = []
    if drift_share >= DRIFT_SHARE_LIMIT:
        reasons.append(f"{drift_share:.0%} of features drifted (limit {DRIFT_SHARE_LIMIT:.0%}): {', '.join(feature_drift)}")
    if perf and perf["mae_ratio_vs_training"] >= MAE_RATIO_LIMIT:
        reasons.append(f"live MAE {perf['live_mae']:.1f} is {perf['mae_ratio_vs_training']:.2f}x the training MAE {ref_mae:.1f}")

    summary = {
        "live_rows": len(current),
        "labelled_rows": len(labelled),
        "drifted_features": feature_drift,
        "drift_share": round(drift_share, 3),
        "prediction_drift": "prediction" in drifted,
        "training_mae": ref_mae,
        **{k: round(v, 3) for k, v in perf.items()},
        "retrain_needed": bool(reasons),
        "reasons": reasons,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    drift_snap.save_html(str(args.out / "data_drift_report.html"))
    if len(labelled) >= MIN_FEEDBACK_ROWS:
        reg_def = DataDefinition(numerical_columns=["target", "prediction"],
                                 regression=[Regression(target="target", prediction="prediction")])
        ref_sample = reference[["target", "prediction"]].sample(min(5000, len(reference)), random_state=0)
        Report([RegressionPreset()]).run(
            reference_data=Dataset.from_pandas(ref_sample, data_definition=reg_def),
            current_data=Dataset.from_pandas(labelled[["target", "prediction"]], data_definition=reg_def),
        ).save_html(str(args.out / "model_performance_report.html"))
    (args.out / "monitoring_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print("\nRETRAIN NEEDED" if reasons else "\nModel healthy: no retraining needed")


if __name__ == "__main__":
    main()
