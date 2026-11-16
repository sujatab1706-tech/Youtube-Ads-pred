"""Retrain the model on recent traffic and only replace it if the new one is better.

This is a "champion vs challenger" check:
  champion   - the model currently in production (models/model.joblib)
  challenger - a new model trained on the original data plus recent labelled traffic

Recent ads are split 70/30. The challenger trains on the original data plus 70% of the
recent ads (repeated so recent behaviour carries more weight). Both models are then scored
on the 30% of recent ads that neither has seen. The challenger is promoted only if it wins.

Usage:
    python src/retrain.py --log-dir logs
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
from features import CATEGORICAL, NUMERIC, TARGET, add_features, clean  # noqa: E402
from monitor import load_current  # noqa: E402
from train import SEED, build_models, evaluate  # noqa: E402

# Each recent ad counts as this many rows. With ~2,000 recent ads vs 20,000 historical ones,
# 30 makes recent behaviour outweigh history about 3:1, so the model adapts to the new world
# while still learning from the large historical dataset. 3 adapted too little in testing.
RECENT_WEIGHT = 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/youtube_ads.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/model.joblib"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--recent-weight", type=int, default=RECENT_WEIGHT)
    args = parser.parse_args()

    history = add_features(clean(pd.read_csv(args.data)))
    live = load_current(args.log_dir).dropna(subset=["target"]).rename(columns={"target": TARGET})
    live["is_weekend"] = (live["day_of_week"] >= 5).astype(int)
    live = add_features(live)
    if len(live) < 200:
        sys.exit(f"Only {len(live)} labelled recent ads; need at least 200 to retrain safely.")

    cols = NUMERIC + CATEGORICAL
    live_train, live_test = train_test_split(live, test_size=0.3, random_state=SEED)
    train = pd.concat([history] + [live_train] * args.recent_weight, ignore_index=True)

    champion = joblib.load(args.model)
    challenger = build_models()["linear_log"].fit(train[cols], train[TARGET])

    y = live_test[TARGET].to_numpy()
    scores = {
        "champion": evaluate(y, np.clip(champion.predict(live_test[cols]), 0, None)),
        "challenger": evaluate(y, np.clip(challenger.predict(live_test[cols]), 0, None)),
    }
    promote = scores["challenger"]["mae"] < scores["champion"]["mae"]

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "history_rows": len(history),
        "recent_rows_train": len(live_train),
        "recent_rows_test": len(live_test),
        "recent_weight": args.recent_weight,
        "scores_on_recent_holdout": scores,
        "promoted": promote,
    }
    if promote:
        backup = args.model.with_name("model_previous.joblib")
        shutil.copy(args.model, backup)
        joblib.dump(challenger, args.model)
        metrics_path = args.model.with_name("metrics.json")
        metrics = json.loads(metrics_path.read_text())
        metrics["test_metrics"][metrics["best_model"]] = scores["challenger"]
        metrics["last_retrained"] = report["timestamp"]
        metrics_path.write_text(json.dumps(metrics, indent=2))

    args.model.with_name("retrain_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nChallenger PROMOTED (old model kept as model_previous.joblib)" if promote else "\nChampion kept")


if __name__ == "__main__":
    main()
