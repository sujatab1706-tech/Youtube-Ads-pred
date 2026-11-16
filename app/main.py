"""FastAPI service that predicts clicks for a YouTube ad.

Run locally:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs to try it in the browser.
"""

import json
import os
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from features import CATEGORICAL, NUMERIC, add_features  # noqa: E402

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
METRICS_PATH = MODEL_PATH.with_name("metrics.json")
MAX_BATCH = 1000
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT / "logs"))
PREDICTION_LOG = LOG_DIR / "predictions.jsonl"
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"
_log_lock = threading.Lock()


class Ad(BaseModel):
    """Details of one ad. Values outside the allowed options are rejected with a clear error."""

    category: Literal["Gaming", "Tech", "Beauty", "Education", "Finance", "Food", "Travel", "Fitness"]
    device: Literal["Mobile", "Desktop", "TV", "Tablet"]
    ad_format: Literal["Skippable", "Non-skippable", "Bumper", "In-feed"]
    age_group: Literal["18-24", "25-34", "35-44", "45-54", "55+"]
    hour: int = Field(ge=0, le=23, description="Hour of day the ad runs (0-23)")
    day_of_week: int = Field(ge=0, le=6, description="0 = Monday ... 6 = Sunday")
    ad_length_sec: int = Field(ge=1, le=600)
    daily_budget_usd: float = Field(gt=0)
    impressions: int = Field(gt=0, description="Expected number of times the ad is shown")
    channel_subscribers: Optional[int] = Field(default=None, ge=0, description="Optional; filled with the median if missing")
    video_watch_pct: Optional[float] = Field(default=None, ge=0, le=100, description="Optional; filled with the median if missing")
    thumbnail_quality: int = Field(ge=1, le=5)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "category": "Gaming", "device": "Mobile", "ad_format": "In-feed", "age_group": "18-24",
                "hour": 20, "day_of_week": 5, "ad_length_sec": 15, "daily_budget_usd": 150.0,
                "impressions": 40000, "channel_subscribers": 250000, "video_watch_pct": 60.0,
                "thumbnail_quality": 4,
            }]
        }
    }


class Prediction(BaseModel):
    prediction_id: str = Field(description="Send this back to /feedback with the real clicks once you know them")
    predicted_clicks: float
    predicted_ctr_pct: float = Field(description="Predicted clicks / impressions, as a percentage")


class Feedback(BaseModel):
    prediction_id: str
    actual_clicks: int = Field(ge=0)


class BatchRequest(BaseModel):
    ads: list[Ad] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


@lru_cache
def get_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run `python src/train.py` first.")
    return joblib.load(MODEL_PATH)


def to_features(ads: list[Ad]) -> pd.DataFrame:
    """Turn API input into exactly the feature table the model was trained on."""
    df = pd.DataFrame([ad.model_dump() for ad in ads])
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["channel_subscribers"] = df["channel_subscribers"].astype(float)
    df["video_watch_pct"] = df["video_watch_pct"].astype(float)
    return add_features(df)[NUMERIC + CATEGORICAL]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_jsonl(path: Path, records: list[dict]) -> None:
    """Append records to a JSON Lines file. Logging must never break a prediction."""
    try:
        with _log_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
    except OSError as exc:  # pragma: no cover
        print(f"WARNING: could not write {path}: {exc}")


def predict(ads: list[Ad]) -> list[Prediction]:
    clicks = np.clip(get_model().predict(to_features(ads)), 0, None)
    out, logs, ts = [], [], now()
    for ad, c in zip(ads, clicks):
        c = min(float(c), float(ad.impressions))  # can't click more than it was shown
        pred = Prediction(
            prediction_id=str(uuid.uuid4()),
            predicted_clicks=round(c, 1),
            predicted_ctr_pct=round(100 * c / ad.impressions, 3),
        )
        out.append(pred)
        logs.append({"timestamp": ts, "prediction_id": pred.prediction_id, **ad.model_dump(),
                     "predicted_clicks": pred.predicted_clicks})
    append_jsonl(PREDICTION_LOG, logs)
    return out


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_model()  # fail fast at startup if the model file is missing
    yield


app = FastAPI(
    lifespan=lifespan,
    title="YouTube Ad Click Predictor",
    description="Predicts daily clicks for a YouTube ad with a linear regression model trained on log(1 + clicks).",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/model-info")
def model_info():
    info = {"model_path": str(MODEL_PATH.name), "features": NUMERIC + CATEGORICAL}
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        info.update({"model_type": m["best_model"], "test_metrics": m["test_metrics"][m["best_model"]], "cv_r2_mean": m["cv_r2_mean"]})
    return info


@app.post("/predict", response_model=Prediction)
def predict_one(ad: Ad):
    try:
        return predict([ad])[0]
    except Exception as exc:  # pragma: no cover - unexpected model errors
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    return BatchResponse(predictions=predict(req.ads))


@app.post("/feedback")
def feedback(fb: Feedback):
    """Record the real clicks for an earlier prediction, so model accuracy can be monitored."""
    append_jsonl(FEEDBACK_LOG, [{"timestamp": now(), **fb.model_dump()}])
    return {"status": "recorded"}
