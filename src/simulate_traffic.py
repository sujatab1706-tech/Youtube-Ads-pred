"""Simulate a week of real traffic hitting the API, so monitoring has something to watch.

It sends ads to the API (which logs each prediction), then sends back the "real" clicks
through /feedback, just as an ad platform would report results a day later.

Scenarios:
  normal - new ads that look like the training data
  drift  - the world has changed:
           * data drift: far more ads now run on TV and to older audiences
           * concept drift: ad fatigue cuts the real click-through rate by 35%

Usage:
    python src/simulate_traffic.py --scenario drift --rows 3000 --log-dir logs
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT))

from generate_data import generate  # noqa: E402

API_FIELDS = [
    "category", "device", "ad_format", "age_group", "hour", "day_of_week", "ad_length_sec",
    "daily_budget_usd", "impressions", "channel_subscribers", "video_watch_pct", "thumbnail_quality",
]


def make_ads(scenario: str, rows: int, seed: int):
    pool = generate(rows * 8, seed=seed).drop_duplicates("ad_id")
    rng = np.random.default_rng(seed)
    if scenario == "normal":
        return pool.sample(rows, random_state=seed)

    # Data drift: oversample TV viewers and older audiences.
    weight = np.ones(len(pool))
    weight[pool["device"].eq("TV").to_numpy()] *= 6
    weight[pool["age_group"].isin(["45-54", "55+"]).to_numpy()] *= 3
    ads = pool.sample(rows, weights=weight, random_state=seed).copy()
    # Concept drift: the same ad now gets fewer clicks than the model expects.
    ads["clicks"] = rng.binomial(ads["clicks"].to_numpy(), 0.65)
    return ads


def to_payload(row) -> dict:
    ad = {k: row[k] for k in API_FIELDS}
    for k in ("channel_subscribers", "video_watch_pct"):
        if ad[k] != ad[k]:  # NaN -> leave out, the API fills it
            ad[k] = None
    for k in ("hour", "day_of_week", "ad_length_sec", "impressions", "thumbnail_quality"):
        ad[k] = int(ad[k])
    if ad["channel_subscribers"] is not None:
        ad["channel_subscribers"] = int(ad["channel_subscribers"])
    return ad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=["normal", "drift"], default="normal")
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()

    os.environ["LOG_DIR"] = str(args.log_dir.resolve())
    from fastapi.testclient import TestClient  # imported late so LOG_DIR is picked up

    from app.main import app

    ads = make_ads(args.scenario, args.rows, args.seed)
    payloads = [to_payload(r) for _, r in ads.iterrows()]
    actual = ads["clicks"].to_numpy()

    with TestClient(app) as client:
        ids = []
        for i in range(0, len(payloads), 1000):
            r = client.post("/predict/batch", json={"ads": payloads[i:i + 1000]})
            r.raise_for_status()
            ids += [p["prediction_id"] for p in r.json()["predictions"]]
        for pid, clicks in zip(ids, actual):
            client.post("/feedback", json={"prediction_id": pid, "actual_clicks": int(clicks)}).raise_for_status()

    print(f"Sent {len(ids):,} '{args.scenario}' predictions and their real clicks; logs in {args.log_dir}/")


if __name__ == "__main__":
    main()
