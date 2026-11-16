"""Generate a realistic synthetic YouTube ads dataset.

YouTube does not publish ad-level click data, so we simulate it. Each row is
one ad campaign's performance over a day. Clicks are driven by how many people
saw the ad (impressions) multiplied by a click-through rate (CTR) that depends
on the ad's properties, plus random noise. The rules below are the "hidden
truth" the model will try to learn.

Usage:
    python src/generate_data.py --rows 20000 --seed 42 --out data/youtube_ads.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["Gaming", "Tech", "Beauty", "Education", "Finance", "Food", "Travel", "Fitness"]
DEVICES = ["Mobile", "Desktop", "TV", "Tablet"]
AD_FORMATS = ["Skippable", "Non-skippable", "Bumper", "In-feed"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]

# How much each option multiplies the base click-through rate.
CATEGORY_LIFT = {"Gaming": 1.30, "Tech": 1.15, "Beauty": 1.20, "Education": 0.90,
                 "Finance": 0.75, "Food": 1.05, "Travel": 1.00, "Fitness": 1.10}
DEVICE_LIFT = {"Mobile": 1.20, "Desktop": 1.00, "TV": 0.45, "Tablet": 0.95}
FORMAT_LIFT = {"Skippable": 1.00, "Non-skippable": 1.25, "Bumper": 0.70, "In-feed": 1.40}
AGE_LIFT = {"18-24": 1.25, "25-34": 1.15, "35-44": 1.00, "45-54": 0.85, "55+": 0.70}


def generate(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    category = rng.choice(CATEGORIES, rows)
    device = rng.choice(DEVICES, rows, p=[0.62, 0.20, 0.12, 0.06])
    ad_format = rng.choice(AD_FORMATS, rows, p=[0.45, 0.20, 0.15, 0.20])
    age_group = rng.choice(AGE_GROUPS, rows, p=[0.22, 0.30, 0.22, 0.15, 0.11])
    hour = rng.integers(0, 24, rows)
    day_of_week = rng.integers(0, 7, rows)  # 0 = Monday
    is_weekend = (day_of_week >= 5).astype(int)

    # Bumper ads are 6s; others vary.
    ad_length_sec = np.where(ad_format == "Bumper", 6, rng.integers(10, 121, rows))
    daily_budget_usd = np.round(rng.lognormal(mean=5.0, sigma=0.8, size=rows), 2)
    # Impressions grow with budget (roughly $4 CPM) with noise.
    impressions = np.maximum(100, (daily_budget_usd / 4.0 * 1000 * rng.lognormal(0, 0.25, rows))).astype(int)
    channel_subscribers = np.round(rng.lognormal(mean=11, sigma=1.5, size=rows)).astype(int)
    video_watch_pct = np.clip(rng.normal(55, 18, rows), 5, 100).round(1)
    thumbnail_quality = rng.integers(1, 6, rows)  # 1-5 rating

    # Build the click-through rate from the rules above.
    base_ctr = 0.008
    evening = ((hour >= 18) & (hour <= 22)).astype(float)
    night = ((hour >= 1) & (hour <= 5)).astype(float)
    ctr = (
        base_ctr
        * np.vectorize(CATEGORY_LIFT.get)(category)
        * np.vectorize(DEVICE_LIFT.get)(device)
        * np.vectorize(FORMAT_LIFT.get)(ad_format)
        * np.vectorize(AGE_LIFT.get)(age_group)
        * (1 + 0.25 * evening - 0.30 * night + 0.10 * is_weekend)
        * (1 + 0.08 * (thumbnail_quality - 3))
        * (1 + 0.006 * (video_watch_pct - 55))
        * (1 - 0.002 * (ad_length_sec - 30))  # longer ads lose attention
        * rng.lognormal(0, 0.15, rows)  # unexplained variation
    )
    ctr = np.clip(ctr, 0.0005, 0.08)
    clicks = rng.binomial(impressions, ctr)

    df = pd.DataFrame({
        "ad_id": [f"AD{i:06d}" for i in range(rows)],
        "category": category,
        "device": device,
        "ad_format": ad_format,
        "age_group": age_group,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "ad_length_sec": ad_length_sec,
        "daily_budget_usd": daily_budget_usd,
        "impressions": impressions,
        "channel_subscribers": channel_subscribers,
        "video_watch_pct": video_watch_pct,
        "thumbnail_quality": thumbnail_quality,
        "clicks": clicks,
    })

    # Real data is messy: add a few missing values and duplicates to clean later.
    for col in ["video_watch_pct", "channel_subscribers"]:
        df.loc[rng.random(rows) < 0.01, col] = np.nan
    df = pd.concat([df, df.sample(frac=0.005, random_state=seed)], ignore_index=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/youtube_ads.csv"))
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
