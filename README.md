# YouTube Ads Click Prediction

An end-to-end classic machine learning project that predicts how many clicks a YouTube ad will get, using **linear regression**. It covers the full lifecycle: data, training, evaluation, deployment and monitoring, and it runs entirely on a laptop with free tools.

## Project roadmap

| Step | What it does | Status |
|------|--------------|--------|
| 1. Data and exploration | Generate a realistic dataset, explore it, clean it, engineer features | ✅ Done |
| 2. Train and evaluate | Linear regression pipeline, RMSE / MAE / R², residual plots, baseline comparison | ⏳ Next |
| 3. Deploy | FastAPI prediction service in a Docker container | ⏳ |
| 4. Monitor | Prediction logging, drift reports with Evidently, CI and retraining with GitHub Actions | ⏳ |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/generate_data.py      # writes data/youtube_ads.csv
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Project structure

```
├── data/                          # generated CSVs (not committed, recreate with the script)
├── notebooks/
│   └── 01_data_exploration.ipynb  # Step 1: EDA, cleaning, feature engineering
├── src/
│   ├── generate_data.py           # creates the synthetic dataset
│   └── features.py                # cleaning + feature pipeline reused by training and the API
└── requirements.txt
```

---

## Step 1: Data and exploration

### Why synthetic data?
YouTube doesn't publish ad-level click data. Instead, `src/generate_data.py` simulates 20,000 ad campaigns using rules that mirror how real ad performance works:

**clicks ≈ impressions × click-through rate (CTR)**

Impressions come from the daily budget (about $4 per 1,000 views). CTR starts at about 0.8% and is raised or lowered by the ad's properties. Random noise is added so the model can't be perfect, just like real life.

### The dataset

| Column | Meaning |
|--------|---------|
| `category` | Ad topic: Gaming, Tech, Beauty, Education, Finance, Food, Travel, Fitness |
| `device` | Mobile, Desktop, TV, Tablet |
| `ad_format` | Skippable, Non-skippable, Bumper (6s), In-feed |
| `age_group` | Viewer age band |
| `hour`, `day_of_week`, `is_weekend` | When the ad ran |
| `ad_length_sec` | Ad duration |
| `daily_budget_usd` | Spend per day |
| `impressions` | Times the ad was shown |
| `channel_subscribers` | Size of the channel the ad ran on |
| `video_watch_pct` | Average % of the host video watched |
| `thumbnail_quality` | 1–5 quality score |
| **`clicks`** | **Target: what we predict** |

To practise real-world cleaning, the data also contains about 1% missing values and about 0.5% duplicate rows.

### What the exploration found
The notebook `notebooks/01_data_exploration.ipynb` walks through:

1. **Data quality:** duplicates removed (20,100 → 20,000 rows). Missing values are filled with the median inside the model pipeline, so new data gets the same treatment.
2. **Target shape:** clicks are heavily right-skewed (median 321, max over 11,000). Taking `log(1 + clicks)` makes the distribution close to normal, which suits linear regression better.
3. **Main driver:** impressions. On a log scale, log(impressions) vs log(clicks) is a straight line (correlation 0.85).
4. **CTR drivers:** in-feed and non-skippable formats, mobile devices, younger audiences, Gaming/Beauty/Tech categories and evening hours all lift CTR. TV viewers and 55+ audiences click the least.
5. **Useless feature:** `channel_subscribers` has almost no relationship with clicks. Not every plausible feature helps.
6. **Multicollinearity:** budget and impressions are strongly linked, because budget buys impressions. Predictions stay fine, but coefficients get harder to interpret.

### Feature engineering (`src/features.py`)
- Log transforms: `log_impressions`, `log_budget`, `log_subscribers`
- `time_of_day`: hours grouped into night / morning / afternoon / evening
- Numeric features: median imputation, then standard scaling
- Categorical features: one-hot encoding

These steps live in a scikit-learn `ColumnTransformer`, so exactly the same transformations run during training and when the API makes predictions. This prevents "training/serving skew".
