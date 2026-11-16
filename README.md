# YouTube Ads Click Prediction

An end-to-end classic machine learning project that predicts how many clicks a YouTube ad will get, using **linear regression**. It covers the full lifecycle: data, training, evaluation, deployment and monitoring, and it runs entirely on a laptop with free tools.

## Project roadmap

| Step | What it does | Status |
|------|--------------|--------|
| 1. Data and exploration | Generate a realistic dataset, explore it, clean it, engineer features | ✅ Done |
| 2. Train and evaluate | Linear regression pipeline, RMSE / MAE / R², residual plots, baseline comparison | ✅ Done |
| 3. Deploy | FastAPI prediction service in a Docker container | ✅ Done |
| 4. Monitor | Prediction logging, drift reports with Evidently, CI and retraining with GitHub Actions | ⏳ Next |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/generate_data.py      # writes data/youtube_ads.csv
python src/train.py              # trains, evaluates, saves models/model.joblib
python src/compare_models.py     # compares six models on the same test set
python -m pytest                 # data, model and API tests
jupyter notebook notebooks/      # 01 exploration, 02 training/evaluation, 03 model comparison
```

## Project structure

```
├── app/
│   └── main.py                    # FastAPI prediction service
├── data/                          # generated CSVs (not committed, recreate with the script)
├── models/
│   ├── model.joblib               # trained model (cleaning + features + regression in one object)
│   ├── metrics.json               # Step 2 test scores
│   └── comparison.json            # scores of all six models compared
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Step 1: EDA, cleaning, feature engineering
│   ├── 02_train_evaluate.ipynb    # Step 2: training, metrics, residuals, coefficients
│   └── 03_model_comparison.ipynb  # Step 2b: linear vs Ridge, Lasso, Poisson, logistic, boosting
├── src/
│   ├── generate_data.py           # creates the synthetic dataset
│   ├── features.py                # cleaning + feature pipeline reused by training and the API
│   ├── train.py                   # trains, compares and saves the model
│   └── compare_models.py          # compares linear regression with five alternatives
├── tests/
│   ├── test_pipeline.py           # sanity checks for data, features and model
│   └── test_api.py                # API tests
├── Dockerfile                     # container for the API
├── requirements.txt               # everything for development
└── requirements-api.txt           # minimal, pinned dependencies for serving
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

---

## Step 2: Train and evaluate

### Approach
`src/train.py` splits the data 80/20 into train and test sets and compares three models on the same test set:

| Model | Idea |
|-------|------|
| `baseline` | Always predicts the average clicks. Any real model has to beat this |
| `linear_raw` | Linear regression predicting clicks directly |
| `linear_log` | Linear regression predicting `log(1 + clicks)`, converted back with `exp(x) − 1` |

Every model is a single scikit-learn pipeline (feature preprocessing plus regression), so the saved file contains everything needed to make a prediction from raw ad details.

### Results (test set, 4,000 ads)

| Model | RMSE | MAE | R² | Typical error |
|-------|-----:|----:|---:|--------------:|
| baseline | 626 | 392 | 0.00 | 74% |
| linear_raw | 356 | 196 | 0.68 | 48% |
| **linear_log** | **139** | **72** | **0.95** | **12%** |

5-fold cross-validation R²: **0.956 ± 0.004**, so the score is stable across different splits.

- **RMSE:** typical miss in clicks, with large misses weighted more.
- **MAE:** average miss in clicks.
- **R²:** share of the variation in clicks the model explains (1 = perfect).
- **Typical error:** median absolute percentage error.

### Why the log target wins
Clicks = impressions × CTR, and every feature *multiplies* the CTR. Linear regression *adds* effects, so on the raw scale it has the wrong shape. Logs turn multiplication into addition (log clicks = log impressions + log CTR effects), so a linear model fits the true structure. Residuals on the log scale are centred on zero, evenly spread and close to normal, so the linear regression assumptions hold.

### What the model learned
With a log target, each coefficient converts to a percentage effect on clicks: `(exp(coef) − 1) × 100`.

- **Impressions** dominate: +1 standard deviation more than doubles clicks.
- **In-feed** (+100%), **non-skippable** (+78%) and **skippable** (+43%) formats beat 6-second bumper ads.
- **TV** viewers (−55%) and the **55+** audience (−44%) click the least. **Mobile** (+20%) and **evening** slots (+24%) click more.
- **Budget** and **subscribers** end up near zero. Budget matters only *through* impressions (multicollinearity), and subscribers never mattered.

### Limitations
- The data is synthetic, so real-world accuracy would be lower.
- Converting log predictions back slightly underestimates the mean (retransformation bias).
- Interactions, such as mobile *and* evening, aren't modelled unless added as features.

---

## Step 2b: Is linear regression the right model?

`src/compare_models.py` and `notebooks/03_model_comparison.ipynb` test the chosen model against five alternatives on the same test set.

| Model | MAE | RMSE | R² | Notes |
|-------|----:|-----:|---:|-------|
| Lasso (log clicks) | 71.6 | 139.0 | 0.951 | Sets budget, subscribers and "morning" to exactly zero |
| Ridge (log clicks) | 71.6 | 139.1 | 0.951 | No gain: enough data that plain regression doesn't overfit |
| **Linear (log clicks)** | **71.6** | **139.1** | **0.951** | **Chosen: joint-best, fastest, simplest to explain** |
| Poisson GLM | 72.2 | 139.1 | 0.951 | The textbook model for counts. Equally good |
| Gradient boosting | 76.1 | 147.0 | 0.945 | Flexible, but the true pattern is multiplicative, which logs capture exactly |
| Logistic (click rate) | 79.0 | 152.5 | 0.941 | Predicts P(click) per impression × impressions |

**Why logistic regression is framed on click rate:** logistic regression predicts yes/no outcomes, and clicks is a count. So each impression is treated as a yes/no trial, and predicted clicks = P(click) × impressions. It trails here because it lets very large ads dominate the fit, and because it assumes an S-shaped curve while these click rates multiply. Its odds ratios are still useful for explaining what makes *each viewer* more likely to click. Impressions and budget come out near 1, because they change how many people see the ad, not how likely each one is to click.

**Verdict:** keep linear regression on log clicks. Re-test gradient boosting once real data is available, since real behaviour usually has interactions that trees can learn.

---

## Step 3: Deploy as a web service

The trained model is served by a **FastAPI** app (`app/main.py`) and packaged in a **Docker** container, so it runs the same on any laptop, server or cloud.

### Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| GET | `/health` | Liveness check, used by Docker and load balancers |
| GET | `/model-info` | Model type, features and test scores |
| POST | `/predict` | Predicted clicks and click-through rate for one ad |
| POST | `/predict/batch` | Same, for up to 1,000 ads in one request |
| GET | `/docs` | Interactive documentation, where you can try every endpoint in the browser |

### Run it without Docker

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://127.0.0.1:8000/docs
```

### Run it with Docker

```bash
docker build -t youtube-ads-api .
docker run -p 8000:8000 youtube-ads-api
```

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"category": "Gaming", "device": "Mobile", "ad_format": "In-feed", "age_group": "18-24",
       "hour": 20, "day_of_week": 5, "ad_length_sec": 15, "daily_budget_usd": 150,
       "impressions": 40000, "channel_subscribers": 250000, "video_watch_pct": 60,
       "thumbnail_quality": 4}'
```

```json
{"predicted_clicks": 1369.8, "predicted_ctr_pct": 3.425}
```

The same budget and impressions for a Finance bumper ad on TV, shown to a 55+ audience at 3 AM, predicts about **39 clicks (0.1% CTR)**. That's 35× fewer.

### Design decisions
- **Input validation:** Pydantic rejects bad input (an unknown device, hour 25, zero impressions) with a clear 422 error, before it reaches the model.
- **Same features in training and serving:** the API reuses `src/features.py`, and the saved model contains the full preprocessing pipeline, so there's no training/serving skew.
- **Optional fields:** `channel_subscribers` and `video_watch_pct` can be left out. The pipeline fills them with the training median.
- **Sanity limits:** predictions are clipped to between 0 and the number of impressions.
- **Pinned versions:** `requirements-api.txt` pins scikit-learn to the version the model was trained with, because saved models can break across versions.
- **Small, safe image:** the container holds only the API code, feature code and model, and runs as a non-root user with a health check.

### Moving to AWS later
The same container can be deployed without code changes. Push the image to **Amazon ECR**, then run it on **AWS App Runner** (simplest) or **ECS Fargate**. For SageMaker, the model file can be served with the SageMaker scikit-learn container.
