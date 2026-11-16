"""Data cleaning and feature engineering shared by the notebook, training and the API."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "clicks"
CATEGORICAL = ["category", "device", "ad_format", "age_group", "time_of_day"]
NUMERIC = [
    "log_impressions", "log_budget", "log_subscribers", "ad_length_sec",
    "video_watch_pct", "thumbnail_quality", "is_weekend",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and impossible values."""
    df = df.drop_duplicates(subset="ad_id").copy()
    df = df[(df["impressions"] > 0) & (df["clicks"] >= 0) & (df["clicks"] <= df["impressions"])]
    return df.reset_index(drop=True)


def time_of_day(hour: pd.Series) -> pd.Series:
    bins = [-1, 5, 11, 17, 22, 23]
    labels = ["night", "morning", "afternoon", "evening", "late"]
    return pd.cut(hour, bins=bins, labels=labels, ordered=False).astype(str).replace("late", "night")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features. Log transforms turn skewed counts into near-linear signals."""
    df = df.copy()
    df["log_impressions"] = np.log1p(df["impressions"])
    df["log_budget"] = np.log1p(df["daily_budget_usd"])
    df["log_subscribers"] = np.log1p(df["channel_subscribers"])
    df["time_of_day"] = time_of_day(df["hour"])
    return df


def build_preprocessor() -> ColumnTransformer:
    """Impute missing values, scale numbers and one-hot encode categories."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore", drop="first")
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
