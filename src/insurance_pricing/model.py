"""Separate frequency and severity models for a transparent pure-premium baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["driver_age", "vehicle_age", "annual_mileage", "prior_claims"]
CATEGORICAL = ["vehicle_type", "territory"]
FEATURES = NUMERIC + CATEGORICAL


def _pipeline(regressor):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    return Pipeline([("features", preprocess), ("model", regressor)])


def fit_frequency(train: pd.DataFrame) -> Pipeline:
    """Fit annual claim frequency with exposure-weighted Poisson regression."""
    model = _pipeline(PoissonRegressor(alpha=.03, max_iter=500))
    model.fit(train[FEATURES], train["claim_count"] / train["exposure"],
              model__sample_weight=train["exposure"])
    return model


def fit_severity(train: pd.DataFrame) -> Pipeline:
    """Fit mean cost per claim, weighting policy averages by claim count."""
    claims = train.loc[train["claim_count"] > 0].copy()
    claims["severity"] = claims["aggregate_loss"] / claims["claim_count"]
    model = _pipeline(GammaRegressor(alpha=.05, max_iter=500))
    model.fit(claims[FEATURES], claims["severity"],
              model__sample_weight=claims["claim_count"])
    return model


def predict_pure_premium(frequency_model, severity_model, frame: pd.DataFrame) -> pd.DataFrame:
    """Return annualized frequency, severity, and indicated pure premium."""
    frequency = np.maximum(frequency_model.predict(frame[FEATURES]), 1e-8)
    severity = np.maximum(severity_model.predict(frame[FEATURES]), 1e-8)
    return pd.DataFrame({"predicted_frequency": frequency,
                         "predicted_severity": severity,
                         "predicted_pure_premium": frequency * severity}, index=frame.index)

