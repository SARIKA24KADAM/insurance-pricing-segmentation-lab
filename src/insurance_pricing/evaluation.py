"""Out-of-time model diagnostics and portfolio-level segment summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_gamma_deviance, mean_poisson_deviance, mean_squared_error


def calibration_table(frame: pd.DataFrame, predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    scored = frame[["exposure", "claim_count", "aggregate_loss"]].join(predictions)
    scored["risk_decile"] = pd.qcut(scored["predicted_pure_premium"], q=bins, duplicates="drop")
    result = scored.groupby("risk_decile", observed=True).agg(
        policies=("exposure", "size"),
        exposure=("exposure", "sum"),
        claims=("claim_count", "sum"),
        loss=("aggregate_loss", "sum"),
        expected_frequency=("predicted_frequency", "mean"),
        expected_severity=("predicted_severity", "mean"),
        expected_pure_premium=("predicted_pure_premium", "mean"),
    ).reset_index(drop=True)
    result["observed_frequency"] = result["claims"] / result["exposure"]
    result["observed_pure_premium"] = result["loss"] / result["exposure"]
    result["observed_to_expected"] = result["observed_pure_premium"] / result["expected_pure_premium"].clip(lower=1e-8)
    result.insert(0, "risk_decile", np.arange(1, len(result) + 1))
    return result


def score_metrics(frame: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    actual_frequency = frame["claim_count"] / frame["exposure"]
    claim_rows = frame["claim_count"] > 0
    actual_severity = frame.loc[claim_rows, "aggregate_loss"] / frame.loc[claim_rows, "claim_count"]
    predicted_severity = predictions.loc[claim_rows, "predicted_severity"]
    actual_loss_rate = frame["aggregate_loss"] / frame["exposure"]
    return {
        "holdout_policy_year": int(frame["policy_year"].min()),
        "holdout_policies": int(len(frame)),
        "frequency_poisson_deviance": float(mean_poisson_deviance(
            actual_frequency, predictions["predicted_frequency"], sample_weight=frame["exposure"])),
        "severity_gamma_deviance": float(mean_gamma_deviance(
            actual_severity, predicted_severity, sample_weight=frame.loc[claim_rows, "claim_count"])),
        "pure_premium_mae": float(mean_absolute_error(actual_loss_rate, predictions["predicted_pure_premium"])),
        "pure_premium_rmse": float(np.sqrt(mean_squared_error(actual_loss_rate, predictions["predicted_pure_premium"]))),
        "observed_loss_per_exposure": float(frame["aggregate_loss"].sum() / frame["exposure"].sum()),
        "predicted_pure_premium_mean": float(predictions["predicted_pure_premium"].mean()),
    }


def portfolio_segments(frame: pd.DataFrame, predictions: pd.DataFrame,
                       expense_and_margin_load: float = 1.30,
                       max_rate_change: float = .25) -> pd.DataFrame:
    """Estimate capped indicated premium changes; all economics are illustrative."""
    scored = frame.copy().join(predictions)
    scored["driver_age_band"] = pd.cut(scored["driver_age"], [17, 24, 34, 49, 64, 80],
                                        labels=["18-24", "25-34", "35-49", "50-64", "65+"])
    outputs = []
    for field in ["territory", "vehicle_type", "driver_age_band"]:
        grouped = scored.groupby(field, observed=True).agg(
            policies=("exposure", "size"),
            exposure=("exposure", "sum"),
            observed_claims=("claim_count", "sum"),
            observed_loss=("aggregate_loss", "sum"),
            current_premium=("written_premium", "sum"),
            indicated_pure_premium=("predicted_pure_premium", "mean"),
        ).reset_index().rename(columns={field: "segment"})
        grouped.insert(0, "segment_dimension", field)
        grouped["observed_loss_cost"] = grouped["observed_loss"] / grouped["exposure"]
        grouped["indicated_premium"] = grouped["indicated_pure_premium"] * expense_and_margin_load
        grouped["raw_rate_change"] = grouped["indicated_premium"] / (grouped["current_premium"] / grouped["policies"]) - 1
        grouped["capped_rate_change"] = grouped["raw_rate_change"].clip(-max_rate_change, max_rate_change)
        grouped["illustrative_premium_delta"] = grouped["current_premium"] * grouped["capped_rate_change"]
        outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True)

