"""Run a reproducible synthetic insurance pricing analysis end to end."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_pricing.data import make_synthetic_policies
from insurance_pricing.evaluation import calibration_table, portfolio_segments, score_metrics
from insurance_pricing.model import fit_frequency, fit_severity, predict_pure_premium


def run(output: Path, n_policies: int = 30_000, seed: int = 42) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    charts = output / "graphs"
    charts.mkdir(parents=True, exist_ok=True)
    policies = make_synthetic_policies(n_policies=n_policies, seed=seed)
    train = policies[policies["policy_year"] < 2024].copy()
    holdout = policies[policies["policy_year"] == 2024].copy()
    if train.empty or holdout.empty:
        raise ValueError("The generated sample must include both pre-2024 training and 2024 holdout policies")

    frequency_model = fit_frequency(train)
    severity_model = fit_severity(train)
    predictions = predict_pure_premium(frequency_model, severity_model, holdout)
    metrics = score_metrics(holdout, predictions)
    calibration = calibration_table(holdout, predictions)
    segments = portfolio_segments(holdout, predictions)
    policies.to_csv(output / "policy_sample.csv", index=False)
    calibration.to_csv(output / "calibration_by_risk_decile.csv", index=False)
    segments.to_csv(output / "portfolio_segment_impact.csv", index=False)
    (output / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(calibration["risk_decile"], calibration["observed_pure_premium"], marker="o", label="Observed loss cost")
    ax.plot(calibration["risk_decile"], calibration["expected_pure_premium"], marker="o", label="Expected pure premium")
    ax.set(title="Out-of-time pure premium calibration", xlabel="Predicted risk decile (2024 holdout)", ylabel="Loss per exposure")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(charts / "pure_premium_calibration.png", dpi=180)
    plt.close(fig)

    territory = segments[segments["segment_dimension"] == "territory"].sort_values("capped_rate_change")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#b64a4a" if value < 0 else "#3478a8" for value in territory["capped_rate_change"]]
    ax.barh(territory["segment"], territory["capped_rate_change"] * 100, color=colors)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(title="Illustrative capped indicated rate change by territory", xlabel="Change from synthetic current premium (%)", ylabel="Territory")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(charts / "territory_rate_impact.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(calibration["risk_decile"], calibration["observed_to_expected"], color="#547c55")
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1, label="Observed = expected")
    ax.set(title="Observed-to-expected loss cost by risk decile", xlabel="Predicted risk decile", ylabel="Observed / expected")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(charts / "observed_expected_by_decile.png", dpi=180)
    plt.close(fig)

    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--policies", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.policies, args.seed), indent=2))


if __name__ == "__main__":
    main()

