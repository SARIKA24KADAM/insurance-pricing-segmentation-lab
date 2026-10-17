"""Generate a deterministic, clearly synthetic personal auto policy dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_policies(n_policies: int = 30_000, seed: int = 42) -> pd.DataFrame:
    """Create policy-year observations with claim frequency and aggregate loss."""
    if n_policies < 1_000:
        raise ValueError("Use at least 1,000 policies for stable train/test segments")
    rng = np.random.default_rng(seed)
    years = rng.choice([2021, 2022, 2023, 2024], size=n_policies, p=[.24, .25, .25, .26])
    age = rng.integers(18, 81, size=n_policies)
    vehicle_age = rng.integers(0, 19, size=n_policies)
    mileage = rng.integers(3_000, 25_001, size=n_policies)
    prior_claims = rng.choice([0, 1, 2, 3, 4], size=n_policies, p=[.70, .20, .07, .025, .005])
    vehicle_type = rng.choice(["compact", "sedan", "suv", "pickup"], size=n_policies,
                              p=[.22, .38, .29, .11])
    territory = rng.choice(["rural", "suburban", "urban", "metro"], size=n_policies,
                           p=[.18, .35, .30, .17])
    exposure = rng.uniform(.35, 1.0, size=n_policies).round(3)

    age_factor = np.where(age < 25, 1.65, np.where(age > 70, 1.22, 1.0))
    territory_frequency = pd.Series(territory).map(
        {"rural": .78, "suburban": .92, "urban": 1.20, "metro": 1.42}
    ).to_numpy()
    vehicle_frequency = pd.Series(vehicle_type).map(
        {"compact": .88, "sedan": 1.0, "suv": 1.08, "pickup": 1.16}
    ).to_numpy()
    mileage_factor = np.exp((mileage - 14_000) / 14_000 * .28)
    claims_factor = 1 + .35 * prior_claims
    # Mild year drift is intentional so the latest-year holdout is not identical to training.
    year_factor = 1 + (years - 2021) * .025
    annual_frequency = .075 * age_factor * territory_frequency * vehicle_frequency * mileage_factor * claims_factor * year_factor
    claim_count = rng.poisson(annual_frequency * exposure)

    territory_severity = pd.Series(territory).map(
        {"rural": .92, "suburban": 1.0, "urban": 1.12, "metro": 1.22}
    ).to_numpy()
    vehicle_severity = pd.Series(vehicle_type).map(
        {"compact": .90, "sedan": 1.0, "suv": 1.15, "pickup": 1.18}
    ).to_numpy()
    expected_severity = 3_400 * territory_severity * vehicle_severity * (1 + .012 * np.maximum(vehicle_age - 8, 0))
    total_loss = np.zeros(n_policies, dtype=float)
    has_claim = claim_count > 0
    # A sum of k Gamma(2, mean/2) severities is Gamma(2k, mean/2).
    total_loss[has_claim] = rng.gamma(
        shape=2 * claim_count[has_claim], scale=expected_severity[has_claim] / 2
    )
    # Existing premium is a noisy benchmark, not a true market rate.
    written_premium = annual_frequency * expected_severity * 1.28 * rng.lognormal(0, .12, n_policies)

    return pd.DataFrame({
        "policy_year": years,
        "driver_age": age,
        "vehicle_age": vehicle_age,
        "annual_mileage": mileage,
        "prior_claims": prior_claims,
        "vehicle_type": vehicle_type,
        "territory": territory,
        "exposure": exposure,
        "claim_count": claim_count,
        "aggregate_loss": total_loss.round(2),
        "written_premium": written_premium.round(2),
    })

