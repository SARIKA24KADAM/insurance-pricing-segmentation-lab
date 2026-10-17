import pandas as pd

from insurance_pricing.data import make_synthetic_policies
from insurance_pricing.evaluation import calibration_table


def test_synthetic_generator_is_reproducible_and_nonnegative():
    first = make_synthetic_policies(1_000, seed=7)
    second = make_synthetic_policies(1_000, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert (first["exposure"] > 0).all()
    assert (first["claim_count"] >= 0).all()
    assert (first["aggregate_loss"] >= 0).all()


def test_calibration_has_ordered_deciles_and_positive_expectations():
    frame = make_synthetic_policies(2_000, seed=9)
    predictions = pd.DataFrame({
        "predicted_frequency": .1,
        "predicted_severity": 3_000.0,
        "predicted_pure_premium": 300.0,
    }, index=frame.index)
    table = calibration_table(frame, predictions, bins=5)
    assert table["risk_decile"].tolist() == list(range(1, len(table) + 1))
    assert (table["expected_pure_premium"] > 0).all()

