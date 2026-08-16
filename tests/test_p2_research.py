from __future__ import annotations

import numpy as np
import pandas as pd

from p2_restore.features import FeatureTable
from p2_restore.research import (
    P2ResearchBlendModel,
    append_public_dynamics,
    append_public_m2_harmonics,
    paired_day_bootstrap,
    paired_rmse_bootstrap,
    pchip_profile_prediction,
    physics_blend,
    select_lean_m2_dynamics,
)


def _observations() -> pd.DataFrame:
    rows = []
    times = pd.date_range("2025-01-01", periods=200, freq="10min", tz="Asia/Seoul")
    for number, time in enumerate(times):
        for layer, depth in ((1, 4.2), (5, 19.6), (6, 30.7), (7, 39.5), (8, 49.4)):
            rows.append(
                {
                    "time": time.isoformat(),
                    "layer": layer,
                    "temp": 20 - 0.1 * depth + np.sin(number / 20),
                    "psal": 31 + 0.01 * depth,
                }
            )
    return pd.DataFrame(rows)


def test_public_dynamics_are_gap_safe_and_target_free() -> None:
    observations = _observations()
    frame = pd.DataFrame(
        {
            "station": ["S-ORS"],
            "layer": [2],
            "time": [observations.iloc[100]["time"]],
            "baseline": [19.0],
            "temp_1_minus_5": [1.0],
        }
    )
    bundle = append_public_dynamics(FeatureTable(frame, ("baseline",)), observations)
    assert len(bundle.frame) == 1
    assert "temp_1_center_change_6h" in bundle.feature_columns
    assert all(
        "temp_2" not in column and "psal_2" not in column for column in bundle.feature_columns
    )
    lean = select_lean_m2_dynamics(FeatureTable(frame, ("baseline",)), bundle)
    assert len(lean.feature_columns) == 21
    assert not any(column.startswith("psal_") for column in lean.feature_columns)


def test_pchip_preserves_linear_profile() -> None:
    frame = pd.DataFrame(
        {
            "baseline": [19.0],
            "target_depth": [10.0],
            **{
                f"temp_{layer}": [20 - 0.1 * depth]
                for layer, depth in zip((1, 5, 6, 7, 8), (4, 20, 30, 40, 50), strict=True)
            },
            **{
                f"nominal_{layer}": [depth]
                for layer, depth in zip((1, 5, 6, 7, 8), (4, 20, 30, 40, 50), strict=True)
            },
        }
    )
    assert np.isclose(pchip_profile_prediction(frame)[0], 19.0)


def test_physics_blend_shrinks_mixed_profile() -> None:
    frame = pd.DataFrame({"baseline": [10.0, 10.0], "temp_1_minus_5": [0.2, 2.0]})
    blended = physics_blend(frame, np.array([12.0, 12.0]))
    assert np.allclose(blended, [10.0, 12.0])


def test_paired_bootstrap_is_deterministic() -> None:
    oof = pd.DataFrame(
        {
            "day": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
            "truth": [0.0, 1.0, 0.0, 1.0],
            "v0": [0.2, 1.2, 0.2, 1.2],
            "lean_m2": [0.1, 1.1, 0.1, 1.1],
            "blend50": [0.15, 1.15, 0.15, 1.15],
        }
    )
    first = paired_day_bootstrap(oof, replicates=50, seed=7)
    second = paired_day_bootstrap(oof, replicates=50, seed=7)
    assert first == second
    assert first["delta_rmse"] < 0


def test_research_blend_weight_is_frozen() -> None:
    class ConstantModel:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, table: FeatureTable) -> np.ndarray:
            return np.full(len(table.frame), self.value)

    table = FeatureTable(pd.DataFrame({"x": [1, 2]}), ("x",))
    model = P2ResearchBlendModel(ConstantModel(1.0), ConstantModel(3.0))
    assert np.allclose(model.predict(table, table), 2.0)


def test_public_m2_harmonics_are_target_free_bounded_and_deterministic() -> None:
    rows = []
    times = pd.date_range("2025-01-01", periods=1200, freq="10min", tz="Asia/Seoul")
    epoch = np.arange(len(times)) * 10 / 60
    for number, time in enumerate(times):
        for layer in (1, 5, 6, 7, 8):
            rows.append(
                {
                    "time": time.isoformat(),
                    "layer": layer,
                    "temp": 20
                    - 0.1 * layer
                    + (1 + layer / 20) * np.sin(2 * np.pi * epoch[number] / 12.42),
                }
            )
    observations = pd.DataFrame(rows)
    frame = pd.DataFrame(
        {
            "station": ["S-ORS"] * 2,
            "layer": [2, 3],
            "time": [times[600].isoformat(), times[601].isoformat()],
            "baseline": [19.0, 18.5],
        }
    )
    source = FeatureTable(frame, ("baseline",))
    first = append_public_m2_harmonics(source, observations)
    second = append_public_m2_harmonics(source, observations)
    additions = [column for column in first.feature_columns if column != "baseline"]
    assert len(additions) == 20
    assert first.frame[additions].equals(second.frame[additions])
    assert first.frame[additions].notna().all().all()
    phase = first.frame[[column for column in additions if "phase_" in column]]
    assert phase.abs().max().max() <= 1.0 + 1e-12
    assert not any("temp_2_" in column or "temp_3_" in column for column in additions)


def test_generic_paired_bootstrap_compares_fixed_reference() -> None:
    oof = pd.DataFrame(
        {
            "day": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
            "truth": [0.0, 1.0, 0.0, 1.0],
            "reference": [0.2, 1.2, 0.2, 1.2],
            "candidate": [0.1, 1.1, 0.1, 1.1],
        }
    )
    result = paired_rmse_bootstrap(
        oof, reference="reference", candidate="candidate", replicates=50, seed=7
    )
    assert result["delta_rmse"] < 0
    assert result["probability_improved"] == 1.0
