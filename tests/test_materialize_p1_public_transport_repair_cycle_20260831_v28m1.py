from __future__ import annotations

import pandas as pd
import pytest

from scripts import materialize_p1_public_transport_repair_cycle_20260831_v28m1 as materializer


def synthetic_keys(rows: int = 169_011) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["A"] * rows,
            "year": [2026] * rows,
            "layer": [1] * rows,
            "time": [f"row-{index}" for index in range(rows)],
        }
    )


def test_frozen_score_priority_contract() -> None:
    deployment, _, result = materializer.load_contract()
    assert result["candidate"]["delta_f1"] > 0.008
    assert result["candidate"]["calibrated_conservative_expected_points_delta"] > 0.22
    assert sorted(
        name for name, passed in result["candidate"]["gates"].items() if not passed
    ) == sorted(deployment["frozen_candidate"]["known_failed_safety_gates"])
    assert deployment["frozen_candidate"]["tuning"] == 0
    assert deployment["data_policy"]["organizer_distributed_data_only"] is True


def test_valid_output_contract_passes() -> None:
    keys = synthetic_keys()
    submission = keys.copy()
    submission["label"] = 0
    assert all(materializer.validate_output_frame(submission, keys).values())


def test_key_order_perturbation_fails_closed() -> None:
    keys = synthetic_keys()
    submission = keys.copy()
    submission["label"] = 0
    submission.loc[[0, 1], "time"] = submission.loc[[1, 0], "time"].to_numpy()
    with pytest.raises(materializer.ContractError):
        materializer.validate_output_frame(submission, keys)


def test_nonbinary_label_fails_closed() -> None:
    keys = synthetic_keys()
    submission = keys.copy()
    submission["label"] = 0
    submission.loc[0, "label"] = 2
    with pytest.raises(materializer.ContractError):
        materializer.validate_output_frame(submission, keys)


def test_preflight_or_terminal_has_zero_forbidden_access() -> None:
    if materializer.ARTIFACT.exists():
        result = pd.read_json(materializer.ARTIFACT / "result.json", typ="series")
        assert result["operations"]["hidden_truth_reads"] == 0
        assert result["operations"]["internet_rows_read"] == 0
        assert result["operations"]["kiost_original_rows_read"] == 0
        assert result["operations"]["uploads"] == 0
    else:
        result = materializer.preflight()
        assert result["status"] == "PASS"
        assert result["official_values_read"] == 0
        assert result["hidden_truth_reads"] == 0
        assert result["external_rows_read"] == 0
        assert result["pretrained_weight_files_loaded"] == 0
        assert result["uploads"] == 0
