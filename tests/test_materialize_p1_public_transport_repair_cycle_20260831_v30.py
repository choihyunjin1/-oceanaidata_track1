from __future__ import annotations

import pandas as pd
import pytest

from scripts import materialize_p1_public_transport_repair_cycle_20260831_v30 as materializer


def synthetic_keys(rows: int = 169_011) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["A"] * rows,
            "year": [2026] * rows,
            "layer": [1] * rows,
            "time": [f"row-{index}" for index in range(rows)],
        }
    )


def test_internal_pass_contract_is_frozen() -> None:
    config, result = materializer.load_contract()
    assert result["candidate"]["strict_internal_pass"] is True
    assert result["pass_count"] == 1
    assert config["candidate"] == "P1_1_LABEL_FREE_RELIABILITY_GUARDED_LABEL_SHIFT_EM"


def test_valid_output_contract_passes() -> None:
    keys = synthetic_keys()
    submission = keys.copy()
    submission["label"] = 0
    checks = materializer.validate_output_frame(submission, keys)
    assert all(checks.values())


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


def test_preflight_has_no_official_value_read() -> None:
    if materializer.ARTIFACT.exists():
        result = pd.read_json(materializer.ARTIFACT / "result.json", typ="series")
        assert result["status"] == "MATERIALIZED_NOT_UPLOADED"
        assert result["operations"]["hidden_truth_reads"] == 0
        assert result["operations"]["uploads"] == 0
    else:
        result = materializer.preflight()
        assert result["status"] == "PASS"
        assert result["official_values_read"] == 0
        assert result["hidden_truth_reads"] == 0
        assert result["uploads"] == 0
