from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from p2_restore.joint_hydrographic_multitask_layer4_checkpoint_v1 import (
    CONFIG_RELATIVE,
    build_execution_plan,
    exact_best_epoch,
    exclusive_bytes,
    median_epoch,
    split_inner_times,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / CONFIG_RELATIVE).read_text(encoding="utf-8"))


def test_registered_config_and_plan_are_exact() -> None:
    config = _config()
    validate_config(config)
    plan = build_execution_plan(config)
    assert plan["inner_fits"] == 45
    assert plan["full_prefix_refits"] == 45
    assert plan["total_fits"] == 90
    assert plan["outer_prediction_arrays"] == 45
    assert plan["max_epochs"] == 120
    assert plan["full_steps_per_epoch_all_cells"] == 219
    assert plan["worst_case_optimizer_steps"] > 219 * 120
    assert plan["candidate_predictions"] == 0
    assert plan["test_predictions"] == 0
    assert plan["uploads"] == 0


def test_inner_split_is_chronological_prefix_only_with_strict_seven_day_embargo() -> None:
    prefix = pd.date_range("2024-01-01", periods=60 * 144, freq="10min", tz="UTC")
    train, calibration, audit = split_inner_times(
        prefix,
        train_fraction=0.75,
        embargo_days=7,
        pd_module=pd,
    )
    assert calibration.equals(prefix[int(len(prefix) * 0.75) :])
    assert len(train.intersection(calibration)) == 0
    assert train.max() < calibration.min() - pd.Timedelta(days=7)
    assert set(train).issubset(set(prefix))
    assert set(calibration).issubset(set(prefix))
    assert audit["outer_truth_used"] is False
    assert audit["embargo_days"] == 7


def test_exact_checkpoint_rule_uses_minimum_and_earliest_exact_tie() -> None:
    epoch, score = exact_best_epoch(
        [
            {"epoch": 1, "validation_rmse_c": 0.9},
            {"epoch": 2, "validation_rmse_c": 0.7},
            {"epoch": 3, "validation_rmse_c": 0.7},
            {"epoch": 4, "validation_rmse_c": 0.8},
        ]
    )
    assert epoch == 2
    assert score == 0.7
    assert median_epoch([19, 7, 11]) == 11


def test_append_only_writer_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "immutable.bin"
    exclusive_bytes(target, b"first")
    with pytest.raises(FileExistsError):
        exclusive_bytes(target, b"second")
    assert target.read_bytes() == b"first"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_epochs", 121),
        ("patience_epochs", 29),
        ("inner_embargo_days", 6),
        ("inner_train_fraction", 0.8),
    ],
)
def test_checkpoint_protocol_mutation_fails_closed(field: str, value: object) -> None:
    config = copy.deepcopy(_config())
    config["checkpoint_protocol"][field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_source_allowlist_excludes_official_evaluation_files() -> None:
    config = _config()
    assert set(config["source_boundary"]["allowed_files"]) == {
        "README.md",
        "observations.csv",
    }
    assert config["candidate_or_test_prediction_allowed"] is False
    assert config["upload_allowed"] is False
    engine = (ROOT / "src/p2_restore/joint_hydrographic_multitask_layer4_checkpoint_v1.py").read_text(
        encoding="utf-8"
    )
    assert 'resolved_data / "observations.csv"' in engine
    assert 'resolved_data / "test_index.csv"' not in engine
    assert 'resolved_data / "sample_submission.csv"' not in engine
    assert 'resolved_data / "baseline_interp.csv"' not in engine
