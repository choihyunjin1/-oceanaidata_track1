from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts"
    / "run_p1_heterogeneous_event_utility_preflight_20260830_v1.py"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "p1_heterogeneous_event_utility_preflight_20260830_v1.json"
)
SOURCE_NAMES = ["synthetic_tabular", "synthetic_neural"]


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "p1_heterogeneous_event_utility_preflight_test", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _synthetic_frame(
    event_counts: dict[str, int] | None = None,
    *,
    concentrate_q2: bool = False,
    q2_cell_indices: list[int] | None = None,
) -> pd.DataFrame:
    counts = event_counts or {
        "2025_q2": 12,
        "2025_q3": 6,
        "2025_q4": 5,
    }
    fold_starts = {
        "2025_q2": pd.Timestamp("2025-04-01T00:00:00Z"),
        "2025_q3": pd.Timestamp("2025-07-01T00:00:00Z"),
        "2025_q4": pd.Timestamp("2025-10-01T00:00:00Z"),
    }
    cells = [("STA", 1), ("STB", 2), ("STC", 3)]
    rows: list[dict[str, Any]] = []
    for fold, event_count in counts.items():
        start = fold_starts[fold]
        for index in range(12):
            rows.append(
                {
                    "station": "ANCHOR",
                    "year": 2025,
                    "layer": 9,
                    "time": start - pd.Timedelta(hours=index + 1),
                    "fold": fold,
                    "label": 1,
                    "anchor": 1,
                    "synthetic_tabular": 1,
                    "synthetic_neural": 0,
                }
            )
        rows.append(
            {
                "station": "ANCHOR_FP",
                "year": 2025,
                "layer": 8,
                "time": start - pd.Timedelta(days=1),
                "fold": fold,
                "label": 0,
                "anchor": 1,
                "synthetic_tabular": 0,
                "synthetic_neural": 0,
            }
        )
        for index in range(event_count):
            if fold == "2025_q2" and q2_cell_indices is not None:
                cell_index = q2_cell_indices[index]
            elif concentrate_q2 and fold == "2025_q2":
                cell_index = 0
            else:
                cell_index = index % 3
            station, layer = cells[cell_index]
            event_start = start + pd.Timedelta(hours=index)
            source = SOURCE_NAMES[index % len(SOURCE_NAMES)]
            proposal = {name: int(name == source) for name in SOURCE_NAMES}
            rows.append(
                {
                    "station": station,
                    "year": 2025,
                    "layer": layer,
                    "time": event_start,
                    "fold": fold,
                    "label": 1,
                    "anchor": 0,
                    **proposal,
                }
            )
            if index % 2 == 0:
                rows.append(
                    {
                        "station": station,
                        "year": 2025,
                        "layer": layer,
                        "time": event_start + pd.Timedelta(minutes=10),
                        "fold": fold,
                        "label": 0,
                        "anchor": 0,
                        **proposal,
                    }
                )
    return pd.DataFrame(rows)


def test_preregistration_is_sealed_zero_fit_and_target_preserving() -> None:
    runner = _load_runner()
    config = _load_config()
    runner._validate_config(config)

    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == runner.CONFIG_SHA256
    assert config["support_gates"] == {
        "minimum_fit_utility_positive_events": 10,
        "minimum_calibration_utility_positive_events": 4,
        "minimum_fit_utility_positive_station_layer_cells": 3,
        "maximum_fit_single_station_layer_utility_positive_share": 0.7,
        "require_complete_oof_provenance": True,
        "apply_to_every_prefix": True,
        "kill_on_any_failure": True,
    }
    assert config["execution_contract"]["model_fit_count"] == 0
    assert config["execution_contract"]["threshold_search_count"] == 0
    assert config["outputs"]["allowed_file_extensions"] == [".json"]
    assert config["outputs"]["prediction_csv_count"] == 0
    assert config["outlier_policy"]["remove_label_1_rows_or_events"] is False
    assert (
        config["outlier_policy"]["clip_or_remove_raw_temp_anomaly_signal"]
        is False
    )
    assert "diagnostic_only" in config["outlier_policy"][
        "auxiliary_psal_depth_quality_role"
    ]
    assert all(config["prohibitions"].values())

    source = RUNNER.read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "read_csv(" not in source
    assert "to_csv(" not in source
    assert "allow_pickle=False" in source


def test_synthetic_support_passes_both_prefixes_without_fitting() -> None:
    runner = _load_runner()
    result = runner.evaluate_support(
        _synthetic_frame(),
        SOURCE_NAMES,
        _load_config(),
        {"complete": True, "scope": "synthetic"},
    )

    assert result["status"] == "PASS_ZERO_FIT_SUPPORT_PREFLIGHT_RESEARCH_ONLY"
    assert result["failure_reasons"] == []
    assert result["proposal_bank"]["proposal_events"] == 23
    assert result["proposal_bank"]["anchor_positive_removed_rows"] == 0
    assert result["proposal_bank"]["candidate_materialized_or_persisted"] is False
    assert result["execution_audit"]["model_fit_count"] == 0
    assert result["execution_audit"]["prediction_csv_count"] == 0
    assert result["execution_audit"]["target_positive_rows_removed"] == 0
    for prefix in result["prefixes"].values():
        assert prefix["pass"] is True
        assert all(prefix["checks"].values())
        assert prefix["fit"]["proposal_precision_requirement_f1_over_2"] == (
            prefix["fit"]["anchor_f1"] / 2.0
        )


def test_any_prefix_failure_is_terminal_zero_fit_no_go() -> None:
    runner = _load_runner()
    result = runner.evaluate_support(
        _synthetic_frame(
            {"2025_q2": 12, "2025_q3": 6, "2025_q4": 3}
        ),
        SOURCE_NAMES,
        _load_config(),
        {"complete": True},
    )

    assert result["status"] == "NO_GO_ZERO_FIT_SUPPORT_PREFLIGHT"
    assert result["prefixes"]["q2_to_q3"]["pass"] is True
    assert (
        "q2_q3_to_q4:calibration_utility_positive_events_gte_4"
        in result["failure_reasons"]
    )
    assert result["execution_audit"]["model_fit_count"] == 0
    assert result["execution_audit"]["prediction_materialization_count"] == 0


def test_cell_concentration_and_provenance_are_hard_gates() -> None:
    runner = _load_runner()
    concentrated = runner.evaluate_support(
        _synthetic_frame(concentrate_q2=True),
        SOURCE_NAMES,
        _load_config(),
        {"complete": True},
    )
    assert concentrated["status"] == "NO_GO_ZERO_FIT_SUPPORT_PREFLIGHT"
    first = concentrated["prefixes"]["q2_to_q3"]["checks"]
    assert first["fit_utility_positive_station_layer_cells_gte_3"] is False
    assert first["fit_maximum_single_station_layer_share_lte_0_70"] is False

    for invalid_complete in (False, 1, "true"):
        incomplete = runner.evaluate_support(
            _synthetic_frame(),
            SOURCE_NAMES,
            _load_config(),
            {"complete": invalid_complete},
        )
        assert incomplete["status"] == "NO_GO_ZERO_FIT_SUPPORT_PREFLIGHT"
        assert all(
            prefix["checks"]["provenance_complete"] is False
            for prefix in incomplete["prefixes"].values()
        )
        assert incomplete["execution_audit"]["model_fit_count"] == 0


def test_utility_boundary_is_exactly_strict_not_float_positive() -> None:
    runner = _load_runner()
    start = pd.Timestamp("2025-04-01T00:00:00Z")
    rows = [
        {
            "station": "ANCHOR",
            "year": 2025,
            "layer": 1,
            "time": start - pd.Timedelta(hours=1),
            "fold": "2025_q2",
            "label": 1,
            "anchor": 1,
            "synthetic_tabular": 0,
            "synthetic_neural": 0,
        }
    ]
    for minute, label in ((0, 1), (10, 0), (20, 0)):
        rows.append(
            {
                "station": "PROPOSAL",
                "year": 2025,
                "layer": 2,
                "time": start + pd.Timedelta(minutes=minute),
                "fold": "2025_q2",
                "label": label,
                "anchor": 0,
                "synthetic_tabular": 1,
                "synthetic_neural": 0,
            }
        )
    frame = pd.DataFrame(rows)
    events, _ = runner._build_events(frame, SOURCE_NAMES, _load_config())
    support = runner._support_summary(frame, events, ["2025_q2"])

    assert support["anchor_f1_numerator"] == 2
    assert support["anchor_f1_denominator"] == 3
    assert support["proposal_added_tp"] == 1
    assert support["proposal_added_fp"] == 2
    assert support["utility_numerator"] == 0
    assert support["utility_sum"] == 0.0
    assert support["utility_positive_events"] == 0
    assert support["overall_proposal_precision_strictly_above_requirement"] is False


def test_cell_share_gate_is_inclusive_at_exactly_seventy_percent() -> None:
    runner = _load_runner()
    config = _load_config()
    at_limit = runner.evaluate_support(
        _synthetic_frame(
            {"2025_q2": 10, "2025_q3": 6, "2025_q4": 5},
            q2_cell_indices=[0] * 7 + [1] * 2 + [2],
        ),
        SOURCE_NAMES,
        config,
        {"complete": True},
    )
    over_limit = runner.evaluate_support(
        _synthetic_frame(
            {"2025_q2": 10, "2025_q3": 6, "2025_q4": 5},
            q2_cell_indices=[0] * 8 + [1, 2],
        ),
        SOURCE_NAMES,
        config,
        {"complete": True},
    )

    at_limit_fit = at_limit["prefixes"]["q2_to_q3"]["fit"]
    assert at_limit_fit[
        "maximum_single_station_layer_utility_positive_event_count"
    ] == 7
    assert at_limit_fit["maximum_single_station_layer_utility_positive_share"] == 0.7
    assert at_limit["prefixes"]["q2_to_q3"]["checks"][
        "fit_maximum_single_station_layer_share_lte_0_70"
    ] is True
    assert over_limit["prefixes"]["q2_to_q3"]["checks"][
        "fit_maximum_single_station_layer_share_lte_0_70"
    ] is False


def test_later_fold_cannot_change_an_earlier_prefix() -> None:
    runner = _load_runner()
    config = _load_config()
    frame = _synthetic_frame()
    original = runner.evaluate_support(
        frame, SOURCE_NAMES, config, {"complete": True}
    )
    changed_q4 = frame.copy()
    q4_proposals = changed_q4["fold"].eq("2025_q4") & changed_q4["anchor"].eq(0)
    changed_q4.loc[q4_proposals, "label"] = 0
    altered = runner.evaluate_support(
        changed_q4, SOURCE_NAMES, config, {"complete": True}
    )

    assert altered["prefixes"]["q2_to_q3"] == original["prefixes"]["q2_to_q3"]
    assert (
        altered["prefixes"]["q2_q3_to_q4"]["calibration"]
        != original["prefixes"]["q2_q3_to_q4"]["calibration"]
    )


def test_artifact_and_blind_oof_receipts_fail_closed(tmp_path: Path) -> None:
    runner = _load_runner()
    artifact = tmp_path / "frozen.bin"
    artifact.write_bytes(b"oof")
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "score_path": artifact.name,
                "score_bytes": artifact.stat().st_size,
                "score_sha256": artifact_sha,
            }
        ),
        encoding="utf-8",
    )
    pair = {
        "artifact": {
            "path": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": artifact_sha,
        },
        "receipt": {
            "path": receipt.name,
            "bytes": receipt.stat().st_size,
            "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        },
    }
    verified_path, _, observed = runner._verify_artifact_pair(
        tmp_path, pair, [], {}, {}
    )
    assert verified_path == artifact
    assert observed["artifact"]["sha256"] == artifact_sha

    artifact.write_bytes(b"changed")
    with pytest.raises(runner.ProvenanceError, match="bytes or sha256 changed"):
        runner._verify_artifact_pair(tmp_path, pair, [], {}, {})
    with pytest.raises(runner.ProvenanceError, match="missing or inaccessible"):
        runner._resolve_registered(tmp_path, "missing.bin", [])

    blind_receipt = {
        "experiment_id": "sealed_neural",
        "fold": "2025_q2",
        "holdout_rows": 3,
        "ordered_holdout_key_sha256": "ordered-key-digest",
        "same_fold_holdout_truth_columns_opened_before_receipt": 0,
    }
    record = {"expected_experiment_id": "sealed_neural"}
    runner._validate_neural_receipt(
        blind_receipt, record, "2025_q2", 3, "ordered-key-digest"
    )
    blind_receipt["ordered_holdout_key_sha256"] = "wrong-order"
    with pytest.raises(runner.ProvenanceError, match="receipt contract changed"):
        runner._validate_neural_receipt(
            blind_receipt, record, "2025_q2", 3, "ordered-key-digest"
        )


def test_terminal_json_is_exclusive_and_race_cannot_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    created = tmp_path / "created.json"
    runner._atomic_json(created, {"writer": "first"})
    assert json.loads(created.read_text(encoding="utf-8")) == {"writer": "first"}
    with pytest.raises(FileExistsError):
        runner._atomic_json(created, {"writer": "second"})
    assert json.loads(created.read_text(encoding="utf-8")) == {"writer": "first"}

    raced = tmp_path / "raced.json"
    real_link = runner.os.link

    def rival_wins_before_link(source: Path, destination: Path) -> None:
        Path(destination).write_text('{"writer":"rival"}\n', encoding="utf-8")
        real_link(source, destination)

    monkeypatch.setattr(runner.os, "link", rival_wins_before_link)
    with pytest.raises(FileExistsError):
        runner._atomic_json(raced, {"writer": "preflight"})
    assert json.loads(raced.read_text(encoding="utf-8")) == {"writer": "rival"}
    assert list(tmp_path.glob("*.tmp")) == []
