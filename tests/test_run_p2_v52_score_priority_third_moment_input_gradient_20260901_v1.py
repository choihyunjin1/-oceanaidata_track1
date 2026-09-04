from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_v52_score_priority_third_moment_input_gradient_20260901_v1 as v52  # noqa: E402


def test_v52_preflight_is_zero_operation_policy_clean() -> None:
    value = v52.preflight()
    assert value["status"] == "ZERO_OPERATION_PREFLIGHT_READY"
    assert value["candidate_count"] == 1
    assert value["maximum_fit_count"] == 9
    assert value["data_rows_read"] == 0
    assert value["model_fits"] == 0
    assert value["official_test_index_rows_read"] == 0
    assert value["sample_rows_read"] == 0
    assert value["baseline_file_rows_read"] == 0
    assert value["query_support_rows_read"] == 0
    assert value["hidden_rows_read"] == 0
    assert value["submission_csv_created"] == 0
    assert value["uploads"] == 0
    assert value["lineage"]["organizer_distributed_data_only"] is True
    assert value["lineage"]["pretrained_weight_files_loaded"] == 0


def test_v52_full_source_contract_fails_before_lock() -> None:
    config = v52.load_config()
    receipt = v52._source_guard(config)
    assert receipt["synthetic_missing_key_cases"] == 5
    assert receipt["synthetic_missing_key_rejections"] == list(
        v52.v50._INHERITED_SOURCE_KEYS
    )
    for key in v52.v50._INHERITED_SOURCE_KEYS:
        broken = copy.deepcopy(config)
        del broken["source_contract"][key]
        with pytest.raises(v52.v12.ContractError):
            v52.v50._prelock_source_contract_guard(broken)


def test_v52_combines_fixed_v50_model_and_v23_objective() -> None:
    config = v52.load_config()
    assert config["training"]["input_gradient"]["coefficient"] == 0.01
    assert config["training"]["input_gradient"]["token_channel"] == 0
    assert config["training"]["input_gradient"]["coefficient_sweep"] is False
    assert config["training"]["champion_preserving_weight"] == 0.8
    assert config["training"]["model_weight"] == 0.2
    assert config["training"]["maximum_final_action_C"] == 0.5
    pooling = v52.v50._pooling_contract_receipt()
    assert pooling["moment_order"] == 3
    assert pooling["parameters"] == 5889
    assert pooling["initial_function_maximum_abs_error_vs_v13"] == 0.0
    gradient = v52.v23._gradient_scope_receipt()
    assert gradient["penalized_channel"] == 0
    assert gradient["masked_tokens_excluded"] is True


def test_v52_score_gate_is_fail_closed_against_v23() -> None:
    config = v52.load_config()
    candidate = {
        "delta_rmse": -0.052,
        "by_fold": {
            "2024_sep_oct": {"delta_rmse": -0.01},
            "2025_jul_aug": {"delta_rmse": -0.01},
            "2025_nov_dec": {"delta_rmse": -0.01},
        },
        "bootstrap": {"ci90_high": -0.001},
        "canonical_transport_adjusted_pooled_points_delta": 0.54,
    }
    assert v52._score_gate(candidate, config)["pass"] is True
    not_better = copy.deepcopy(candidate)
    not_better["delta_rmse"] = config["evaluation"]["score_priority_gate"][
        "frozen_v23_internal_delta_rmse_C"
    ]
    assert v52._score_gate(not_better, config)["pass"] is False
    weak_transport = copy.deepcopy(candidate)
    weak_transport["canonical_transport_adjusted_pooled_points_delta"] = 0.52
    assert v52._score_gate(weak_transport, config)["pass"] is False


def test_v52_run_orders_guards_before_exactly_once_base() -> None:
    source = inspect.getsource(v52.run)
    assert source.index("_source_guard") < source.index("_BASE_RUN")
    assert source.index("_ready_pair") < source.index("_BASE_RUN")


def test_v52_ready_preflights_are_byte_identical_and_current() -> None:
    config = v52.load_config()
    paths = [ROOT / item for item in config["ready_preflight_contract"]["paths"]]
    assert all(path.is_file() for path in paths)
    first, second = (path.read_bytes() for path in paths)
    assert first == second
    assert json.loads(first.decode("utf-8")) == v52.preflight()


def test_v52_runner_has_no_network_pretrained_or_official_loader() -> None:
    source = v52.RUNNER.read_text(encoding="utf-8").lower()
    forbidden = (
        "import requests",
        "from requests",
        "import urllib",
        "from urllib",
        "torch.load(",
        "load_state_dict(",
        "from_pretrained(",
        "huggingface_hub",
        "cdsapi",
        "ecmwf.datastores",
        "test_index.csv",
        "sample_submission.csv",
        "baseline_interp.csv",
    )
    assert all(value not in source for value in forbidden)
    config = v52.load_config()
    assert config["source_contract"]["only_source_filename"] == "observations.csv"


def test_v52_worst_fold_layer_is_maximum_regression() -> None:
    record = {
        "by_fold_layer": {
            "a": {"2": {"rows": 1, "delta_rmse": -0.1}},
            "b": {"4": {"rows": 1, "delta_rmse": 0.02}},
        }
    }
    worst = v52._worst_fold_layer(record)
    assert worst["fold"] == "b"
    assert worst["layer"] == "4"
    assert worst["delta_rmse"] == 0.02
