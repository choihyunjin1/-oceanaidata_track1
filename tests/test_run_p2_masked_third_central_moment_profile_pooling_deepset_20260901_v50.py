from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_masked_third_central_moment_profile_pooling_deepset_20260901_v50 as v50  # noqa: E402


def test_v50_preflight_is_ready_zero_operation_and_policy_clean() -> None:
    before = v50.ARTIFACT.exists()
    value = v50.preflight()
    assert v50.ARTIFACT.exists() is before
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
    assert value["lineage"]["fresh_random_scratch_initialization"] is True
    assert value["lineage"]["pretrained_weight_files_loaded"] == 0


def test_v50_full_inherited_source_contract_fails_before_lock() -> None:
    config = v50.load_config()
    receipt = v50._synthetic_source_contract_guard_receipt(config)
    assert receipt["all_required_keys_present"] is True
    assert receipt["direct_alias_matches"] is True
    assert receipt["synthetic_missing_key_cases"] == 5
    assert receipt["synthetic_missing_key_rejections"] == list(
        v50._INHERITED_SOURCE_KEYS
    )
    assert all(receipt["inherited_implementation_references"].values())
    assert receipt["synthetic_files_opened"] == 0
    assert receipt["synthetic_data_rows_read"] == 0
    assert receipt["synthetic_model_fits"] == 0
    for key in v50._INHERITED_SOURCE_KEYS:
        broken = copy.deepcopy(config)
        del broken["source_contract"][key]
        with pytest.raises(v50.v12.ContractError):
            v50._prelock_source_contract_guard(broken)


def test_v50_run_orders_source_guard_before_inherited_base_lock() -> None:
    source = inspect.getsource(v50.run)
    assert source.index("_synthetic_source_contract_guard_receipt") < source.index(
        "_BASE_RUN()"
    )
    assert 'source_contract"]["only_source_filename' in inspect.getsource(
        v50.v12.resolve_observations
    )
    assert 'source_contract"]["scoring_frame' in inspect.getsource(v50.v13.run)


def test_v50_pooling_is_identity_initialized_supported_and_distinct() -> None:
    receipt = v50._pooling_contract_receipt()
    assert receipt["descriptor"] == (
        "masked_mean((element_embedding-masked_mean(element_embedding))^3)"
    )
    assert receipt["moment_order"] == 3
    assert receipt["new_head_columns_initial_maximum_abs"] == 0.0
    assert receipt["unchanged_parameter_maximum_abs_error_vs_v13"] == 0.0
    assert receipt["initial_function_maximum_abs_error_vs_v13"] == 0.0
    assert receipt["third_moment_manual_maximum_abs_error"] == 0.0
    assert receipt["all_missing_third_moment_maximum_abs"] == 0.0
    assert receipt["third_moment_nonconstant_range"] > 0.0
    assert receipt["learned_pooling_columns_change_function_maximum_abs"] > 0.0
    assert receipt["new_head_column_gradient_finite_nonzero"] is True
    assert receipt["parameters"] == 5889
    assert receipt["linear_count"] == 5
    assert receipt["normalization_count"] == 0
    assert receipt["dropout_count"] == 0
    assert receipt["attention_count"] == 0


def test_v50_permutation_mask_and_repeat_isolation() -> None:
    receipt = v50._isolation_receipt()
    assert max(receipt.values()) <= 1e-6


def test_v50_ready_preflights_are_byte_identical_and_current() -> None:
    config = v50.load_config()
    paths = [ROOT / value for value in config["ready_preflight_contract"]["paths"]]
    assert all(path.is_file() for path in paths)
    first, second = (path.read_bytes() for path in paths)
    assert first == second
    stored = json.loads(first.decode("utf-8"))
    assert stored == v50.preflight()
    assert stored["status"] == "ZERO_OPERATION_PREFLIGHT_READY"


def test_v50_runner_has_no_network_or_pretrained_weight_loader() -> None:
    source = v50.RUNNER.read_text(encoding="utf-8").lower()
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
    )
    assert all(value not in source for value in forbidden)
    config = v50.load_config()
    assert config["source_contract"]["only_source_filename"] == "observations.csv"
    assert config["source_contract"]["only_direct_source_filename"] == (
        "observations.csv"
    )
    assert config["training"]["pretrained_weight_files_loaded"] == 0
    assert config["result_adaptive_tuning"] is False


def test_v50_masked_third_moment_all_missing_is_finite_zero() -> None:
    torch.manual_seed(50)
    model = v50.MaskedThirdCentralMomentProfileVerticalDeepSet(8, 11, hidden=32)
    encoded = torch.randn(4, 5, 32)
    mask = torch.zeros(4, 5)
    pooled = model.masked_third_central_moment(encoded, mask)
    assert torch.isfinite(pooled).all()
    assert torch.count_nonzero(pooled) == 0
