from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_masked_logmeanexp_profile_pooling_deepset_20260901_v49 as v49  # noqa: E402


def test_v49_preflight_is_ready_zero_operation_and_policy_clean() -> None:
    before = v49.ARTIFACT.exists()
    value = v49.preflight()
    assert v49.ARTIFACT.exists() is before
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


def test_v49_pooling_is_identity_initialized_supported_and_distinct() -> None:
    receipt = v49._pooling_contract_receipt()
    assert receipt["descriptor"] == (
        "masked_log_mean_exp(element_embedding, temperature=1.0)"
    )
    assert receipt["new_head_columns_initial_maximum_abs"] == 0.0
    assert receipt["unchanged_parameter_maximum_abs_error_vs_v13"] == 0.0
    assert receipt["initial_function_maximum_abs_error_vs_v13"] == 0.0
    assert receipt["logmeanexp_manual_maximum_abs_error"] == 0.0
    assert receipt["all_missing_logmeanexp_maximum_abs"] == 0.0
    assert receipt["logmeanexp_minus_mean_minimum"] >= -1e-6
    assert receipt["maximum_minus_logmeanexp_minimum"] >= -1e-6
    assert receipt["learned_pooling_columns_change_function_maximum_abs"] > 0.0
    assert receipt["new_head_column_gradient_finite_nonzero"] is True
    assert receipt["parameters"] == 5889
    assert receipt["linear_count"] == 5
    assert receipt["normalization_count"] == 0
    assert receipt["dropout_count"] == 0
    assert receipt["attention_count"] == 0


def test_v49_permutation_mask_and_repeat_isolation() -> None:
    receipt = v49._isolation_receipt()
    assert max(receipt.values()) <= 1e-6


def test_v49_ready_preflights_are_byte_identical_and_current() -> None:
    config = v49.load_config()
    paths = [ROOT / value for value in config["ready_preflight_contract"]["paths"]]
    assert all(path.is_file() for path in paths)
    first, second = (path.read_bytes() for path in paths)
    assert first == second
    stored = json.loads(first.decode("utf-8"))
    assert stored == v49.preflight()
    assert stored["status"] == "ZERO_OPERATION_PREFLIGHT_READY"


def test_v49_runner_has_no_network_or_pretrained_weight_loader() -> None:
    source = v49.RUNNER.read_text(encoding="utf-8").lower()
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
    config = v49.load_config()
    assert config["source_contract"]["only_direct_source_filename"] == (
        "observations.csv"
    )
    assert config["training"]["pretrained_weight_files_loaded"] == 0
    assert config["result_adaptive_tuning"] is False


def test_v49_masked_logmeanexp_all_missing_is_finite_zero() -> None:
    torch.manual_seed(49)
    model = v49.MaskedLogMeanExpProfileVerticalDeepSet(8, 11, hidden=32)
    encoded = torch.randn(4, 5, 32)
    mask = torch.zeros(4, 5)
    pooled = model.masked_logmeanexp(encoded, mask)
    assert torch.isfinite(pooled).all()
    assert torch.count_nonzero(pooled) == 0
