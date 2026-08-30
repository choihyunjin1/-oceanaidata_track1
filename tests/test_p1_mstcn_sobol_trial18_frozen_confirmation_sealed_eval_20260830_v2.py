from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_mstcn_sobol_trial18_frozen_confirmation_sealed_eval_20260830_v2"
RUNNER_PATH = ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py"


def _load_runner():
    name = f"test_{EXPERIMENT_ID}"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _different_phase_fixture():
    holds = {
        "q3": SimpleNamespace(surface=SimpleNamespace(keys="q3-key-sha")),
        "q4": SimpleNamespace(surface=SimpleNamespace(keys="q4-key-sha")),
    }
    paths = {
        "q3": Path("q3_confirmatory_blind_receipt.json"),
        "q4": Path("q4_confirmatory_blind_receipt.json"),
    }
    base = SimpleNamespace(_ordered_key_sha=lambda keys: keys)
    original = SimpleNamespace(_verify_blind_receipt=None)
    return original, base, holds, paths


def _strict_verifier(calls):
    expected = {
        "q3_confirmatory_blind_receipt.json": "q3-key-sha",
        "q4_confirmatory_blind_receipt.json": "q4-key-sha",
    }

    def verify(path, *, config_sha256, recipe_sha256, expected_key_sha256):
        assert config_sha256 == "config-sha"
        assert recipe_sha256 == "recipe-sha"
        if expected[path.name] != expected_key_sha256:
            raise RuntimeError("cross-phase receipt/key swap")
        calls.append((path.name, expected_key_sha256))
        return {"phase": path.name[:2]}

    return verify


def test_phase_specific_receipts_accept_distinct_q3_q4_keys() -> None:
    runner = _load_runner()
    original, base, holds, paths = _different_phase_fixture()
    calls = []
    audit = runner.verify_phase_receipts(
        original,
        base,
        holds,
        paths,
        config_sha256="config-sha",
        recipe_sha256="recipe-sha",
        verify=_strict_verifier(calls),
    )
    assert calls == [
        ("q3_confirmatory_blind_receipt.json", "q3-key-sha"),
        ("q4_confirmatory_blind_receipt.json", "q4-key-sha"),
    ]
    assert [row["phase"] for row in audit] == ["q3", "q4"]


def test_cross_swapped_receipts_are_rejected() -> None:
    runner = _load_runner()
    original, base, holds, paths = _different_phase_fixture()
    swapped = {"q3": paths["q4"], "q4": paths["q3"]}
    with pytest.raises(RuntimeError, match="cross-phase"):
        runner.verify_phase_receipts(
            original,
            base,
            holds,
            swapped,
            config_sha256="config-sha",
            recipe_sha256="recipe-sha",
            verify=_strict_verifier([]),
        )


def test_full_receipt_loop_dry_run_visits_each_phase_once() -> None:
    runner = _load_runner()
    original, base, holds, paths = _different_phase_fixture()
    calls = []
    runner.verify_phase_receipts(
        original,
        base,
        holds,
        paths,
        config_sha256="config-sha",
        recipe_sha256="recipe-sha",
        verify=_strict_verifier(calls),
    )
    assert len(calls) == 2
    assert len({name for name, _key in calls}) == 2
    assert len({key for _name, key in calls}) == 2


def test_config_allows_only_two_repairs_and_zero_fit() -> None:
    runner = _load_runner()
    config = runner._config()
    recovery = config["recovery_contract"]
    assert len(recovery["code_deltas"]) == 2
    assert recovery["additional_model_fits"] == 0
    assert recovery["reprediction"] is False
    assert recovery["completed_historical_truth_metric_evaluations_before_v2"] == 0
    assert recovery["historical_truth_metric_evaluations_authorized_in_v2"] == 1
    assert all(config["prohibitions"].values())


def test_preflight_pins_both_failures_and_blind_bytes_without_truth() -> None:
    runner = _load_runner()
    before = runner._verify_pins(runner._config())
    result = runner.check_only()
    after = runner._verify_pins(runner._config())
    assert result["decision"] == "PASS"
    assert all(result["checks"].values())
    assert before == after
    assert result["additional_model_fits_authorized"] == 0
    assert result["official_test_sample_submission_hidden_rows_read"] == 0


def test_exclusive_terminal_write_does_not_overwrite(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "terminal_result.json"
    runner._exclusive_json(path, {"attempt": 1})
    with pytest.raises(FileExistsError):
        runner._exclusive_json(path, {"attempt": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"attempt": 1}


def test_v2_source_has_no_fit_predict_csv_or_upload_call() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    prohibited_calls = (
        "._fit_one(",
        "._train_epoch(",
        ".predict_encoded(",
        ".to_csv(",
        ".upload(",
    )
    assert not any(token in source for token in prohibited_calls)


def test_synthetic_smoke_declares_zero_fit_and_one_metric_evaluation() -> None:
    runner = _load_runner()
    result = runner.run_smoke()
    assert result["decision"] == "PASS"
    assert result["code_delta_count"] == 2
    assert result["additional_model_fits_authorized"] == 0
    assert result["historical_truth_metric_evaluations_authorized"] == 1
