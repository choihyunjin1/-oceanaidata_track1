from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_p1_mstcn_checkpoint_diagnostic_20260827_v1.py"


def _load_runner():
    name = "p1_mstcn_checkpoint_diagnostic_tested"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_modes_are_check_only_or_execute_and_execute_requires_reviewed_hash() -> None:
    runner = _load_runner()
    assert runner._parse_args(["--check-only"]).check_only
    assert runner._parse_args(["--execute", "--expected-runner-sha256", "0" * 64]).execute
    with pytest.raises(SystemExit):
        runner._parse_args([])
    with pytest.raises(SystemExit):
        runner._parse_args(["--check-only", "--execute"])
    with pytest.raises(runner.ContractError, match="reviewed runner bytes"):
        runner.execute(expected_runner_sha256="0" * 64)


def test_config_is_hash_sealed_and_has_exact_fixed_recipe() -> None:
    runner = _load_runner()
    config = runner._canonical_config()
    assert runner._sha256(runner.CONFIG_PATH) == runner.EXPECTED_CONFIG_SHA256
    assert config["fixed_recipe"] == {
        "width": 512,
        "epoch": 150,
        "threshold": 0.8,
        "representation": "raw_three_seed_ensemble_mean",
        "seeds": [20260827, 20260839, 20260863],
        "blind_prediction_epochs": [120, 125, 130, 145, 150],
        "saved_state_epochs": [145, 150],
    }
    evaluation = config["evaluation_contract"]
    assert evaluation["truth_scored_epochs"] == [150]
    assert evaluation["same_truth_oracle_diagnostic_epochs"] == [120, 125, 130, 145]
    assert evaluation["same_truth_oracle_promotion_evidence"] is False
    assert evaluation["same_truth_oracle_recipe_mutation_allowed"] is False


def test_all_source_pins_match_without_opening_outer_truth() -> None:
    runner = _load_runner()
    config = runner._canonical_config()
    observed = runner._verify_source_pins(config)
    assert observed == config["source_pins"]
    assert set(observed) == {
        "runner",
        "config",
        "model",
        "data",
        "q2_grid",
        "q2_receipt",
        "current_router_anchor",
        "frozen_oof",
    }


def test_plateau_selection_uses_both_neighbors_and_deterministic_ties() -> None:
    runner = _load_runner()
    widths = np.asarray([256, 256, 256, 512, 512, 512], dtype=np.int16)
    epochs = np.asarray([145, 150, 155, 145, 150, 155], dtype=np.int16)
    thresholds = np.asarray([0.8], dtype=np.float64)
    deltas = np.asarray([[0.06], [0.09], [0.06], [0.076], [0.075], [0.074]])
    selected = runner.select_plateau_recipe(
        widths=widths,
        epochs=epochs,
        thresholds=thresholds,
        deltas=deltas,
        neighbor_offset=5,
    )
    assert selected["width"] == 512
    assert selected["epoch"] == 150
    assert selected["threshold"] == 0.8
    assert selected["neighbor_epochs"] == [145, 150, 155]
    assert selected["worst_neighbor_delta_f1"] == pytest.approx(0.074)


def test_monthly_gate_is_precision_aware() -> None:
    runner = _load_runner()

    class FakeSource:
        @staticmethod
        def binary_metrics(truth, prediction):
            truth = np.asarray(truth)
            prediction = np.asarray(prediction)
            tp = int(np.sum((truth == 1) & (prediction == 1)))
            fp = int(np.sum((truth == 0) & (prediction == 1)))
            fn = int(np.sum((truth == 1) & (prediction == 0)))
            f1 = 2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            return {"tp": tp, "fp": fp, "fn": fn, "f1": f1}

    truth = np.asarray([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
    anchor = np.asarray([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8)
    candidate = np.asarray([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
    months = np.asarray(["A"] * 4 + ["B"] * 4)
    result = runner._fixed_monthly_metrics(
        FakeSource(),
        truth=truth,
        anchor=anchor,
        candidate=candidate,
        months=months,
        registered_months=["A", "B"],
    )
    assert all(row["delta_positive"] for row in result.values())
    assert all(row["added_precision_gate"] for row in result.values())


def test_execute_seals_fixed_decision_before_same_truth_oracle() -> None:
    runner = _load_runner()
    source = inspect.getsource(runner.execute)
    assert source.index("fixed_epoch_150_decision.json") < source.index(
        "_evaluate_same_truth_oracle_diagnostic"
    )
    assert source.index("_evaluate_same_truth_oracle_diagnostic") < source.index(
        "same_truth_oracle_diagnostic.json"
    )


def test_same_truth_oracle_cannot_mutate_recipe_or_be_promotion_evidence(
    tmp_path: Path,
) -> None:
    runner = _load_runner()

    class FakeSource:
        @staticmethod
        def binary_metrics(truth, prediction):
            truth = np.asarray(truth)
            prediction = np.asarray(prediction)
            tp = int(np.sum((truth == 1) & (prediction == 1)))
            fp = int(np.sum((truth == 0) & (prediction == 1)))
            fn = int(np.sum((truth == 1) & (prediction == 0)))
            f1 = 2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            return {"tp": tp, "fp": fp, "fn": fn, "f1": f1}

    recipe_path = tmp_path / "selected_recipe.json"
    fixed_path = tmp_path / "fixed_epoch_150_decision.json"
    recipe_path.write_text('{"epoch":150}\n', encoding="utf-8")
    fixed_path.write_text('{"status":"SEALED"}\n', encoding="utf-8")
    recipe_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    epochs = np.asarray([120, 125, 130, 145, 150], dtype=np.int16)
    q3_candidates = np.asarray(
        [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
        dtype=np.int8,
    )
    q4_candidates = q3_candidates.copy()
    verified = {
        "q3": ({}, {"epochs": epochs, "candidate": q3_candidates}, {}),
        "q4": ({}, {"epochs": epochs, "candidate": q4_candidates}, {}),
    }
    truths = {
        "q3": pd.DataFrame({"label": [1, 1, 0, 0]}),
        "q4": pd.DataFrame({"label": [1, 1, 0, 0]}),
    }
    holdout = SimpleNamespace(
        surface=SimpleNamespace(anchor=np.asarray([1, 0, 0, 0], dtype=np.int8))
    )
    result = runner._evaluate_same_truth_oracle_diagnostic(
        FakeSource(),
        truths,
        {"q3": holdout, "q4": holdout},
        verified,
        fixed_decision_path=fixed_path,
        recipe_path=recipe_path,
        oracle_epochs=[120, 125, 130, 145],
    )
    assert result["same_truth_oracle_best"]["epoch"] in {125, 145}
    assert result["promotion_evidence"] is False
    assert result["recipe_mutated"] is False
    assert hashlib.sha256(recipe_path.read_bytes()).hexdigest() == recipe_sha


def test_runner_has_no_protected_interface_filename_literals() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8").casefold()
    protected = [
        "".join(map(chr, (116, 101, 115, 116, 46, 99, 115, 118))),
        "".join(
            map(
                chr,
                (
                    115,
                    97,
                    109,
                    112,
                    108,
                    101,
                    95,
                    115,
                    117,
                    98,
                    109,
                    105,
                    115,
                    115,
                    105,
                    111,
                    110,
                    46,
                    99,
                    115,
                    118,
                ),
            )
        ),
    ]
    assert all(value not in source for value in protected)
    execute_source = inspect.getsource(_load_runner().execute).casefold()
    assert all(
        forbidden not in execute_source
        for forbidden in ("requests.", "webbrowser", "selenium", "playwright")
    )
    assert '"upload_performed": false' in execute_source
