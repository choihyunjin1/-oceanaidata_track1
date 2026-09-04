from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

RUNNER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_p1_tetad_lite_direct_interval_set_v1.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("p1_tetad_lite_runner_tested", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _minimal_config() -> dict:
    return {
        "chronological_protocol": {
            "q2": {"threshold_grid": [0.5, 0.7, 0.85]},
        },
        "target_definition": {"minimum_original_event_rows": 19},
        "outer_decision": {
            "confirmation_folds": ["2025_q3", "2025_q4"],
            "bootstrap_replicates": 20,
            "bootstrap_seed": 17,
            "go_replication_if_all": {
                "pooled_delta_f1_min": 0.003,
                "each_fold_delta_f1_min": 0.0,
                "paired_bootstrap_delta_f1_ci90_lower_strictly_above": 0.0,
                "normal_false_positive_per_day_relative_increase_max": 0.10,
                "worst_supported_station_layer_f1_drop_max": 0.01,
                "supported_station_layer_positive_rows_min": 50,
            },
        },
    }


def test_check_and_seal_modes_never_construct_a_training_runtime(monkeypatch) -> None:
    runner = _load_runner()
    calls: list[str] = []
    monkeypatch.setattr(runner, "_load_config", lambda: {"fixed": True})
    monkeypatch.setattr(
        runner,
        "build_preflight",
        lambda _config: calls.append("preflight") or {"result": "PASS"},
    )
    monkeypatch.setattr(
        runner,
        "create_execution_seal",
        lambda _config, _preflight: calls.append("seal") or {"sealed": True},
    )

    class ForbiddenRuntime:
        def __init__(self, _config) -> None:
            raise AssertionError("a zero-fit mode constructed the training runtime")

    monkeypatch.setattr(runner, "ScientificRuntime", ForbiddenRuntime)
    assert runner.main(["--check-only"]) == 0
    assert calls == ["preflight"]
    calls.clear()
    assert runner.main(["--seal-only"]) == 0
    assert calls == ["preflight", "seal"]


def test_attempt_lock_is_created_with_o_excl(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    seal_path = tmp_path / "seal.json"
    seal_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "SEAL_PATH", seal_path)
    monkeypatch.setattr(runner, "ATTEMPT_LOCK", tmp_path / "attempt.json")
    monkeypatch.setattr(runner, "TERMINAL_PATH", tmp_path / "terminal.json")
    seal = {"files": {"config": {"sha256": "a" * 64}}}
    runner.acquire_attempt_lock(seal)
    with pytest.raises(FileExistsError):
        runner.acquire_attempt_lock(seal)


def test_truth_access_follows_receipt_and_q4_runs_after_negative_q3(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    config = _minimal_config()
    calls: list[str] = []

    class Runtime:
        def sanity_fit(self):
            calls.append("sanity")
            return {"finite": True}

        def sanity_passes(self, _metrics):
            return True

        def fit_predict_blind(self, fold):
            calls.append(f"fit:{fold}")
            return SimpleNamespace(fold=fold)

        def load_truth_after_receipt(self, blind, receipt):
            assert receipt == tmp_path / f"{blind.fold}.receipt"
            assert f"commit:{blind.fold}" in calls
            calls.append(f"truth:{blind.fold}")
            return SimpleNamespace(fold=blind.fold)

    monkeypatch.setattr(runner, "_load_verified_seal", lambda: {"files": {}})
    monkeypatch.setattr(runner, "acquire_attempt_lock", lambda _seal: calls.append("lock"))

    def commit(blind):
        calls.append(f"commit:{blind.fold}")
        return tmp_path / f"{blind.fold}.receipt"

    monkeypatch.setattr(runner, "_commit_blind", commit)
    monkeypatch.setattr(
        runner,
        "_select_q2_threshold",
        lambda *_args: (0.7, {"selected_threshold": 0.7}, False),
    )

    def score(_config, _runtime, blind, _truth, _threshold):
        delta = -0.2 if blind.fold == "2025_q3" else 0.1
        calls.append(f"score:{blind.fold}:{delta}")
        return {"aggregate": {"f1_delta": delta}}

    monkeypatch.setattr(runner, "_score_fold", score)
    monkeypatch.setattr(
        runner,
        "_outer_decision",
        lambda *_args: ("NO_GO_NEW_ARCHITECTURE", {"gate_checks": {}}),
    )
    writes: list[Path] = []
    monkeypatch.setattr(runner, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(runner, "TERMINAL_PATH", tmp_path / "terminal.json")
    monkeypatch.setattr(
        runner,
        "_exclusive_json",
        lambda path, _value: writes.append(path),
    )

    result = runner.execute_one_shot(config, Runtime())
    assert result["status"] == "NO_GO_NEW_ARCHITECTURE"
    assert calls.index("commit:2025_q3") < calls.index("truth:2025_q3")
    assert calls.index("score:2025_q3:-0.2") < calls.index("fit:2025_q4")
    assert calls.index("commit:2025_q4") < calls.index("truth:2025_q4")
    assert writes[-1] == tmp_path / "terminal.json"


def test_anchor_union_path_cannot_turn_one_into_zero() -> None:
    runner = _load_runner()
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "anchor_preserving_union" in source

    class Column:
        def __init__(self, values):
            self.values = np.asarray(values)

        def to_numpy(self):
            return self.values

    class Truth:
        def get_column(self, name):
            assert name == "label"
            return Column([1, 0, 1, 0])

        def select(self, _columns):
            return self

    class Engine:
        KEY_COLUMNS = ("station", "year", "layer", "time")

        @staticmethod
        def anchor_preserving_union(anchor, proposal):
            result = np.maximum(anchor, proposal).astype(np.int8)
            assert not np.any((np.asarray(anchor) == 1) & (result == 0))
            return result

        @staticmethod
        def compare_anchor_candidate(_truth, anchor, candidate):
            assert not np.any((np.asarray(anchor) == 1) & (candidate == 0))
            return {"overall": {"candidate": {"f1": 1.0}}}

        @staticmethod
        def exact_cadence_segments(_keys):
            return []

        @staticmethod
        def eligible_target_events(*_args, **_kwargs):
            return []

    runtime = SimpleNamespace(np=np, engine=Engine())
    blind = runner.BlindFold(
        fold="2025_q3",
        confidence=np.asarray([0.0, 1.0, 0.0, 0.0]),
        anchor=np.asarray([1, 0, 1, 0], dtype=np.int8),
        keys=None,
    )
    score = runner._score_fold(_minimal_config(), runtime, blind, Truth(), 0.5)
    np.testing.assert_array_equal(score["candidate"], [1, 1, 1, 0])


def test_runner_contains_no_protected_interface_filename_literals() -> None:
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
    assert all(name not in source for name in protected)


def test_blind_receipt_rejects_changed_score_bytes(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    score = tmp_path / "scores.npz"
    score.write_bytes(b"before")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "experiment_id": runner.EXPERIMENT_ID,
                "score_path": score.name,
                "score_sha256": runner._sha256(score),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    runner._verify_blind_receipt(receipt)
    score.write_bytes(b"after")
    with pytest.raises(runner.ContractError, match="changed"):
        runner._verify_blind_receipt(receipt)
