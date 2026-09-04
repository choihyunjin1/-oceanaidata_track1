from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p1_incumbent_rule_distillation_neural_residual_v5r2.py"
SPEC = importlib.util.spec_from_file_location("test_p1_gen5r2_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data_dir() -> Path:
    return ROOT / "데이터셋 원본/데이터셋_P1/P1_qc_anomaly"


def _config() -> dict[str, object]:
    return json.loads((ROOT / runner.CANONICAL_CONFIG).read_text(encoding="utf-8"))


def test_config_bytes_deep_and_preserved_scientific_structure() -> None:
    path = ROOT / runner.CANONICAL_CONFIG
    config = _config()
    assert _sha(path) == runner.EXPECTED_CONFIG_SHA256
    assert runner._deep_sha(config) == runner.EXPECTED_CONFIG_DEEP_SHA256
    assert config["comparison_mode"] == "EXACT_OFFICIAL_PREFIX_REFIT"
    assert config["preserved_structure"]["hypotheses"] == [
        {
            "id": runner.HYPOTHESIS,
            "order": 1,
            "structure": (
                "exact incumbent identity base plus three-block inner cross-fitted incumbent "
                "teacher, a causal station-layer TCN bounded to plus or minus 0.5 logit, and "
                "a fixed held-out train-only no-op gate"
            ),
            "why_structurally_distinct": (
                "identity-preserving error correction rather than replacement learning or "
                "threshold, class-weight, seed, feature, or postprocess variation"
            ),
            "score_derived_tuning": False,
        }
    ]
    assert config["correction_contract"]["fold_major_order"] == list(runner.FOLD_ORDER)
    assert config["correction_contract"]["prefix_order"] == list(runner.FRACTIONS)
    assert config["correction_contract"]["scientific_structure_changed"] is False


def test_full_transitive_closure_matches_and_contains_required_files() -> None:
    config = _config()
    observed = runner._verify_execution_closure(ROOT, config)
    assert observed == config["execution_closure_sha256"]
    required = {
        "src/p1_qc/binary_event_tcn.py",
        "src/p1_qc/data.py",
        "src/p1_qc/config.py",
        "src/p1_qc/models_tabular.py",
        "src/p1_qc/rules.py",
        "src/p1_qc/validation.py",
        "requirements-lock.txt",
    }
    assert required.issubset(observed)


def test_predecessor_seals_preserved_and_tombstone_exact() -> None:
    paths = runner._paths(ROOT)
    receipt = runner._verify_predecessor_tombstone(paths)
    assert receipt["execution_prohibited"] is True
    assert receipt["predecessor_prereg"] == runner.PREDECESSOR_PREREG_SHA256
    assert receipt["predecessor_preseal"] == runner.PREDECESSOR_PRESEAL_SHA256
    assert receipt["predecessor_tombstone"] == runner.PREDECESSOR_TOMBSTONE_SHA256


def test_run_curve_and_full_fit_require_live_capability_as_first_statement() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for name, phase in (
        ("_run_curve", "BLIND_CURVE"),
        ("_full_fit_models", "FULL_FIT_AUTHORIZED"),
    ):
        first = functions[name].body[0]
        assert isinstance(first, ast.Expr)
        assert isinstance(first.value, ast.Call)
        assert isinstance(first.value.func, ast.Name)
        assert first.value.func.id == "_require_capability"
        assert isinstance(first.value.args[1], ast.Constant)
        assert first.value.args[1].value == phase
    with pytest.raises(PermissionError, match="canonical live"):
        runner._require_capability(None, "BLIND_CURVE")
    with pytest.raises(PermissionError, match="canonical.*mint"):
        runner._ExecutionCapability(
            object(),
            root=ROOT,
            qa_sha256="0" * 64,
            authorization_sha256="1" * 64,
            lock_sha256="2" * 64,
            closure={},
        )


def test_all_gate_predictions_are_sealed_before_selective_gate_label_decode() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    curve = source[source.index("def _run_curve(") : source.index("def _full_fit_models(")]
    gate_fit_loop = curve.index("for gate_seed in SEEDS:")
    final_gate_blind_write = curve.index(
        "gate_stages[gate_seed] = {", gate_fit_loop
    )
    gate_label_decode = curve.index(
        'purpose=f"gate_after_three_blind_seals:', final_gate_blind_write
    )
    refit_loop = curve.index("for seed in SEEDS:", gate_label_decode)
    assert gate_fit_loop < final_gate_blind_write < gate_label_decode < refit_loop
    assert "targets.labels_for(\n                train_ids," not in curve
    assert 'purpose=f"curve_pre_gate_training:' in curve
    assert '"all_three_gate_predictions_sealed_before_gate_label_read": True' in curve


def _blind_file(artifact: Path, relative: str, value: bytes) -> dict[str, object]:
    path = artifact / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {"path": relative, "sha256": _sha(path)}


class _TargetAudit:
    def validation_rows(self, fold: str) -> object:
        return runner.np.asarray([runner.FOLD_ORDER.index(fold)], dtype=runner.np.int64)

    def validation_target_decode_counts(self, fold: str) -> dict[str, int]:
        assert fold in runner.FOLD_ORDER
        return {"label": 0, "anomaly_type": 0}


def test_commitment_ledger_is_fold_major_monotone_and_binds_all_files(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    ledger = runner._BlindCommitmentLedger(artifact)
    targets = _TargetAudit()
    with pytest.raises(PermissionError, match="fold-major monotone"):
        ledger.commit_cell(
            fold="2025_q3",
            fraction=0.4,
            validation_ids_sha256="a" * 64,
            seed_blind_predictions=[],
            prediction_part={"path": "missing", "sha256": "0" * 64},
            target_accessor=targets,
        )
    for fold in runner.FOLD_ORDER:
        for fraction in runner.FRACTIONS:
            tag = runner.shared._tag(fraction)
            seed_pins = []
            for seed in runner.SEEDS:
                pin = _blind_file(
                    artifact,
                    f"blind/{fold}/{tag}/{seed}.npy",
                    f"{fold}:{tag}:{seed}".encode(),
                )
                seed_pins.append({"seed": seed, **pin})
            part_pin = _blind_file(
                artifact,
                f"parts/{fold}_{tag}.parquet",
                f"part:{fold}:{tag}".encode(),
            )
            ledger.commit_cell(
                fold=fold,
                fraction=fraction,
                validation_ids_sha256=runner.shared.ids_sha256(
                    targets.validation_rows(fold)
                ),
                seed_blind_predictions=seed_pins,
                prediction_part={**part_pin, "key_order_sha256": "b" * 64},
                target_accessor=targets,
            )
        assert ledger.is_fold_committed(fold)
    complete = ledger.finalize_ledger()
    assert complete["payload"]["cell_count"] == 15
    assert complete["payload"]["seed_blind_prediction_count"] == 45
    first_cell = runner._strict_json(
        artifact / "blind_commitments/cell_01_2025_q2_p040.json"
    )
    assert first_cell["active_outer_validation_labels_decoded_before_commitment"] == 0
    assert "target_labels_decoded_before_commitment" not in first_cell
    predictions_complete = artifact / "predictions_complete.json"
    runner._json_new(
        predictions_complete,
        {
            "fit_cells": 225,
            "blind_commitment_ledger": {
                "path": complete["path"],
                "sha256": complete["sha256"],
            },
            "prediction_parts": [{}] * 15,
            "model_receipts": [{}] * 45,
        },
    )
    ledger.mark_predictions_complete(predictions_complete)
    assert ledger.is_global_committed()


def test_missing_independent_qa_rejects_before_lock_or_artifact() -> None:
    paths = runner._paths(ROOT)
    assert not paths["qa_receipt"].exists()
    assert not paths["authorization"].exists()
    assert not paths["lock"].exists()
    assert not paths["artifact"].exists()
    with pytest.raises(PermissionError, match="independent-QA receipt is missing"):
        runner.run_experiment(root=ROOT, data_dir=_data_dir())
    assert not paths["lock"].exists()
    assert not paths["artifact"].exists()


def test_run_source_orders_qa_authorization_closure_before_lock() -> None:
    source = ast.unparse(ast.parse(RUNNER_PATH.read_text(encoding="utf-8")))
    start = source.index("def run_experiment")
    body = source[start:]
    assert body.index("_verify_independent_qa(") < body.index(
        "_verify_execution_authorization("
    )
    assert body.index("_verify_execution_authorization(") < body.index(
        "replayed_closure = _verify_execution_closure("
    )
    assert body.index("replayed_closure = _verify_execution_closure(") < body.index(
        "lock = _acquire_lock("
    )
    assert body.index("lock = _acquire_lock(") < body.index("return _run_after_lock(")


def test_static_state_has_no_qa_auth_lock_artifact_or_upload() -> None:
    paths = runner._paths(ROOT)
    assert not paths["qa_receipt"].exists()
    assert not paths["authorization"].exists()
    assert not paths["lock"].exists()
    assert not paths["artifact"].exists()
    config = _config()
    assert config["static_counters"] == {
        "qa_receipts": 0,
        "execution_authorizations": 0,
        "attempt_locks": 0,
        "model_fits": 0,
        "predictions": 0,
        "target_fold_scores": 0,
        "test_value_reads": 0,
        "candidate_files": 0,
        "uploads": 0,
    }


def test_canonical_check_only_is_target_free_and_side_effect_free() -> None:
    paths = runner._paths(ROOT)
    before = {name: path.exists() for name, path in paths.items() if name in {"artifact", "lock"}}
    result = runner.check_only(root=ROOT, data_dir=_data_dir())
    assert result["status"] == "CANONICAL_GEN5R2_CHECK_ONLY_PASS"
    assert result["split_cell_count"] == 15
    assert result["opaque_target_index_decoded_scalars"] == 0
    assert result["frozen_oof_target_columns_decoded"] == 0
    assert result["model_fits"] == result["target_fold_scores"] == 0
    assert result["test_value_reads"] == result["candidate_files"] == result["uploads"] == 0
    after = {name: path.exists() for name, path in paths.items() if name in {"artifact", "lock"}}
    assert after == before
