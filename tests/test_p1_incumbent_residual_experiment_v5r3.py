from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from p1_qc import incumbent_residual_experiment_v5r3 as engine

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "src/p1_qc/incumbent_residual_experiment_v5r3.py"
PROJECTION_PATH = (
    ROOT
    / "configs/experiments/p1_incumbent_rule_distillation_neural_residual_v5_science_projection.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deep_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_pure_engine_has_no_runner_or_dynamic_import_dependency() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "importlib" not in imports
    assert not any(name.startswith("scripts") for name in imported_from)
    assert "run_p1_" not in source
    assert "P1_WORKSPACE_ROOT" not in source
    assert "P1_DATA_DIR" not in source


def test_scientific_projection_is_exact_explicit_allowlist() -> None:
    projection = json.loads(PROJECTION_PATH.read_text(encoding="utf-8"))
    assert _sha(PROJECTION_PATH) == (
        "1b571ee9b755b1e0ff791bfd72adc47b22d7ae571f34affbf2bccfcab4eaa72b"
    )
    assert projection["source_config_sha256"] == (
        "da7427dcfa58daff7d9825653c34296aeb6c4d0648d0d2295715c5e8c0179396"
    )
    assert set(projection["science"]) == set(projection["allowed_keys"])
    assert _deep_sha(projection["science"]) == (
        "a4e976c5e627d82dd360d5efef5126e0b59ed9cb62ffbe26762ebebc0f22b4c9"
    )
    forbidden = {
        "canonical_paths",
        "canonical_identity",
        "immutable_inputs",
        "execution_closure_sha256",
        "authorization_contract",
        "v5_ledger_binding",
    }
    for node in ast.walk(ast.parse(repr(projection["science"]))):
        if isinstance(node, ast.Dict):
            keys = {
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            assert not forbidden.intersection(keys)


def test_engine_exports_complete_scientific_surface() -> None:
    required = {
        "_fold_runtime",
        "_pinned_label_free_prefixes",
        "_verify_gen1_parts",
        "_verify_v5_ledger_binding",
        "_model_config",
        "_training_config",
        "_teacher_oof",
        "_gate_decision",
        "_full_fit_models",
        "_score",
        "verify_relative_input_pins",
        "causal_feature_audit",
        "build_three_block_inner_splits",
        "fit_incumbent_residual_model",
        "predict_incumbent_residual_probability",
    }
    assert required.issubset(engine.__all__)
    assert engine.FRACTIONS == (0.4, 0.55, 0.7, 0.85, 1.0)
    assert engine.SEEDS == (20260813, 20260829, 20260847)
    assert engine.FOLD_ORDER == ("2025_q2", "2025_q3", "2025_q4")
