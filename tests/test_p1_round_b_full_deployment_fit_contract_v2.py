from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "p1_round_b_full_deployment_fit_contract_20260825_v2"


def test_v2_corrects_production_derived_minimum_and_preserves_v1() -> None:
    contract = json.loads((ARTIFACT / "deployment_fit_contract.json").read_text(encoding="utf-8"))
    assert contract["semantic_pins_derived_with_production_parser"]["minimum_time_kst"] == "2024-01-01T09:00:00+09:00"
    assert contract["preserved_failed_v1"]["append_only"] is True
    assert contract["preserved_failed_v1"]["v1_real_model_fits"] == 0


def test_execute_materializes_real_semantics_before_start_and_fit() -> None:
    path = ARTIFACT / "run_full_deployment_fit.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    body = ast.get_source_segment(source, execute)
    assert body is not None
    assert body.index("_semantic_preflight(") < body.index("_exclusive_start(") < body.index("model.fit(")
    assert "load_dataset(" in source
    assert ".predict(" not in source
    assert ".predict_proba(" not in source


def test_preflight_receipt_proves_full_prefit_path_without_real_fit() -> None:
    receipt = json.loads((ARTIFACT / "semantic_preflight_receipt_r2.json").read_text(encoding="utf-8"))
    facts = receipt["semantic_facts"]
    assert receipt["decision"] == "SEMANTIC_PREFLIGHT_PASS"
    assert receipt["one_shot_marker_created"] is False
    assert facts["matrix_shape"] == [776706, 80]
    assert facts["matrix_no_infinity"] is True
    assert facts["matrix_each_feature_has_finite_support"] is True
    assert facts["real_model_fits"] == 0
    assert all(facts["cache_alignment"].values())
