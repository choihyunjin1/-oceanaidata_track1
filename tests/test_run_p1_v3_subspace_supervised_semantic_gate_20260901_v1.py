from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/p1_v3_subspace_supervised_semantic_gate_20260901_v1.json"


def test_semantic_duplicate_is_closed_at_zero_fit() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["decision"] == "CLOSE_ZERO_FIT_SEMANTIC_DUPLICATE"
    assert payload["zero_fit_contract"]["model_fits"] == 0
    assert payload["zero_fit_contract"]["target_rows"] == 0
    assert "regularized logistic partial pooling" in payload["semantic_predecessor"]["same_roles"]


def test_prior_q3_to_q4_reversal_is_preserved() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    predecessor = payload["semantic_predecessor"]
    assert predecessor["observed_q3_delta_f1"] > 0
    assert predecessor["observed_q4_delta_f1"] < 0
    assert predecessor["observed_pooled_delta_f1"] < 0


def test_unused_graph_candidate_is_audit_only_and_distinct() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    audit = payload["unused_graph_spatiotemporal_audit"]
    assert audit["exact_repository_name_hits"] == 0
    assert audit["semantic_duplicate"] is False
    assert audit["status"] == "AUDIT_ONLY_NOVEL_REPRESENTATION_REQUIRES_PREREGISTRATION"
