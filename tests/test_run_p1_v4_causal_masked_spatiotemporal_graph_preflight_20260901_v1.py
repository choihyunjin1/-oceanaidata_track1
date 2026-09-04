from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/p1_v4_causal_masked_spatiotemporal_graph_preflight_20260901_v1.json"


def test_architecture_is_novel_but_data_contract_blocked() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["semantic_audit"]["novel"] is True
    assert payload["semantic_audit"]["semantic_duplicate"] is False
    assert payload["data_contract"]["readme_exposes_station_coordinates"] is False
    assert payload["data_contract"]["signed_horizontal_edge_manifest_present"] is False
    assert payload["decision"] == "CLOSE_ZERO_FIT_MISSING_HORIZONTAL_GRAPH_CONTRACT"


def test_capacity_and_fit_ceiling_were_frozen_before_metrics() -> None:
    architecture = json.loads(CONFIG.read_text(encoding="utf-8"))["architecture"]
    assert architecture["maximum_fits_if_unblocked"] == 9
    assert architecture["epochs"] == 20
    assert architecture["hidden_width"] == 64
    assert architecture["weight_decay"] == 0.0001
    assert architecture["sweep_count"] == 0


def test_simpler_candidate_avoids_missing_horizontal_graph() -> None:
    candidate = json.loads(CONFIG.read_text(encoding="utf-8"))["simpler_unused_candidate"]
    assert "no horizontal graph" in candidate["description"]
    assert candidate["status"] == "AUDIT_ONLY_REQUIRES_NEW_PREREGISTRATION"
