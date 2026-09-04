from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p1_mstcn_official_shadow_lower_bound_veto_20260829_v1.py"
SPEC = importlib.util.spec_from_file_location("shadow_veto", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paths_and_contract_are_shadow_only() -> None:
    config = MODULE.json.loads(MODULE.CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["bootstrap_replicates"] == 1000
    assert config["acceptance_frequency"] == 0.9
    assert config["deployment_contract"]["write_candidate_csv"] is False
    assert config["deployment_contract"]["upload"] is False
    assert MODULE.CHAMPION_ENV == "P1_CHAMPION_SUBMISSION"
    assert MODULE.resolve_champion_path({"P1_CHAMPION_SUBMISSION": str(SCRIPT)}).name == SCRIPT.name


def test_candidate_preservation_logic() -> None:
    champion = np.array([1, 1, 0, 1, 1], dtype=np.int8)
    anchor = np.array([1, 0, 0, 1, 0], dtype=np.int8)
    segment_indices = [np.array([1]), np.array([4])]
    acceptance = np.array([True, False])
    shadow = champion.copy()
    for keep, positions in zip(acceptance, segment_indices, strict=True):
        if not keep:
            shadow[positions] = 0
    assert np.sum((anchor == 1) & (shadow == 0)) == 0
    assert shadow.tolist() == [1, 1, 0, 1, 0]
