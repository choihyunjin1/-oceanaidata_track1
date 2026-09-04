from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_public_transport_calibration_20260831_v2.py"


def load_module():
    spec = importlib.util.spec_from_file_location("transport_calibration_v2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def built():
    module = load_module()
    v1 = json.loads(module.V1_CALIBRATION.read_text(encoding="utf-8"))
    registry = json.loads(module.PASS_REGISTRY.read_text(encoding="utf-8"))
    return module, module.build_calibration(v1, registry)


def test_family_and_tier_penalties_are_recomputed_from_official_pairs() -> None:
    _, result = built()
    assert len(result["observed_pairs"]) == 6
    assert abs(
        result["family_gates"]["P3_FIXED_KMA_LONGLEAD_FACTOR"][
            "transport_penalty_points"
        ]
        - 0.04958605409228893
    ) < 1e-12
    assert abs(
        result["tier_gates"]["LOW_DOF_FIXED"][
            "minimum_raw_expected_points_delta"
        ]
        - 0.05958605409228893
    ) < 1e-12
    assert abs(
        result["tier_gates"]["SMOOTH_LEARNED_PROFILE"][
            "minimum_raw_expected_points_delta"
        ]
        - 0.13168209161000616
    ) < 1e-12
    assert abs(
        result["tier_gates"]["HARD_CONDITIONAL_ROUTER"][
            "minimum_raw_expected_points_delta"
        ]
        - 0.33190568975947593
    ) < 1e-12


def test_exact_family_precedes_tier() -> None:
    module, result = built()
    gate = module.select_penalty(
        result,
        family_id="P1_FIXED_ADD_ONLY_UNION",
        tier_id="LOW_DOF_FIXED",
        representation_changed=False,
        routing_discontinuous=False,
    )
    assert gate["source"] == "exact_family:P1_FIXED_ADD_ONLY_UNION"
    assert abs(gate["minimum_raw_expected_points_delta"] - 0.015383691373120248) < 1e-12


def test_unseen_low_dof_family_uses_tier_worst() -> None:
    module, result = built()
    gate = module.select_penalty(
        result,
        family_id="P3_FIXED_ERA5_GLOBAL_SHRINK",
        tier_id="LOW_DOF_FIXED",
        representation_changed=True,
        routing_discontinuous=False,
    )
    assert gate["source"] == "complexity_tier:LOW_DOF_FIXED"
    assert abs(gate["minimum_raw_expected_points_delta"] - 0.05958605409228893) < 1e-12


def test_compound_change_uses_global_fallback() -> None:
    module, result = built()
    gate = module.select_penalty(
        result,
        family_id="NEW_COMPOUND",
        tier_id="LOW_DOF_FIXED",
        representation_changed=True,
        routing_discontinuous=True,
    )
    assert gate["source"] == "global_unknown_or_compound_fallback"
    assert abs(gate["minimum_raw_expected_points_delta"] - 0.33190568975947593) < 1e-12
    assert result["minimum_calibrated_expected_points_delta"] == 0.01
