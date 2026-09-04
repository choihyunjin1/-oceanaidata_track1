from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_public_transport_calibration_20260831_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("transport_calibration", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_calibration_recomputes_observed_residuals_and_thresholds() -> None:
    module = load_module()
    official = json.loads(module.OFFICIAL_RESULTS.read_text(encoding="utf-8"))
    registry = json.loads(module.PASS_REGISTRY.read_text(encoding="utf-8"))
    legacy_receipts = [
        json.loads(path.read_text(encoding="utf-8")) for path in module.LEGACY_RECEIPTS
    ]
    result = module.build_calibration(official, registry, legacy_receipts)

    assert result["minimum_point_gain"] == 0.01
    assert result["gates"]["P1"]["observations"] == 1
    assert result["gates"]["P2"]["observations"] == 3
    assert result["gates"]["P3"]["observations"] == 2
    assert abs(
        result["gates"]["P1"]["minimum_uncalibrated_expected_points_delta"]
        - 0.015383691373120247
    ) < 1e-12
    assert abs(
        result["gates"]["P2"]["minimum_uncalibrated_expected_points_delta"]
        - 0.13168209161000617
    ) < 1e-12
    assert abs(
        result["gates"]["P3"]["minimum_uncalibrated_expected_points_delta"]
        - 0.3319056897594759
    ) < 1e-12


def test_every_gate_uses_inclusive_point_floor() -> None:
    module = load_module()
    official = json.loads(module.OFFICIAL_RESULTS.read_text(encoding="utf-8"))
    registry = json.loads(module.PASS_REGISTRY.read_text(encoding="utf-8"))
    legacy_receipts = [
        json.loads(path.read_text(encoding="utf-8")) for path in module.LEGACY_RECEIPTS
    ]
    result = module.build_calibration(official, registry, legacy_receipts)
    assert all(
        gate["minimum_calibrated_expected_points_delta"] >= 0.01
        for gate in result["gates"].values()
    )
