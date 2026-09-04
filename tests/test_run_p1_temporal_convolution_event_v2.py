from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p1_station_layer_temporal_convolution_event_v2.py"
SPEC = importlib.util.spec_from_file_location("p1_temporal_event_runner_test", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)
DATA_DIR = ROOT / "데이터셋 원본/데이터셋_P1/P1_qc_anomaly"


def _deep_sha(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_config_byte_deep_structure_and_training_contract() -> None:
    path = ROOT / runner.CANONICAL_CONFIG
    config = json.loads(path.read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == runner.EXPECTED_CONFIG_SHA256
    assert _deep_sha(config) == runner.EXPECTED_CONFIG_DEEP_SHA256
    assert [item["id"] for item in config["hypotheses"]] == [runner.HYPOTHESIS]
    assert config["prefix_fractions"] == list(runner.FRACTIONS)
    assert config["seeds"] == list(runner.SEEDS)
    assert config["model"]["receptive_field_rows"] == 31
    assert config["model"]["heads"] == ["onset", "interior", "offset"]
    assert config["training"]["expected_curve_fit_cells"] == 45
    assert config["training"]["expected_curve_optimizer_steps"] == 5400
    assert config["on_pass"]["test_value_reads"] == 0
    assert all(config["prohibitions"].values())


def test_canonical_authorization_and_all_pins_match() -> None:
    paths = runner._paths(ROOT)
    config, authorized, pins = runner.authorize_entry(
        root=ROOT,
        data_dir=DATA_DIR,
        requested_config=paths["config"],
        requested_artifact=paths["artifact"],
    )
    assert authorized == paths
    assert config["experiment_id"] == "p1_station_layer_temporal_convolution_event_v2"
    assert len(pins) == len(config["immutable_inputs"])
    assert len(runner._verify_gen1_parts(ROOT, paths)) == 15


def test_authorization_rejects_config_copy_and_arbitrary_artifact(tmp_path: Path) -> None:
    paths = runner._paths(ROOT)
    copied = tmp_path / "copied.json"
    copied.write_bytes(paths["config"].read_bytes())
    with pytest.raises(PermissionError, match="non-canonical config"):
        runner.authorize_entry(
            root=ROOT,
            data_dir=DATA_DIR,
            requested_config=copied,
            requested_artifact=paths["artifact"],
        )
    with pytest.raises(PermissionError, match="non-canonical artifact"):
        runner.authorize_entry(
            root=ROOT,
            data_dir=DATA_DIR,
            requested_config=paths["config"],
            requested_artifact=tmp_path / "other",
        )


def test_direct_call_lock_and_cli_are_fail_closed() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'add_argument("--config"' not in source
    assert 'add_argument("--artifact"' not in source
    assert "_verify_lock(paths[\"lock\"], attempt)" in source
    assert "_acquire_lock(paths[\"lock\"], config)" in source
    assert '"test_value_reads": 0' in source
    assert '"candidate": None' in source
    assert '"uploads": 0' in source


def test_static_targets_are_absent_before_seal_and_attempt() -> None:
    paths = runner._paths(ROOT)
    assert not paths["artifact"].exists()
    assert not paths["lock"].exists()


def test_no_placeholder_survives_final_static_seal() -> None:
    content = (ROOT / runner.CANONICAL_CONFIG).read_text(encoding="utf-8")
    content += RUNNER_PATH.read_text(encoding="utf-8")
    assert "PENDING_" not in content
