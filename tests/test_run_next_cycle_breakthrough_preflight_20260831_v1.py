from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_next_cycle_breakthrough_preflight_20260831_v1 import (
    DEFAULT_CONFIG,
    ROOT,
    ContractError,
    run,
)


def test_real_immutable_inputs_produce_fail_fast_no_go(tmp_path: Path) -> None:
    result = run(DEFAULT_CONFIG, tmp_path)

    assert result["decision"] == "NO_NEW_ONE_SHOT_AUTHORIZED_UNTIL_NEW_INFORMATION"
    assert {item["action"] for item in result["decisions"].values()} == {
        "NO_NEW_FIT",
        "NO_NEW_FIT_OR_BIN_EXPANSION",
        "NO_NEW_KMA_OR_ERA5_MICROTUNE",
    }
    assert result["operations"]["model_fits"] == 0
    assert result["operations"]["official_test_sample_submission_rows_read"] == 0
    assert (tmp_path / "result.json").is_file()
    qa = json.loads((tmp_path / "independent-qa.json").read_text(encoding="utf-8"))
    assert qa["status"] == "PASS"
    assert all(qa["checks"].values())


def test_input_hash_drift_is_rejected(tmp_path: Path) -> None:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["inputs"]["p1_validation_audit"]["sha256"] = "0" * 64
    altered = tmp_path / "altered-config.json"
    altered.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ContractError, match="immutable input drift"):
        run(altered, tmp_path / "out")


def test_config_paths_resolve_inside_repository() -> None:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    for spec in config["inputs"].values():
        path = (ROOT / spec["path"]).resolve()
        assert path.is_relative_to(ROOT)
        assert path.is_file()
