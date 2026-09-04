from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_p1_mstcn_lower_bound_veto_probe_package_20260829_v1.py"
SPEC = importlib.util.spec_from_file_location("veto_package", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_environment_contract_has_no_personal_absolute_paths() -> None:
    assert set(MODULE.ENVIRONMENT_PATHS.values()) == {
        "P1_DATA_DIR",
        "P1_CURRENT_ROUTER",
        "P1_CHAMPION_SUBMISSION",
        "P1_SUBMISSION_ROOT",
    }
    source = SCRIPT.read_text(encoding="utf-8")
    assert "C:\\Users" not in source


def test_config_is_one_candidate_and_no_upload() -> None:
    config = MODULE.json.loads(MODULE.CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["candidate_creation_authorized"] is True
    assert config["upload_authorized"] is False
    assert config["expected"]["accepted_e150_rows"] == 8
    assert config["expected"]["positive_rows"] == 6071
