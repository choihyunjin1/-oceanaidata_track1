from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path

import joblib
import pytest


def _runner():
    path = Path("scripts/run_p2_conservative_stack_improvement_v1_resume.py").resolve()
    spec = importlib.util.spec_from_file_location("p2_stack_resume_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(module):
    return json.loads(module.RESUME_CONFIG.read_text(encoding="utf-8"))


def _simulate_windows_text_write(payload: bytes) -> bytes:
    return payload.replace(b"\n", b"\r\n")


def test_canonical_resume_config_is_same_attempt_zero_refit() -> None:
    module = _runner()
    config = _config(module)
    assert module._canonical_preflight(config, module.RESUME_CONFIG) == config
    assert config["same_original_attempt"] == 1
    assert config["new_generation_or_attempt"] is False
    assert config["fit_accounting"]["correction_model_refits"] == 0
    assert all(value == 0 for value in config["behavior_changes"].values())


def test_direct_call_rejects_winner_or_weight_change() -> None:
    module = _runner()
    config = copy.deepcopy(_config(module))
    config["winner_exposed_before_correction"]["candidate_weight"] = 0.5
    with pytest.raises(ValueError, match="differs from canonical"):
        module._canonical_preflight(config, module.RESUME_CONFIG)


def test_text_expansion_repair_restores_arbitrary_binary() -> None:
    module = _runner()
    original = b"header\x00line\nexisting\r\nfooter\xff"
    damaged = _simulate_windows_text_write(original)
    assert module.repair_windows_text_expansion(damaged) == original


def test_text_expansion_repair_restores_joblib_payload() -> None:
    module = _runner()
    buffer = io.BytesIO()
    joblib.dump({"winner": "STACK_W0625", "weight": 0.625}, buffer)
    # A small pickle may contain no LF byte, so append one harmless trailing LF
    # to exercise the Windows text-mode transformation deterministically.
    original = buffer.getvalue() + b"\n"
    damaged = _simulate_windows_text_write(original)
    repaired = module.repair_windows_text_expansion(damaged)
    assert joblib.load(io.BytesIO(repaired)) == {
        "winner": "STACK_W0625",
        "weight": 0.625,
    }


def test_exclusive_binary_preserves_bytes_and_refuses_overwrite(tmp_path: Path) -> None:
    module = _runner()
    path = tmp_path / "binary.joblib"
    payload = b"\x80\x04\x00\n\xff"
    module._exclusive_binary(path, payload)
    assert path.read_bytes() == payload
    with pytest.raises(FileExistsError, match="already exists"):
        module._exclusive_binary(path, payload)


def test_all_repaired_targets_are_append_only_new_paths() -> None:
    module = _runner()
    config = _config(module)
    paths = module._planned_paths(config)
    repaired_root = module._path_from_logical(config["correction_output"]["repaired_root"])
    for role, path in paths.items():
        if role.startswith("repaired:"):
            path.relative_to(repaired_root)
            assert not str(path).startswith(str(module.OUTPUT / "models"))


def test_initial_affected_binary_pins_are_exact() -> None:
    module = _runner()
    records = module._verify_initial_pins(_config(module))
    assert len(records) == 15
    assert sum(key.startswith("affected:") for key in records) == 10
