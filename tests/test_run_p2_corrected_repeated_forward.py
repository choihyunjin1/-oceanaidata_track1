from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


def _runner():
    path = Path("scripts/run_p2_corrected_repeated_forward.py").resolve()
    spec = importlib.util.spec_from_file_location("p2_corrected_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(module):
    return json.loads(module.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_canonical_preflight_accepts_only_exact_file_mapping_and_output() -> None:
    module = _runner()
    config = _config(module)
    assert module._canonical_preflight(
        config, module.DEFAULT_CONFIG, module.DEFAULT_OUTPUT
    ) == config


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config["candidate"].__setitem__("seed", 7),
        lambda config: config["validation"]["folds"][0]["outer"].__setitem__(
            0, "2024-08-31T00:00:00+09:00"
        ),
        lambda config: config["masking"].__setitem__(
            "hidden_window_kst_half_open",
            ["2025-09-02T00:00:00+09:00", "2025-11-01T00:00:00+09:00"],
        ),
        lambda config: config["validation"]["bootstrap"].__setitem__("replicates", 2001),
        lambda config: config["output_contract"].__setitem__(
            "columns", ["station", "time", "layer", "temp"]
        ),
    ],
)
def test_direct_call_rejects_mutated_behavior_before_data_or_fit(mutation) -> None:
    module = _runner()
    config = copy.deepcopy(_config(module))
    mutation(config)
    lock_before = module.ATTEMPT_LOCK.exists()
    output_before = module.DEFAULT_OUTPUT.exists()
    with pytest.raises(ValueError, match="differs from the reloaded canonical"):
        module._run(
            config,
            module.DEFAULT_CONFIG,
            Path("does-not-exist"),
            module.DEFAULT_OUTPUT,
        )
    assert module.ATTEMPT_LOCK.exists() is lock_before
    assert module.DEFAULT_OUTPUT.exists() is output_before


def test_config_copy_and_noncanonical_output_are_rejected(tmp_path: Path) -> None:
    module = _runner()
    config = _config(module)
    copied = tmp_path / "copied.json"
    copied.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical v2 config path"):
        module._dry_run(config, copied, tmp_path, module.DEFAULT_OUTPUT)
    with pytest.raises(ValueError, match="canonical v2 output"):
        module._dry_run(config, module.DEFAULT_CONFIG, tmp_path, tmp_path / "output")


@pytest.mark.parametrize("escape_kind", ["traversal", "absolute"])
def test_candidate_path_escape_is_rejected(tmp_path: Path, escape_kind: str) -> None:
    module = _runner()
    config = copy.deepcopy(_config(module))
    if escape_kind == "traversal":
        candidate = "../submissions/p2/frozen.csv"
    else:
        candidate = str((tmp_path / "escaped.csv").resolve())
    config["output_contract"]["candidate_relative_path"] = candidate
    with pytest.raises(ValueError, match="escapes the canonical output"):
        module._planned_write_paths(config, module.DEFAULT_OUTPUT)


def test_model_path_escape_through_fold_name_is_rejected() -> None:
    module = _runner()
    config = copy.deepcopy(_config(module))
    config["validation"]["folds"][0]["name"] = "../../../submissions/p2/escaped"
    with pytest.raises(ValueError, match="escapes the canonical output"):
        module._planned_write_paths(config, module.DEFAULT_OUTPUT)


def test_exclusive_attempt_lock_refuses_second_run(tmp_path: Path) -> None:
    module = _runner()
    lock = tmp_path / "control" / "attempt.lock"
    first = module._acquire_attempt_lock(lock)
    assert first["sha256"]
    with pytest.raises(FileExistsError, match="already consumed"):
        module._acquire_attempt_lock(lock)


def test_prewrite_guard_refuses_existing_output(tmp_path: Path) -> None:
    module = _runner()
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        module._prewrite_guard(_config(module), tmp_path, output)
