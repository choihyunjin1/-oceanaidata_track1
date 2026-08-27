from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from p2_restore.architecture_matched_prefix_refit import (
    CONFIG_RELATIVE,
    CONFIG_SHA256,
    ArchitectureContractError,
    contained_path,
    exclusive_bytes,
    inspect_schema_and_keys,
    load_canonical_config,
    sha256_file,
    validate_config,
    verify_stage_a_seal,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _key_digest(rows: list[tuple[str, str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(list(row), ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def test_canonical_config_sha_and_deep_equality_guard() -> None:
    root = _root()
    config = load_canonical_config(root)
    assert sha256_file(root / CONFIG_RELATIVE) == CONFIG_SHA256
    assert config["comparison_mode"] == "ARCHITECTURE_MATCHED_TIME_SAFE_BASELINE"
    assert config["exact_official_incumbent_comparison"] is False
    load_canonical_config(root, supplied_config=deepcopy(config))

    changed = deepcopy(config)
    changed["training_recipe"]["embargo_days"] = 6
    with pytest.raises(ArchitectureContractError, match="deep equality"):
        load_canonical_config(root, supplied_config=changed)


def test_config_copy_is_rejected_even_when_bytes_match(tmp_path: Path) -> None:
    root = _root()
    copied = tmp_path / "copied.json"
    copied.write_bytes((root / CONFIG_RELATIVE).read_bytes())
    with pytest.raises(ArchitectureContractError, match="canonical config path"):
        load_canonical_config(root, copied)


def test_p2_mode_and_non_exact_labels_fail_closed() -> None:
    config = load_canonical_config(_root())
    for field, value in (
        ("problem", "P1"),
        ("comparison_mode", "EXACT_OFFICIAL_PREFIX_REFIT"),
        ("exact_official_incumbent_comparison", True),
    ):
        changed = deepcopy(config)
        changed[field] = value
        with pytest.raises(ArchitectureContractError):
            validate_config(changed)


def test_output_containment_rejects_escape(tmp_path: Path) -> None:
    output = tmp_path / "output"
    assert contained_path(output, "seal.json") == (output / "seal.json").resolve()
    with pytest.raises(ArchitectureContractError, match="unsafe"):
        contained_path(output, "../escape.json")
    with pytest.raises(ArchitectureContractError, match="unsafe"):
        contained_path(output, tmp_path / "absolute.json")


def test_o_excl_write_and_rerun_are_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "attempt.lock"
    exclusive_bytes(path, b"first")
    assert path.read_bytes() == b"first"
    with pytest.raises(FileExistsError):
        exclusive_bytes(path, b"second")
    assert path.read_bytes() == b"first"


def test_stage_a_seal_deeply_binds_all_five_prefix_oof_files(tmp_path: Path) -> None:
    config = deepcopy(load_canonical_config(_root()))
    config["canonical_paths"]["stage_a_output"] = "reference"
    config["canonical_paths"]["stage_a_control"] = "reference_control"
    config["canonical_paths"]["stage_a_authorization"] = "reference_control/authorization.json"
    config["canonical_paths"]["stage_a_attempt_lock"] = "reference_control/attempt.lock"
    output = tmp_path / "reference"
    output.mkdir()
    artifacts = config["stage_a_reference_contract"]["artifacts"]
    (output / artifacts["deployed_graph_manifest"]).write_text(
        json.dumps(config["deployed_inference_graph"]), encoding="utf-8"
    )
    (output / artifacts["training_recipe"]).write_text(
        json.dumps(config["training_recipe"]), encoding="utf-8"
    )
    (output / artifacts["reference_curve_metrics"]).write_text("{}", encoding="utf-8")
    pins = {}
    names = {
        "0.4": "reference_oof_040.parquet",
        "0.55": "reference_oof_055.parquet",
        "0.7": "reference_oof_070.parquet",
        "0.85": "reference_oof_085.parquet",
        "1.0": artifacts["reference_oof_100"],
    }
    for fraction, name in names.items():
        path = output / name
        path.write_bytes(f"target-free-{fraction}".encode())
        pins[fraction] = {"path": name, "sha256": sha256_file(path)}
    manifest = {
        "schema_version": "p2_architecture_matched_reference.manifest.v1",
        "reference_oof_by_fraction": pins,
    }
    (output / artifacts["manifest"]).write_text(json.dumps(manifest), encoding="utf-8")
    verified_pins = {
        fraction: {"path": pin["path"], "sha256": pin["sha256"]} for fraction, pin in pins.items()
    }
    seal = {
        "schema_version": "p2_architecture_matched_reference.seal.v1",
        "comparison_mode": "ARCHITECTURE_MATCHED_TIME_SAFE_BASELINE",
        "exact_official_incumbent_comparison": False,
        "complete": True,
        "all_five_prefixes_sealed": True,
        "challenger_fit_or_score_count_before_seal": 0,
        "reference_oof_by_fraction": verified_pins,
        "binding": {
            "stage_a_config_sha256": CONFIG_SHA256,
            "deployed_graph_manifest_sha256": sha256_file(
                output / artifacts["deployed_graph_manifest"]
            ),
            "training_recipe_sha256": sha256_file(output / artifacts["training_recipe"]),
            "reference_oof_100_sha256": pins["1.0"]["sha256"],
        },
    }
    (output / artifacts["seal"]).write_text(json.dumps(seal), encoding="utf-8")

    result = verify_stage_a_seal(tmp_path, config)
    assert set(result["reference_oof_by_fraction"]) == set(names)
    assert result["verified_before_challenger_import_fit_score"] is True

    (output / names["0.7"]).write_bytes(b"tampered")
    with pytest.raises(ArchitectureContractError, match="prefix OOF SHA"):
        verify_stage_a_seal(tmp_path, config)


def test_schema_key_check_never_reads_target_columns(tmp_path: Path) -> None:
    (tmp_path / "observations.csv").write_text(
        "station,year,layer,time,temp,psal,depth,nominal_depth\n",
        encoding="utf-8",
    )
    (tmp_path / "test_index.csv").write_text(
        "station,layer,time,nominal_depth\nS,2,2025-01-01T00:00:00Z,5\n",
        encoding="utf-8",
    )
    (tmp_path / "sample_submission.csv").write_text(
        "station,layer,time,temp\nS,2,2025-01-01T00:00:00Z,NOT_A_NUMBER\n",
        encoding="utf-8",
    )
    config = deepcopy(load_canonical_config(_root()))
    config["data_contract"]["canonical_test_rows"] = 1
    config["data_contract"]["canonical_station_layer_time_key_sha256"] = _key_digest(
        [("S", "2", "2025-01-01T00:00:00Z")]
    )
    result = inspect_schema_and_keys(tmp_path, config)
    assert result["target_columns_read"] == []
    assert result["observations_header_only"] is True
    assert result["test_sample_key_deep_equal"] is True


def test_stage_b_verifies_reference_before_lock_or_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_p2_meaningful_learning_curve_generation_v2 as runner

    order: list[str] = []
    config = deepcopy(load_canonical_config(_root()))
    authorization = tmp_path / config["canonical_paths"]["stage_b_authorization"]
    authorization.parent.mkdir(parents=True)
    authorization.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(runner, "load_canonical_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        runner,
        "verify_stage_a_seal",
        lambda *_args, **_kwargs: order.append("seal") or {"seal_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        runner,
        "static_preflight",
        lambda *_args, **_kwargs: order.append("preflight") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        runner,
        "verify_execution_authorization",
        lambda *_args, **_kwargs: order.append("authorization") or {"ok": True},
    )
    monkeypatch.setattr(runner, "sha256_file", lambda _path: "b" * 64)
    monkeypatch.setattr(
        runner,
        "consume_attempt_lock",
        lambda *_args, **_kwargs: order.append("lock") or (tmp_path / "lock"),
    )
    engine = SimpleNamespace(execute_stage_b=lambda **_kwargs: order.append("fit") or {"ok": True})
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda _name: order.append("import") or engine,
    )

    result = runner.run_authorized(root=tmp_path, data_dir=tmp_path)
    assert order == ["seal", "preflight", "authorization", "lock", "import", "fit"]
    assert result["exact_official_incumbent_comparison"] is False
    assert result["local_pass_can_promote"] is False


def test_stage_a_static_config_refuses_before_authorization_or_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_p2_architecture_matched_reference_v1 as runner

    order: list[str] = []
    config = deepcopy(load_canonical_config(_root()))
    assert config["execution_policy"]["static_check_only_now"] is True
    monkeypatch.setattr(runner, "load_canonical_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        runner,
        "static_preflight",
        lambda *_args, **_kwargs: order.append("preflight") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        runner,
        "verify_execution_authorization",
        lambda *_args, **_kwargs: order.append("authorization"),
    )
    monkeypatch.setattr(
        runner,
        "consume_attempt_lock",
        lambda *_args, **_kwargs: order.append("lock"),
    )
    with pytest.raises(RuntimeError, match="static-check-only"):
        runner.run_authorized(root=tmp_path, data_dir=tmp_path)
    assert order == ["preflight"]
    assert not (tmp_path / config["canonical_paths"]["stage_a_attempt_lock"]).exists()
