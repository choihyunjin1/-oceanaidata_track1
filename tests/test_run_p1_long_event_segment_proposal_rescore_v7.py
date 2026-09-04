from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_long_event_segment_proposal_rescore_v7.py"
MODULE = ROOT / "src/p1_qc/long_event_segment_proposal_rescore_execution_v7.py"
TEMPLATE = (
    ROOT
    / "configs/experiments/p1_long_event_segment_proposal_rescore_v7_execution_authorization_template.json"
)
AMENDMENT = (
    ROOT
    / "configs/experiments/p1_long_event_segment_proposal_rescore_v7_full_runtime_replay_firewall_amendment.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def r7() -> ModuleType:
    spec = importlib.util.spec_from_file_location("p1_segment_r7_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_authority_and_science_counts(r7: ModuleType) -> None:
    assert sha256(AMENDMENT) == (
        "71563c954a5c529044d82c63af0e44ddf313dcc55b787c784afd153fc14434ff"
    )
    assert sha256(MODULE) == (
        "9f532581bfda500f4bdf1f923ad602c16965ccb39a4ab6157c3f88c3af88c061"
    )
    assert r7.MAXIMUM_LIFETIME_PHYSICAL_FITS == 72
    assert r7.MAXIMUM_SCIENTIFIC_MATERIALIZATIONS == 21
    assert r7.HARD_WALL_SECONDS == 21600


def test_bootstrap_top_level_imports_are_stdlib_only(r7: ModuleType) -> None:
    imported = r7._verify_top_level_stdlib_only(RUNNER)
    forbidden = {
        "numpy",
        "pandas",
        "lightgbm",
        "sklearn",
        "scipy",
        "pyarrow",
        "joblib",
        "p1_qc",
    }
    assert forbidden.isdisjoint(imported)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":{"b":1,"b":2}}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-Infinity}',
    ],
)
def test_wire_parser_rejects_duplicate_keys_and_nonfinite(
    r7: ModuleType, raw: bytes
) -> None:
    with pytest.raises(r7.WireFormatError):
        r7._json_from_bytes(raw, label="adversarial")


def test_authorization_schema_rejects_template_and_extra_key(r7: ModuleType) -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    with pytest.raises(r7.AuthorizationError):
        r7._require_authorization_schema(template)
    template["rogue"] = 1
    with pytest.raises(r7.AuthorizationError):
        r7._require_authorization_schema(template)


def test_template_is_zero_operation_not_authorized(r7: ModuleType) -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert template["authorized"] is False
    assert template["status"] == "NOT_AUTHORIZED_PENDING_INDEPENDENT_QA"
    assert template["runner_sha256"] == sha256(RUNNER)
    assert template["runner_normalized_sha256"] == sha256(RUNNER)
    assert set(template["zero_prior_state"].values()) == {0}
    assert template["operation_authorization"] == {
        "single_attempt": True,
        "maximum_lifetime_physical_fits": 72,
        "maximum_scientific_materializations": 21,
        "outer_scores": 1,
        "candidate_files": 0,
        "uploads": 0,
    }


def test_full_runtime_inventory_exact_closure(r7: ModuleType) -> None:
    inventory = r7._full_runtime_inventory()
    assert len(inventory["files"]) == 10745
    assert len(inventory["excluded_nonimport_cli"]) == 2
    assert inventory["distribution_versions"]["tzdata"] == "2026.3"
    assert sum(
        1
        for record in inventory["files"].values()
        if record["owners"] == ["tzdata"]
    ) == 656
    assert sum(
        relative.endswith(".pyc")
        and record["owners"] != ["tzdata"]
        for relative, record in inventory["files"].items()
    ) == 3849
    assert sum(relative.endswith(".pyc") for relative in inventory["files"]) == 3871
    assert any(relative.endswith(".pyd") for relative in inventory["files"])
    assert any(relative.endswith(".dll") for relative in inventory["files"])
    assert any(name.lower().endswith("python312.dll") for name in inventory["host_files"])


def test_numpy_cli_exclusions_are_exact(r7: ModuleType) -> None:
    inventory = r7._full_runtime_inventory()
    observed = {
        item["name"].lower(): (item["bytes"], item["sha256"])
        for item in inventory["excluded_nonimport_cli"]
    }
    assert observed == {
        name: (record["bytes"], record["sha256"])
        for name, record in r7.EXPECTED_NUMPY_CLI.items()
    }


def test_atomic_create_is_create_only(r7: ModuleType, tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    r7._atomic_create_bytes(target, b"first")
    with pytest.raises(FileExistsError):
        r7._atomic_create_bytes(target, b"second")
    assert target.read_bytes() == b"first"


def test_exact_tree_rejects_undeclared_insertion(r7: ModuleType, tmp_path: Path) -> None:
    declared = tmp_path / "declared.py"
    declared.write_bytes(b"x=1\n")
    expected = {
        "declared.py": {"bytes": declared.stat().st_size, "sha256": sha256(declared)}
    }
    r7._verify_exact_tree(tmp_path, expected, label="probe")
    (tmp_path / "rogue.py").write_bytes(b"marker=True\n")
    with pytest.raises(RuntimeError, match="exact membership"):
        r7._verify_exact_tree(tmp_path, expected, label="probe")


def test_held_handle_reads_same_bytes_and_denies_mutation(
    r7: ModuleType, tmp_path: Path
) -> None:
    target = tmp_path / "held.bin"
    target.write_bytes(b"held")
    with r7.HeldReadDenyMutation([target]) as guard:
        assert guard.read_bytes(target) == b"held"
        record = guard.records[str(target.resolve())]
        assert record["sha256"] == sha256(target)
        if os.name == "nt":
            with pytest.raises(PermissionError):
                target.write_bytes(b"mutated")
            with pytest.raises(PermissionError):
                target.unlink()
    target.write_bytes(b"released")
    assert target.read_bytes() == b"released"


def test_transitive_runtime_and_native_handles_all_deny_write_delete(
    r7: ModuleType, tmp_path: Path
) -> None:
    paths = [
        tmp_path / "transitive_project.py",
        tmp_path / "runtime_dependency.py",
        tmp_path / "native_payload.pyd",
    ]
    for ordinal, path in enumerate(paths):
        path.write_bytes(f"sealed-{ordinal}".encode())
    with r7.HeldReadDenyMutation(paths) as guard:
        assert len(guard.file_ids()) == 3
        if os.name == "nt":
            for path in paths:
                with pytest.raises(PermissionError):
                    path.write_bytes(b"changed")
                with pytest.raises(PermissionError):
                    path.rename(path.with_suffix(path.suffix + ".moved"))
                with pytest.raises(PermissionError):
                    path.unlink()
    for path in paths:
        path.unlink()


def test_source_identity_is_bound_to_held_handle(r7: ModuleType, tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    identity = r7._source_identity(source)
    expected = {
        "bytes": 6,
        "sha256": sha256(source),
        "source_identity": identity,
    }
    with r7.HeldReadDenyMutation([source]) as guard:
        guard.assert_record(source, expected, require_source_identity=True)
        forged = dict(expected)
        forged["source_identity"] = dict(identity, inode=identity["inode"] + 1)
        with pytest.raises(RuntimeError, match="identity"):
            guard.assert_record(source, forged, require_source_identity=True)


def test_exact_tree_rejects_link_when_supported(r7: ModuleType, tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    target.write_bytes(b"x=1\n")
    link = tmp_path / "linked.py"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Windows symlink creation is unavailable")
    expected = {
        "linked.py": {"bytes": target.stat().st_size, "sha256": sha256(target)},
        "target.py": {"bytes": target.stat().st_size, "sha256": sha256(target)},
    }
    with pytest.raises(RuntimeError, match="link/reparse"):
        r7._verify_exact_tree(tmp_path, expected, label="link-probe")


def test_manifest_is_explicitly_transport_only(r7: ModuleType, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    runtime = tmp_path / "runtime"
    snapshot.mkdir()
    runtime.mkdir()
    manifest = r7._manifest_value(
        snapshot_root=snapshot,
        runtime_root=runtime,
        snapshot_inventory={"x": {"bytes": 1, "sha256": "0" * 64}},
        runtime_inventory_binding={"bytes": 2, "sha256": "1" * 64},
        authorization_sha256="2" * 64,
        seal_sha256="3" * 64,
        qa_sha256="4" * 64,
        created_epoch_ns=1,
        deadline_epoch_ns=1 + 21600 * 1_000_000_000,
        parent_identity={
            "pid": 1,
            "creation_filetime": 2,
            "image_path": "python.exe",
            "image_sha256": "5" * 64,
        },
        launch_nonce="6" * 64,
    )
    assert manifest["manifest_role"] == "TRANSPORT_RECEIPT_NEVER_AUTHORITY"
    assert r7._require_manifest_schema(manifest) is manifest
    forged = dict(manifest)
    forged["runtime_inventory_binding"] = {"bytes": 2, "sha256": "7" * 64}
    assert r7._require_manifest_schema(forged) is forged
    source = RUNNER.read_text(encoding="utf-8")
    assert 'runtime = seal_value["full_runtime_inventory"]' in source
    assert 'manifest["runtime_inventory_binding"] != _canonical_record(runtime)' in source


def test_manifest_and_claim_schema_reject_substitution(r7: ModuleType) -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for literal in (
        "canonical_project_root",
        "canonical_artifact_relative_path",
        "single_attempt_namespace",
        "launch_nonce",
        "deadline_epoch_ns",
        "parent_identity",
        "runtime_inventory_binding",
        "snapshot_inventory_binding",
    ):
        assert literal in source
    assert "ANY_EXISTING_CLAIM_PERMANENTLY_BLOCKS_REPLAY" in source
    assert "worker_start.lease" in source


def test_claim_publication_rejects_partial_or_complete_replay(
    r7: ModuleType, tmp_path: Path
) -> None:
    claim = tmp_path / "canonical_launch.claim"
    r7._atomic_create_bytes(claim, b"partial")
    with pytest.raises(FileExistsError):
        r7._atomic_create_json(claim, {"status": "forged-second-claim"})
    assert claim.read_bytes() == b"partial"


def test_runtime_tree_and_manifest_never_select_scientific_ceilings() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "MAXIMUM_LIFETIME_PHYSICAL_FITS = 72" in source
    assert "MAXIMUM_SCIENTIFIC_MATERIALIZATIONS = 21" in source
    assert 'manifest["maximum_lifetime_physical_fits"]' not in source
    assert 'manifest["maximum_scientific_materializations"]' not in source


def test_worker_command_isolated_and_exact(r7: ModuleType, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    runner = snapshot / "scripts/run_p1_long_event_segment_proposal_rescore_v7.py"
    runner.parent.mkdir(parents=True)
    runner.write_bytes(RUNNER.read_bytes())
    manifest = {
        "launch_nonce": "a" * 64,
        "deadline_epoch_ns": 21600 * 1_000_000_000,
        "parent_identity": {"pid": 123},
        "authorization_sha256": "b" * 64,
        "seal_sha256": "c" * 64,
        "qa_sha256": "d" * 64,
        "snapshot_inventory_binding": {"sha256": "e" * 64},
        "runtime_inventory_binding": {"sha256": "f" * 64},
        "canonical_project_root": str(ROOT),
        "canonical_artifact_relative_path": r7.CANONICAL_ARTIFACT_RELATIVE,
        "single_attempt_namespace": r7.SINGLE_ATTEMPT_NAMESPACE,
    }
    command = r7._worker_command(
        snapshot,
        tmp_path / "manifest.json",
        "1" * 64,
        "2" * 64,
        manifest,
    )
    assert command[1:4] == ["-I", "-S", "-B"]
    assert command.count("--worker") == 1
    assert "--launch-claim-sha256" in command
    assert "--deadline-epoch-ns" in command


def test_full_source_has_four_runtime_checkpoints_and_no_sentinel_fill() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "private snapshot preimport" in source
    assert "isolated runtime preimport" in source
    assert '"POSTIMPORT"' in source
    assert '"PRECLAIM"' in source
    assert 'checkpoint("PREFIT")' in source
    module_source = MODULE.read_text(encoding="utf-8")
    assert "probability1" not in module_source
    assert "pred0" not in module_source


def test_execution_wrapper_only_delegates_frozen_science() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_authorized_screen"
        for node in calls
    )
    assert not any(
        isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "predict"}
        for node in calls
    )


def test_execute_without_external_digest_fails_before_claim(
    r7: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(r7.AUTHORIZATION_ENV_VAR, raising=False)
    existed = r7.LAUNCH_CLAIM_PATH.exists()
    with pytest.raises(r7.AuthorizationError):
        r7.execute_parent()
    assert r7.LAUNCH_CLAIM_PATH.exists() is existed


def test_zero_operation_namespace_and_actual_authorization_absent(r7: ModuleType) -> None:
    assert not r7.AUTHORIZATION_PATH.exists()
    assert not r7.LAUNCH_CLAIM_PATH.exists()
    assert not (r7.CANONICAL_ARTIFACT_DIR / "execution.lock").exists()
    assert not (r7.CANONICAL_ARTIFACT_DIR / "result.json").exists()
