from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from p2_restore import joint_hydrographic_multitask_layer4_contract_r3 as r3
from p2_restore import (
    joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v1 as verifier,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return verifier.load_config(ROOT)


def _audit_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    config = _config()
    output = ROOT / config["canonical_paths"]["r3_output"]
    receipt = json.loads((output / "training_receipt.json").read_text(encoding="utf-8"))
    committed: dict[str, object] = {}
    for fold in config["fold_audit_compatibility_contract"]["canonical_fold_order"]:
        payload = json.loads(
            (output / "folds" / fold / "fold_commitment.json").read_text(encoding="utf-8")
        )
        committed[fold] = payload["blind_input_audit"]
    return (
        receipt["fold_blind_input_audits"],
        committed,
        config["fold_audit_compatibility_contract"],
    )


def _frozen_snapshot() -> dict[str, object]:
    config = _config()
    return {
        "r3_implementation": verifier.verify_pin_map(
            ROOT, config["r3_implementation_pins"], label="r3 implementation"
        ),
        "r3_controls": verifier.verify_pin_map(ROOT, config["r3_control_pins"], label="r3 control"),
        "r3_core": verifier.verify_pin_map(ROOT, config["r3_core_artifact_pins"], label="r3 core"),
        "control_inventory": verifier.frozen_inventory(
            ROOT / config["canonical_paths"]["r3_control"]
        ),
        "output_inventory": verifier.frozen_inventory(
            ROOT / config["canonical_paths"]["r3_output"]
        ),
        "compatibility_control_exists": (
            ROOT / config["canonical_paths"]["compatibility_control"]
        ).exists(),
    }


def test_config_identity_firewall_and_implementation_paths_are_exact() -> None:
    config = _config()
    assert config["identity"] == verifier.IDENTITY
    assert config["verifier_only"] is config["check_only_default"] is True
    for key in (
        "r3_mutation_allowed",
        "r3_rerun_or_resume_allowed",
        "execution_authorization_or_lock_allowed",
        "fit_prediction_truth_decode_or_scoring_allowed",
        "official_promotion_allowed",
        "candidate_or_test_prediction_allowed",
        "upload_allowed",
    ):
        assert config[key] is False
    assert config["implementation_roles"] == verifier.IMPLEMENTATION_ROLES


def test_exact_r3_pins_inventories_allowlist_and_v9_anchor_are_frozen() -> None:
    config = _config()
    snapshot = _frozen_snapshot()
    assert snapshot["control_inventory"] == config["r3_control_inventory"]
    assert snapshot["output_inventory"] == config["r3_output_inventory"]
    r3_config = r3.load_canonical_config(ROOT)
    output = ROOT / config["canonical_paths"]["r3_output"]
    assert {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == (
        r3.expected_output_files(r3_config)
    )
    assert {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_dir()} == (
        r3.expected_output_directories(r3_config)
    )
    assert config["v9_anchor"] == {
        "path": "artifacts/meaningful_score_goal_v9/registry.jsonl",
        "sha256": "232b6ed3133de11ee05150ec439efe05baa315bbb64ea0f319ffcbddd421b965",
        "bytes": 15812,
        "record_count": 3,
        "sequences": [3, 4, 5],
        "head_sequence": 5,
        "head_event_sha256": ("1b3e01be70c6f8ed2df04038deac3b3642804f70f9f17a238826c64d68090317"),
        "uploads": 0,
    }


def test_original_verifier_has_only_the_pinned_false_negative() -> None:
    config = _config()
    r3_config = r3.load_canonical_config(ROOT)
    with pytest.raises(r3.Layer4ContractError) as caught:
        r3.verify_seal(ROOT, r3_config)
    assert str(caught.value) == config["original_verifier_expected_failure"]["message"]


def test_frozen_r3_guard_is_authenticated_before_dynamic_execution() -> None:
    config = _config()
    guard = config["r3_implementation_pins"]["GUARD"]
    assert verifier.R3_GUARD_SHA256 == guard["sha256"]
    assert verifier.R3_GUARD_BYTES == guard["bytes"]
    source = (ROOT / verifier.IMPLEMENTATION_ROLES["HELPER"]).read_text(encoding="utf-8")
    assert source.index("guard_bytes = path.read_bytes()") < source.index(
        "spec.loader.exec_module(module)"
    )


def test_static_compatibility_passes_without_writes_or_numerical_imports() -> None:
    before = _frozen_snapshot()
    result = verifier.verify_static_compatibility(ROOT)
    after = _frozen_snapshot()
    assert before == after
    assert result["status"] == "PASS_R3_COMPATIBILITY_VERIFIER_ONLY_RESEARCH_RESULT_LOCAL_FAIL"
    assert (
        result["corrected_fold_audit_verification"]["exact_fold_commitment_audit_equality"] is True
    )
    assert result["result_checks"] == {
        "status": True,
        "local": True,
        "passed": True,
        "promotion": True,
        "candidate": True,
        "test": True,
        "upload": True,
        "finite_metrics": True,
    }
    assert result["new_numerical_modules"] == []
    summary_payload = dict(result)
    claimed_summary = summary_payload.pop("summary_sha256")
    assert hashlib.sha256(r3.canonical_json_bytes(summary_payload)).hexdigest() == claimed_summary
    for key in (
        "files_written",
        "execution_authorizations_created",
        "attempt_locks_created",
        "model_fits",
        "predictions",
        "truth_scalar_decodes",
        "scores_computed",
        "candidate_predictions",
        "test_predictions",
        "uploads",
    ):
        assert result[key] == 0


def test_mapping_order_is_ignored_but_canonical_fold_semantics_are_enforced() -> None:
    audits, committed, contract = _audit_inputs()
    reversed_audits = dict(reversed(list(audits.items())))
    result = verifier._verify_fold_audit_compatibility(  # noqa: SLF001
        reversed_audits, committed, contract
    )
    assert [row["verified_prior_fold_commitments"] for row in result["verified"]] == [0, 1, 2]
    assert [row["fold"] for row in result["verified"]] == contract["canonical_fold_order"]


@pytest.mark.parametrize("kind", ["missing", "extra"])
def test_forged_fold_mapping_key_set_is_rejected(kind: str) -> None:
    audits, committed, contract = _audit_inputs()
    forged = copy.deepcopy(audits)
    if kind == "missing":
        forged.pop("outer_2025_may_jun")
    else:
        forged["outer_forged"] = copy.deepcopy(forged["outer_2024_sep_oct"])
    with pytest.raises(verifier.CompatibilityVerifierError, match="key set"):
        verifier._verify_fold_audit_compatibility(forged, committed, contract)  # noqa: SLF001


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fold", "outer_2025_jul_aug", "differs from commitment"),
        ("verified_prior_fold_commitments", 2, "differs from commitment"),
        (
            "active_fold_target_temp_psal_scalar_fields_decoded_or_converted",
            1,
            "differs from commitment",
        ),
        ("withheld_target_temp_psal_scalar_fields_decoded_or_converted", True, "differs"),
        ("anomaly_or_hidden_target_proxy_reads", 1, "differs"),
    ],
)
def test_forged_fold_audit_content_is_rejected(field: str, value: object, message: str) -> None:
    audits, committed, contract = _audit_inputs()
    forged = copy.deepcopy(audits)
    forged["outer_2025_may_jun"][field] = value
    with pytest.raises(verifier.CompatibilityVerifierError, match=message):
        verifier._verify_fold_audit_compatibility(forged, committed, contract)  # noqa: SLF001


def test_extra_fold_audit_content_is_rejected() -> None:
    audits, committed, contract = _audit_inputs()
    forged = copy.deepcopy(audits)
    forged["outer_2024_sep_oct"]["forged"] = 0
    with pytest.raises(verifier.CompatibilityVerifierError, match="differs from commitment"):
        verifier._verify_fold_audit_compatibility(forged, committed, contract)  # noqa: SLF001


def test_forged_commitment_audit_is_rejected() -> None:
    audits, committed, contract = _audit_inputs()
    forged = copy.deepcopy(committed)
    forged["outer_2025_jul_aug"]["verified_prior_fold_commitments"] = 1
    with pytest.raises(verifier.CompatibilityVerifierError, match="differs from commitment"):
        verifier._verify_fold_audit_compatibility(audits, forged, contract)  # noqa: SLF001


def test_inventory_extra_is_detected_before_original_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = verifier.frozen_inventory

    def forged_inventory(path: Path) -> dict[str, object]:
        observed = original(path)
        if path.name == "p2_joint_hydrographic_multitask_layer4_execution_r3":
            observed["files"] += 1
        return observed

    monkeypatch.setattr(verifier, "frozen_inventory", forged_inventory)
    with pytest.raises(verifier.CompatibilityVerifierError, match="output inventory changed"):
        verifier.verify_static_compatibility(ROOT)


def test_strict_pinned_reparse_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_bytes(b'{"a":1,"a":2}\n')
    pin = {
        "path": "duplicate.json",
        "sha256": r3.sha256_file(path),
        "bytes": path.stat().st_size,
    }
    with pytest.raises(verifier.CompatibilityVerifierError, match="duplicate JSON key"):
        verifier._strict_pinned_json(tmp_path, pin, label="duplicate test")  # noqa: SLF001


def test_strict_pinned_reparse_detects_mid_parse_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "racing.json"
    path.write_bytes(b'{"value":1}\n')
    pin = {
        "path": "racing.json",
        "sha256": r3.sha256_file(path),
        "bytes": path.stat().st_size,
    }
    original = Path.read_bytes

    def mutate_after_read(candidate: Path) -> bytes:
        value = original(candidate)
        if candidate == path:
            candidate.write_bytes(b'{"value":2}\n')
        return value

    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)
    with pytest.raises(verifier.CompatibilityVerifierError, match="changed during parse"):
        verifier._strict_pinned_json(tmp_path, pin, label="race test")  # noqa: SLF001


def test_frozen_inventory_rejects_linklike_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ordinary = tmp_path / "ordinary.bin"
    ordinary.write_bytes(b"x")
    original = Path.is_symlink

    def classify(path: Path) -> bool:
        return path == ordinary or original(path)

    monkeypatch.setattr(Path, "is_symlink", classify)
    with pytest.raises(verifier.CompatibilityVerifierError, match="link-like"):
        verifier.frozen_inventory(tmp_path)
    with pytest.raises(verifier.CompatibilityVerifierError, match="link-like"):
        verifier._path(tmp_path, "ordinary.bin")  # noqa: SLF001


def test_real_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    link = tmp_path / "link.bin"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("local Windows policy does not permit test symlinks")
    with pytest.raises(verifier.CompatibilityVerifierError, match="link-like"):
        verifier.frozen_inventory(tmp_path)
    with pytest.raises(verifier.CompatibilityVerifierError, match="link-like"):
        verifier._path(tmp_path, "link.bin")  # noqa: SLF001


def test_in_memory_or_noncanonical_config_is_rejected(tmp_path: Path) -> None:
    forged = copy.deepcopy(_config())
    forged["expected_result"]["local_qualification"] = True
    with pytest.raises(verifier.CompatibilityVerifierError, match="supplied"):
        verifier.load_config(ROOT, supplied_config=forged)
    other = tmp_path / "config.json"
    other.write_text("{}", encoding="utf-8")
    with pytest.raises(verifier.CompatibilityVerifierError, match="canonical"):
        verifier.load_config(ROOT, requested_path=other)


def test_cli_is_check_only_and_preserves_every_frozen_byte() -> None:
    before = _frozen_snapshot()
    command = [
        str(ROOT / ".venv-p1" / "Scripts" / "python.exe"),
        "-B",
        str(
            ROOT
            / "scripts"
            / "verify_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v1.py"
        ),
        "--root",
        str(ROOT),
        "--mode",
        "check-only",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS_R3_COMPATIBILITY_VERIFIER_ONLY_RESEARCH_RESULT_LOCAL_FAIL"
    assert payload["files_written"] == 0
    assert payload["compatibility_control_exists"] is False
    assert payload["check_only_parent_process"] == {
        "numerical_modules_before": [],
        "numerical_modules_after": [],
        "new_numerical_modules": [],
        "helper_imported": True,
        "r3_guard_imported": True,
        "r3_engine_imported": False,
    }
    assert _frozen_snapshot() == before


def test_source_has_no_write_execution_or_personal_path_surface() -> None:
    paths = [ROOT / verifier.IMPLEMENTATION_ROLES[role] for role in ("CONFIG", "HELPER", "CLI")]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert not re.search(
        r"(?i)(?:[A-Z]:[\\/]Users[\\/][A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+)",
        text,
    )
    helper = (ROOT / verifier.IMPLEMENTATION_ROLES["HELPER"]).read_text(encoding="utf-8")
    cli = (ROOT / verifier.IMPLEMENTATION_ROLES["CLI"]).read_text(encoding="utf-8")
    for forbidden in (
        "exclusive_json(",
        "exclusive_bytes(",
        "consume_attempt_lock(",
        "execute_layer4_curve(",
        "_load_metric_truth_after_commitment(",
    ):
        assert forbidden not in helper
        assert forbidden not in cli


def test_compatibility_control_and_receipts_are_absent() -> None:
    config = _config()
    assert not (ROOT / config["canonical_paths"]["compatibility_control"]).exists()
    assert not (ROOT / config["canonical_paths"]["pre_execution_qa"]).exists()
    assert not (ROOT / config["canonical_paths"]["compatibility_receipt"]).exists()


def test_new_implementation_pin_roles_are_complete() -> None:
    pins = verifier.implementation_pins(ROOT)
    assert set(pins) == {"CONFIG", "HELPER", "CLI", "TESTS"}
    assert all(pin["bytes"] > 0 and len(pin["sha256"]) == 64 for pin in pins.values())


def test_v9_is_still_seq5_and_upload_zero() -> None:
    result = verifier.verify_static_compatibility(ROOT)["v9"]
    assert result["sequences"] == [3, 4, 5]
    assert result["head_sequence"] == 5
    assert result["head_event_sha256"] == (
        "1b3e01be70c6f8ed2df04038deac3b3642804f70f9f17a238826c64d68090317"
    )
    assert result["uploads"] == 0


def test_python_optimized_mode_does_not_remove_verifier_checks() -> None:
    helper = ROOT / verifier.IMPLEMENTATION_ROLES["HELPER"]
    source = helper.read_text(encoding="utf-8")
    assert "assert " not in source
    command = [
        str(ROOT / ".venv-p1" / "Scripts" / "python.exe"),
        "-O",
        "-B",
        str(ROOT / verifier.IMPLEMENTATION_ROLES["CLI"]),
        "--root",
        str(ROOT),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout)["files_written"] == 0


def test_no_new_numerical_stack_is_imported_by_fresh_check_process() -> None:
    cli = ROOT / verifier.IMPLEMENTATION_ROLES["CLI"]
    completed = subprocess.run(
        [
            str(ROOT / ".venv-p1" / "Scripts" / "python.exe"),
            "-B",
            str(cli),
            "--root",
            str(ROOT),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    process = payload["check_only_parent_process"]
    assert process["numerical_modules_before"] == []
    assert process["numerical_modules_after"] == []
    assert process["new_numerical_modules"] == []
    assert process["r3_engine_imported"] is False


def test_result_metrics_are_finite_without_recomputation() -> None:
    config = _config()
    metrics = json.loads(
        (ROOT / config["r3_core_artifact_pins"]["METRICS"]["path"]).read_text(encoding="utf-8")
    )
    assert [point["fraction"] for point in metrics["points"]] == [0.4, 0.55, 0.7, 0.85, 1.0]
    assert all(
        isinstance(point[key], float)
        for point in metrics["points"]
        for key in ("incumbent", "challenger", "delta")
    )


def test_config_sha_constant_matches_exact_bytes() -> None:
    config_path = ROOT / verifier.CONFIG_RELATIVE
    assert r3.sha256_file(config_path) == verifier.CONFIG_SHA256
    assert config_path.stat().st_size == 8891
