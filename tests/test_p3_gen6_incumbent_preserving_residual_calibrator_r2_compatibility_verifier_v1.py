from __future__ import annotations

import copy
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from p3_wave import gen6_incumbent_preserving_residual_calibrator_contract_r2 as r2
from p3_wave import (
    gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v1 as verifier,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return verifier.load_config(ROOT)


def _snapshot() -> dict[str, object]:
    config = _config()
    output = ROOT / config["canonical_paths"]["r2_output"]
    control = ROOT / config["canonical_paths"]["r2_control"]
    compatibility_control = ROOT / config["canonical_paths"]["compatibility_control"]
    return {
        "r2_implementation": verifier.verify_pin_map(
            ROOT, config["r2_implementation_pins"], label="r2 implementation"
        ),
        "r2_control": verifier.verify_pin_map(ROOT, config["r2_control_pins"], label="r2 control"),
        "r2_output": verifier.verify_pin_map(ROOT, config["r2_output_pins"], label="r2 output"),
        "control_inventory": verifier.frozen_inventory(control),
        "output_inventory": verifier.frozen_inventory(output),
        "v9": verifier._verify_v9(ROOT, config["v9_anchor"]),  # noqa: SLF001
        "compatibility_control_exists": compatibility_control.exists(),
    }


def test_config_identity_and_read_only_firewall_are_exact() -> None:
    config = _config()
    assert config["identity"] == verifier.IDENTITY
    assert config["verifier_only"] is config["check_only_default"] is True
    assert config["implementation_roles"] == verifier.IMPLEMENTATION_ROLES
    for key in (
        "r2_mutation_allowed",
        "r2_rerun_or_resume_allowed",
        "execution_authorization_or_attempt_lock_allowed",
        "fit_prediction_source_truth_decode_or_experiment_scoring_allowed",
        "compatibility_receipt_write_allowed",
        "official_promotion_allowed",
        "candidate_or_test_prediction_allowed",
        "registry_append_allowed",
        "upload_allowed",
    ):
        assert config[key] is False
    assert all(value == 0 for value in config["static_counters"].values())


def test_exact_r2_pins_whole_tree_allowlist_and_v9_are_frozen() -> None:
    config = _config()
    snapshot = _snapshot()
    assert snapshot["control_inventory"] == config["r2_control_inventory"]
    assert snapshot["output_inventory"] == config["r2_output_inventory"]
    output = ROOT / config["canonical_paths"]["r2_output"]
    directories = {"."} | {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_dir()
    }
    files = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    assert directories == set(config["r2_output_contract"]["allowed_directories"])
    assert files == set(config["r2_output_contract"]["allowed_files"])
    assert len(config["r2_output_pins"]) == 14
    assert snapshot["v9"] == {
        "path": "artifacts/meaningful_score_goal_v9/registry.jsonl",
        "bytes": 15812,
        "sha256": "232b6ed3133de11ee05150ec439efe05baa315bbb64ea0f319ffcbddd421b965",
        "sequences": [3, 4, 5],
        "head_sequence": 5,
        "head_event_sha256": ("1b3e01be70c6f8ed2df04038deac3b3642804f70f9f17a238826c64d68090317"),
        "uploads": 0,
    }


def test_prefix_is_derived_from_all_pinned_sources_and_defect_is_exact() -> None:
    result = verifier.derive_science_prefixes(ROOT, _config())
    corrected = [0.4, 0.55, 0.7, 0.85, 1.0]
    assert result["corrected_prefix_fractions"] == corrected
    assert result["frozen_r2_erroneous_prefix_fractions"] == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert all(value == corrected for value in result["sources"].values())
    assert tuple(r2.PREFIX_FRACTIONS) == (0.2, 0.4, 0.6, 0.8, 1.0)


def test_prefix_consensus_rejects_one_forged_source() -> None:
    expected = [0.4, 0.55, 0.7, 0.85, 1.0]
    sources = {"science": expected, "commitment": [0.4, 0.55, 0.7, 0.9, 1.0]}
    with pytest.raises(verifier.CompatibilityVerifierError, match="disagree"):
        verifier._require_prefix_consensus(sources, expected)  # noqa: SLF001


def test_only_exact_historical_failure_receipt_is_accepted() -> None:
    config = _config()
    receipt = verifier.verify_historical_failure_receipt(ROOT, config)
    assert receipt["exception_type"] == "R2ContractError"
    assert receipt["message_sha256"] == (
        "d3e1c00aaadbcd5f2a62a00b1a9e16999ae6e3877ba91ce7484f27e5852b99b1"
    )
    assert receipt["capability_revoked"] is True
    for field, value in (
        ("uploads", 1),
        ("candidate_created", True),
        ("message_sha256", "0" * 64),
    ):
        forged = copy.deepcopy(receipt)
        forged[field] = value
        with pytest.raises(verifier.CompatibilityVerifierError, match="semantics"):
            verifier._validate_failure_payload(  # noqa: SLF001
                forged, config["compatibility_contract"]
            )
    forged = copy.deepcopy(receipt)
    forged["extra"] = 0
    with pytest.raises(verifier.CompatibilityVerifierError, match="field set"):
        verifier._validate_failure_payload(  # noqa: SLF001
            forged, config["compatibility_contract"]
        )


def test_full_compatibility_verifier_passes_and_preserves_every_frozen_byte() -> None:
    before = _snapshot()
    original_prefixes = r2.PREFIX_FRACTIONS
    original_inventory = r2._control_inventory  # noqa: SLF001
    result = verifier.verify_static_compatibility(ROOT)
    assert _snapshot() == before
    assert r2.PREFIX_FRACTIONS is original_prefixes
    assert r2._control_inventory is original_inventory  # noqa: SLF001
    assert result["status"] == "PASS_R2_COMPATIBILITY_VERIFIER_RESEARCH_ONLY_NO_PROMOTION"
    assert result["frozen_r2_verifier"]["status"] == (
        "POST_PUBLISH_VERIFIED_EXACT_ALLOWLIST_AND_LINEAGE"
    )
    assert result["oof_reconciliation"]["truth_bytes_exact_to_sealed_gen1"] is True
    assert result["oof_reconciliation"]["keys_exact_to_sealed_gen1"] is True
    metric = result["independent_metric_verification"]
    assert metric["bootstrap_replicates_total"] == 25000
    assert metric["points_deep_equal"] is True
    assert metric["gate_deep_equal"] is True
    assert metric["central_evidence_deep_equal"] is True
    assert metric["gate"]["decision"] == "RESEARCH_ONLY"
    assert metric["gate"]["passed"] is False
    for key in (
        "files_written",
        "independent_qa_receipts_created",
        "compatibility_receipts_created",
        "execution_authorizations_created",
        "attempt_locks_created",
        "model_fit_calls",
        "prediction_calls",
        "source_train_target_scalar_decodes",
        "experiment_score_calls",
        "candidate_files",
        "test_prediction_files",
        "registry_appends",
        "uploads",
    ):
        assert result[key] == 0


def test_patch_scope_restores_r2_globals_after_exception() -> None:
    config = _config()
    r2_config, _raw = r2.load_canonical_config(ROOT, ROOT / r2.CONFIG_RELATIVE)
    original_prefixes = r2.PREFIX_FRACTIONS
    original_inventory = r2._control_inventory  # noqa: SLF001
    with pytest.raises(LookupError, match="forced"):
        with verifier._r2_compatibility_scope(  # noqa: SLF001
            ROOT, config, r2_config, [0.4, 0.55, 0.7, 0.85, 1.0]
        ):
            assert tuple(r2.PREFIX_FRACTIONS) == (0.4, 0.55, 0.7, 0.85, 1.0)
            raise LookupError("forced")
    assert r2.PREFIX_FRACTIONS is original_prefixes
    assert r2._control_inventory is original_inventory  # noqa: SLF001


def test_alternate_or_in_memory_config_is_rejected(tmp_path: Path) -> None:
    forged = copy.deepcopy(_config())
    forged["official_promotion_allowed"] = True
    with pytest.raises(verifier.CompatibilityVerifierError, match="supplied"):
        verifier.load_config(ROOT, supplied_config=forged)
    alternate = tmp_path / "config.json"
    alternate.write_text("{}", encoding="utf-8")
    with pytest.raises(verifier.CompatibilityVerifierError, match="alternate"):
        verifier.load_config(ROOT, requested_path=alternate)


def test_inventory_and_containment_reject_linklike_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ordinary = tmp_path / "ordinary.bin"
    ordinary.write_bytes(b"x")
    original = Path.is_symlink

    def classify(path: Path) -> bool:
        return path == ordinary or original(path)

    monkeypatch.setattr(Path, "is_symlink", classify)
    with pytest.raises(verifier.CompatibilityVerifierError, match="link/reparse"):
        verifier.frozen_inventory(tmp_path)
    with pytest.raises(verifier.CompatibilityVerifierError, match="contained"):
        verifier.contained_path(ROOT, "../outside", must_exist=False)


def test_real_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    link = tmp_path / "link.bin"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("local Windows policy does not permit test symlinks")
    with pytest.raises(verifier.CompatibilityVerifierError, match="link/reparse"):
        verifier.frozen_inventory(tmp_path)


def test_cli_is_check_only_and_preserves_r2_and_v9() -> None:
    before = _snapshot()
    command = [
        str(ROOT / ".venv-p1" / "Scripts" / "python.exe"),
        "-B",
        str(ROOT / verifier.IMPLEMENTATION_ROLES["CLI"]),
        "--root",
        str(ROOT),
        "--mode",
        "check-only",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS_R2_COMPATIBILITY_VERIFIER_RESEARCH_ONLY_NO_PROMOTION"
    assert payload["compatibility_control_exists"] is False
    assert payload["compatibility_qa_receipt_exists"] is False
    assert payload["compatibility_receipt_exists"] is False
    assert payload["check_only_parent_process"] == {
        "mode": "check-only",
        "numerical_modules_before": [],
        "new_numerical_modules": ["numpy", "pandas", "pyarrow"],
        "helper_imported": True,
        "r2_contract_imported": True,
        "r2_engine_imported": False,
    }
    assert _snapshot() == before


def test_source_has_no_write_execution_engine_or_personal_path_surface() -> None:
    paths = [ROOT / verifier.IMPLEMENTATION_ROLES[role] for role in ("CONFIG", "HELPER", "CLI")]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert not re.search(
        r"(?i)(?:[A-Z]:[\\/]Users[\\/][A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+)",
        text,
    )
    helper = (ROOT / verifier.IMPLEMENTATION_ROLES["HELPER"]).read_text(encoding="utf-8")
    cli = (ROOT / verifier.IMPLEMENTATION_ROLES["CLI"]).read_text(encoding="utf-8")
    for forbidden in (
        "robust_write_exclusive(",
        "write_output_exclusive(",
        "write_failure_receipt(",
        "create_attempt_lock(",
        "issue_execution_capability(",
        "execute_gen6_curve(",
        "gen6_incumbent_preserving_residual_calibrator_execution_r2 import",
    ):
        assert forbidden not in helper
        assert forbidden not in cli
    assert "assert " not in helper


def test_compatibility_control_qa_and_receipt_are_absent() -> None:
    config = _config()
    for key in ("compatibility_control", "pre_execution_qa", "compatibility_receipt"):
        assert not (ROOT / config["canonical_paths"][key]).exists()


def test_new_implementation_roles_and_config_pin_are_complete() -> None:
    pins = verifier.implementation_pins(ROOT)
    assert set(pins) == {"CONFIG", "HELPER", "CLI", "TESTS"}
    assert all(pin["bytes"] > 0 and len(pin["sha256"]) == 64 for pin in pins.values())
    config_path = ROOT / verifier.CONFIG_RELATIVE
    assert r2.sha256_file(config_path) == verifier.CONFIG_SHA256
    assert config_path.stat().st_size == 12799


def test_python_optimized_mode_keeps_all_checks() -> None:
    completed = subprocess.run(
        [
            str(ROOT / ".venv-p1" / "Scripts" / "python.exe"),
            "-O",
            "-B",
            str(ROOT / verifier.IMPLEMENTATION_ROLES["CLI"]),
            "--root",
            str(ROOT),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    payload = json.loads(completed.stdout)
    assert payload["files_written"] == 0
    assert payload["independent_metric_verification"]["gate"]["passed"] is False
