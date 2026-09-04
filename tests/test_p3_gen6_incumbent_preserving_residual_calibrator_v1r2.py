from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave import gen6_incumbent_preserving_residual_calibrator_contract_r2 as guard
from p3_wave import gen6_incumbent_preserving_residual_calibrator_execution_r2 as engine

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / guard.CONFIG_RELATIVE
V1_PINS = {
    "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_v1.json": (
        16930,
        "f7584a811524f0dbe1a17f288cd82f3d3665bd44385d3355006a13bb49fb9968",
    ),
    "src/p3_wave/gen6_incumbent_preserving_residual_calibrator.py": (
        56425,
        "5f88fd05977ae2fa1dd7dcabacc31153fa63263c4f2b9167e4effb7d523414f8",
    ),
    "scripts/run_p3_gen6_incumbent_preserving_residual_calibrator_v1.py": (
        4909,
        "27fa6732e8461432ca4bc61f06ba0ccac95f6f42bd8cc94268044bbebc8ac6aa",
    ),
    "tests/test_p3_gen6_incumbent_preserving_residual_calibrator_v1.py": (
        7643,
        "53368c54b6dbbe1e67b027525fc93b579e77c6d42e8b23b766427f2e141bac22",
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_is_byte_exact_and_owner_no_go_is_append_only() -> None:
    for relative, (size, digest) in V1_PINS.items():
        path = ROOT / relative
        assert path.stat().st_size == size
        assert _sha(path) == digest
    lineage = (
        ROOT
        / "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1_no_go"
    )
    owner = json.loads((lineage / "OWNER_STATIC_QA_NO_GO_20260823.json").read_bytes())
    tombstone = json.loads((lineage / "EXECUTION_TOMBSTONE.json").read_bytes())
    assert owner["review"] == {
        **owner["review"],
        "reviewer": "/root/ledger_v9_independent_qa",
        "verdict": "NO-GO",
        "p0_count": 0,
        "p1_count": 3,
    }
    assert owner["review"]["independent_receipt_file_exists"] is False
    assert tombstone["status"] == "PERMANENTLY_TOMBSTONED_NEVER_EXECUTE"


def test_r2_config_is_one_science_preserving_seq5_path() -> None:
    config, raw = guard.load_canonical_config(ROOT, CONFIG)
    v1 = json.loads((ROOT / config["science_source"]["path"]).read_bytes())
    projection = {
        key: v1[key] for key in config["science_source"]["deep_equal_keys"]
    }
    assert guard.deep_sha256(projection) == guard.EXPECTED_SCIENCE_DEEP_SHA256
    assert len(raw) == guard.EXPECTED_CONFIG_BYTES
    assert guard.sha256_bytes(raw) == guard.EXPECTED_CONFIG_SHA256
    assert guard.deep_sha256(config) == guard.EXPECTED_CONFIG_DEEP_SHA256
    anchor = config["central_ledger_anchor"]
    assert anchor == {
        "ledger_id": "ocean_ai_meaningful_score_replay_ledger_v9",
        "path": "artifacts/meaningful_score_goal_v9/registry.jsonl",
        "bytes": 15812,
        "sha256": "232b6ed3133de11ee05150ec439efe05baa315bbb64ea0f319ffcbddd421b965",
        "physical_event_lines": 3,
        "global_head_seq": 5,
        "head_event_sha256": (
            "1b3e01be70c6f8ed2df04038deac3b3642804f70f9f17a238826c64d68090317"
        ),
        "official_uploads": 0,
        "automatic_rebase_allowed": False,
        "on_change": "FAIL_CLOSE_BEFORE_QA_AUTH_LOCK_ENGINE_OR_PUBLISH",
    }
    fits = config["fit_count_contract"]
    assert fits["static_observed_fit_calls"] == 0
    assert fits["authorized_maximum_fit_calls"] == 20
    assert fits["maximum_is_an_authorization_bound_not_a_promised_or_preregistered_observed_count"]
    assert config["static_counters"] == {
        "independent_qa_receipts": 0,
        "authorizations": 0,
        "attempt_locks": 0,
        "fit_calls": 0,
        "prediction_cells": 0,
        "score_calls": 0,
        "test_value_reads": 0,
        "candidate_files": 0,
        "registry_appends": 0,
        "uploads": 0,
    }


def test_output_contract_is_exact() -> None:
    config = json.loads(CONFIG.read_bytes())
    assert tuple(config["output_contract"]["allowed_directories"]) == (
        ".",
        "blind",
        "commitments",
    )
    assert tuple(config["output_contract"]["allowed_files"]) == guard.ALLOWED_FILES
    assert len(guard.ALLOWED_FILES) == 14
    assert guard.CORE_FILES == guard.ALLOWED_FILES[:11]


def test_robust_writer_handles_partial_writes_and_rejects_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "exclusive.bin"
    real_write = os.write

    def partial_write(descriptor: int, value: object) -> int:
        return real_write(descriptor, bytes(value)[:3])

    monkeypatch.setattr(guard.os, "write", partial_write)
    guard.robust_write_exclusive(path, b"0123456789")
    assert path.read_bytes() == b"0123456789"
    with pytest.raises(FileExistsError):
        guard.robust_write_exclusive(path, b"replacement")


def test_output_inventory_rejects_extra_missing_and_reparse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    (output / "blind").mkdir()
    (output / "commitments").mkdir()
    config = {"canonical_paths": {"output": "out"}}
    report = guard._assert_output_subset(tmp_path, config, final=False)
    assert report["directories"] == [".", "blind", "commitments"]
    with pytest.raises(guard.R2ContractError, match="missing"):
        guard._assert_output_subset(tmp_path, config, final=True)
    extra = output / "unexpected.txt"
    extra.write_bytes(b"x")
    with pytest.raises(guard.R2ContractError, match="extra file"):
        guard._assert_output_subset(tmp_path, config, final=False)
    extra.unlink()
    real_has_reparse = guard._has_reparse
    monkeypatch.setattr(
        guard,
        "_has_reparse",
        lambda path: path.name == "blind" or real_has_reparse(path),
    )
    with pytest.raises(guard.R2ContractError, match="symlink/reparse"):
        guard._assert_output_subset(tmp_path, config, final=False)


def test_live_registry_rejects_direct_replay_tamper_and_revoked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(guard.R2CapabilityError, match="direct or forged"):
        guard.capability_context(object())

    identity = object()
    capability = guard._LiveCapability(identity, guard._CAPABILITY_SENTINEL)
    record = guard._LiveRecord(
        capability=capability,
        identity=identity,
        root=tmp_path,
        data_dir=tmp_path,
        config={"nested": {"value": 1}},
        phase="ENGINE_LOAD_INPUTS",
        lock_payload={"nonce": "opaque"},
        lock_sha256="0" * 64,
        qa_sha256="1" * 64,
        authorization_sha256="2" * 64,
        static_lineage={
            "implementation_pins": {},
            "central_v9_anchor": {},
            "nested": {"value": 1},
        },
        process_id=os.getpid(),
    )
    guard._LIVE_REGISTRY[id(capability)] = record
    monkeypatch.setattr(guard, "_revalidate_live", lambda _record: None)
    try:
        context = guard.capability_context(capability)
        context["config"]["nested"]["value"] = 999
        assert record.config["nested"]["value"] == 1
        guard.enter_engine_phase(
            capability,
            expected_phase="ENGINE_LOAD_INPUTS",
            next_phase="FOLD_0_PREDICT_COMMIT",
            entry_name="test",
        )
        with pytest.raises(guard.R2CapabilityError, match="replay/order"):
            guard.enter_engine_phase(
                capability,
                expected_phase="ENGINE_LOAD_INPUTS",
                next_phase="FOLD_0_PREDICT_COMMIT",
                entry_name="replay",
            )
        object.__setattr__(capability, "_LiveCapability__identity", object())
        with pytest.raises(guard.R2CapabilityError, match="opaque identity"):
            guard.capability_context(capability)
        object.__setattr__(capability, "_LiveCapability__identity", identity)
        guard.revoke_capability(capability, expected_phase="FOLD_0_PREDICT_COMMIT")
        with pytest.raises(guard.R2CapabilityError, match="revoked"):
            guard.capability_context(capability)
    finally:
        guard._LIVE_REGISTRY.pop(id(capability), None)


@pytest.mark.parametrize(
    "entry,args",
    [
        (engine.load_key_input_only, (object(),)),
        (engine.predict_and_commit_fold, (object(), object())),
        (engine.release_committed_fold_truth, (object(), object())),
        (engine.commit_predictions_complete, (object(), object())),
        (engine.score_and_write_core, (object(), object())),
        (engine.publish_manifest_sidecar_seal, (object(), object())),
    ],
)
def test_every_engine_entry_rejects_a_direct_call(entry: object, args: tuple[object, ...]) -> None:
    kwargs = {}
    if entry in {engine.predict_and_commit_fold, engine.release_committed_fold_truth}:
        kwargs["fold_index"] = 0
    with pytest.raises(guard.R2CapabilityError, match="direct or forged"):
        entry(*args, **kwargs)


def test_target_vault_constructor_rejects_a_direct_call(tmp_path: Path) -> None:
    with pytest.raises(guard.R2CapabilityError, match="direct or forged"):
        engine.SelectiveOfficialTargetVault(
            object(),
            tmp_path / "train_wave.csv",
            tmp_path / "out",
            pd.DataFrame(),
            pd.DataFrame(),
            expected_sha256="0" * 64,
            expected_bytes=0,
        )


def test_final_fold_truth_requires_predictions_complete_before_phase_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = type("State", (), {"predictions_complete": None})()
    entered: list[bool] = []
    monkeypatch.setattr(engine, "_require_state", lambda *_args: state)
    monkeypatch.setattr(
        guard, "enter_engine_phase", lambda *args, **kwargs: entered.append(True)
    )
    with pytest.raises(engine.R2ExecutionError, match="predictions-complete"):
        engine.release_committed_fold_truth(object(), state, fold_index=2)
    assert entered == []


def _synthetic_wave(tmp_path: Path, *, target: str) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    starts = pd.to_datetime(
        [
            "2024-01-01T00:00:00+09:00",
            "2024-03-01T00:00:00+09:00",
            "2024-05-01T00:00:00+09:00",
        ],
        utc=True,
    )
    anchors = pd.DataFrame(
        {
            "anchor_id": [0, 1, 2],
            "station": list(engine.STATIONS),
            "anchor_time": starts,
        }
    )
    validation = pd.DataFrame(
        {
            "fold": list(guard.FOLD_ORDER),
            "anchor_id": [0, 1, 2],
            "station": list(engine.STATIONS),
            "episode_id": [10, 20, 30],
        }
    )
    lines = ["station,time,hs,tp,hmax,wvdir"]
    for index, (station, start) in enumerate(zip(engine.STATIONS, starts, strict=True)):
        for lead in engine.LEADS:
            stamp = (start + pd.Timedelta(hours=lead)).tz_convert("Asia/Seoul")
            value = target if target == "POISON" else f"{index + lead / 100:.4f}"
            lines.append(f"{station},{stamp.isoformat()},{value},1.0,2.0,180")
    wave = tmp_path / "train_wave.csv"
    wave.write_text("\n".join(lines) + "\n", encoding="ascii", newline="")
    return wave, anchors, validation


def _vault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: str,
) -> tuple[engine.SelectiveOfficialTargetVault, object, Path]:
    wave, anchors, validation = _synthetic_wave(tmp_path, target=target)
    output = tmp_path / "out"
    (output / "blind").mkdir(parents=True)
    (output / "commitments").mkdir()
    capability = object()
    monkeypatch.setattr(guard, "verify_live_phase", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        guard,
        "capability_context",
        lambda _capability: {
            "root": tmp_path,
            "data_dir": tmp_path,
            "config": {
                "canonical_paths": {"output": "out"},
                "immutable_inputs": {
                    "source/train_wave.csv": {"path": "train_wave.csv"}
                },
            },
        },
    )
    vault = engine.SelectiveOfficialTargetVault(
        capability,
        wave,
        output,
        anchors,
        validation,
        expected_sha256=_sha(wave),
        expected_bytes=wave.stat().st_size,
        expected_rows=18,
        expected_validation_cases=3,
    )
    return vault, capability, output


def _commit_fold(output: Path, index: int, fold: str) -> tuple[Path, str]:
    prediction = output / f"blind/fold_{index:02d}_{fold}.npy"
    prediction.write_bytes(b"durable-blind")
    commitment = {
        "fold_index": index,
        "fold": fold,
        "truth_attached": False,
        "blind_prediction": guard.file_pin(prediction, root=output),
    }
    path = output / f"commitments/fold_{index:02d}_{fold}.json"
    path.write_bytes(guard.canonical_json_bytes(commitment) + b"\n")
    return path, _sha(path)


def test_target_poison_is_not_decoded_until_fold_commitment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault, capability, output = _vault(tmp_path, monkeypatch, target="POISON")
    assert vault.access_audit()["unique_source_target_scalar_decodes"] == 0
    commitment, digest = _commit_fold(output, 0, guard.FOLD_ORDER[0])
    with pytest.raises(engine.R2ExecutionError, match="valid ASCII float"):
        vault.release(
            capability,
            guard.FOLD_ORDER[0],
            fold_commitment_path=commitment,
            fold_commitment_sha256=digest,
        )
    assert vault.access_audit()["unique_source_target_scalar_decodes"] == 0


def test_target_release_is_fold_ordered_committed_and_nonreplayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault, capability, output = _vault(tmp_path, monkeypatch, target="NUMERIC")
    fold_paths = [
        _commit_fold(output, index, fold)
        for index, fold in enumerate(guard.FOLD_ORDER)
    ]
    with pytest.raises(PermissionError, match="order/replay"):
        vault.release(
            capability,
            guard.FOLD_ORDER[1],
            fold_commitment_path=fold_paths[1][0],
            fold_commitment_sha256=fold_paths[1][1],
        )
    assert vault.access_audit()["unique_source_target_scalar_decodes"] == 0
    for index, fold in enumerate(guard.FOLD_ORDER):
        truth = vault.release(
            capability,
            fold,
            fold_commitment_path=fold_paths[index][0],
            fold_commitment_sha256=fold_paths[index][1],
        )
        assert len(truth) == 6
        assert np.isfinite(truth["target_hs"].to_numpy()).all()
    assert vault.access_audit()["unique_source_target_scalar_decodes"] == 18
    with pytest.raises(PermissionError, match="replay"):
        vault.release(
            capability,
            guard.FOLD_ORDER[2],
            fold_commitment_path=fold_paths[2][0],
            fold_commitment_sha256=fold_paths[2][1],
        )


def test_static_preflight_is_read_only_when_canonical_environment_is_available() -> None:
    data = os.environ.get("P3_DATA_DIR")
    workspace = os.environ.get("P3_WORKSPACE_ROOT")
    if not data or not workspace:
        pytest.skip("canonical P3 environment is not active")
    before = {
        path: (ROOT / path).exists()
        for path in (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2",
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2_control",
        )
    }
    report = guard.static_preflight(ROOT, Path(data), requested_config=CONFIG)
    assert report["status"] == "STATIC_PREFLIGHT_PASS_NO_WRITES"
    assert report["target_scalar_decodes"] == 0
    assert report["fit_calls"] == 0
    assert report["prediction_cells"] == 0
    assert report["score_calls"] == 0
    assert report["uploads"] == 0
    assert before == {path: (ROOT / path).exists() for path in before}
