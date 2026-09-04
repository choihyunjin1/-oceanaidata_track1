"""Build the one-shot, no-truth P3 champion-matched ERA5 Hs2 candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p3_wave.champion_matched_era5_hs2_official_deploy import (  # noqa: E402
    align_transfer_predictions,
    build_relative_test_features,
    make_candidate,
)
from p3_wave.era5_context_transfer import (  # noqa: E402
    FixedContextTransferRegressor,
    build_source_cases,
    common_feature_columns,
)
from p3_wave.submission import validate_submission  # noqa: E402

EXPERIMENT_ID = "p3_champion_matched_era5_hs2_official_deploy_20260828_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
MODULE = ROOT / "src/p3_wave/champion_matched_era5_hs2_official_deploy.py"
RUNNER = Path(__file__).resolve()
QA_RUNNER = ROOT / f"scripts/qa_{EXPERIMENT_ID}.py"
TEST_FILE = ROOT / "tests/test_p3_champion_matched_era5_hs2_official_deploy_20260828_v1.py"
FROZEN_RUNNER = ROOT / "scripts/run_p3_era5_context_transfer_v1.py"
EXPECTED_ENVIRONMENT = {
    "python": "3.12.10",
    "catboost": "1.2.10",
    "numpy": "2.3.5",
    "pandas": "3.0.1",
    "pyarrow": "25.0.1",
    "scikit_learn": "1.9.0",
}
EXPECTED_CHAMPION_SHA = "ea65370a5c9291868769ad9e54a54707035dc93a01ffa4772d9fd26342f357aa"
SOURCE_YEARS = tuple(range(2014, 2024))
KEYS = ["case_id", "station", "lead_h"]


class DeploymentContractError(RuntimeError):
    """Fail-closed deployment contract error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DeploymentContractError(f"JSON root is not an object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_frozen_runner() -> Any:
    spec = importlib.util.spec_from_file_location("_p3_frozen_era5_deploy", FROZEN_RUNNER)
    if spec is None or spec.loader is None:
        raise DeploymentContractError("frozen ERA5 runner cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def environment_receipt() -> dict[str, Any]:
    import catboost
    import pyarrow
    import sklearn

    observed = {
        "python": platform.python_version(),
        "catboost": catboost.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__,
    }
    if observed != EXPECTED_ENVIRONMENT:
        raise DeploymentContractError(
            f"deployment environment drifted: expected={EXPECTED_ENVIRONMENT}, observed={observed}"
        )
    return {"passed": True, "sys_executable": sys.executable, "versions": observed}


def _resolve_input(spec: dict[str, Any]) -> Path:
    raw = Path(str(spec["path"]))
    return raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()


def validate_contract(config: dict[str, Any], *, require_output_absent: bool = True) -> dict[str, Any]:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise DeploymentContractError("experiment ID drifted")
    lineage = config["deployment_lineage"]
    if tuple(lineage["source"]["years"]) != SOURCE_YEARS:
        raise DeploymentContractError("full source years drifted")
    if lineage["model"] != {
        "family": "frozen FixedContextTransferRegressor",
        "seed": 20260824,
        "scientific_fit_count": 2,
        "parameter_search_count": 0,
        "source_or_local_hyperparameter_changes": 0,
        "fixed_model_postprocess": "20 percent persistence at leads 12,18,24",
    }:
        raise DeploymentContractError("frozen fit/search/model contract drifted")
    candidate = config["candidate"]
    if (
        candidate["champion_sha256"] != EXPECTED_CHAMPION_SHA
        or tuple(candidate["active_leads_h"]) != (18, 24)
        or tuple(candidate["inactive_leads_h"]) != (3, 6, 9, 12)
        or float(candidate["energy_weight"]) != 0.25
    ):
        raise DeploymentContractError("candidate contract drifted")
    official = config["official_contract"]
    if official.get("data_dir_env") != "P3_DATA_DIR":
        raise DeploymentContractError("official data resolver must be P3_DATA_DIR")
    if official.get("files_read") != ["test_context.parquet", "test_index.csv"]:
        raise DeploymentContractError("official input allowlist drifted")
    if int(official.get("official_truth_rows", -1)) != 0:
        raise DeploymentContractError("official truth boundary opened")
    policy = config["execution_policy"]
    if policy.get("official_upload_authorized") is not False or int(policy.get("upload_count", -1)) != 0:
        raise DeploymentContractError("upload boundary opened")
    if policy.get("result_driven_retry_or_tuning") is not False:
        raise DeploymentContractError("result-driven retry boundary opened")

    checked: dict[str, Any] = {}
    for name, spec in config["immutable_inputs"].items():
        path = _resolve_input(spec)
        if not path.is_file() or path.stat().st_size != int(spec["bytes"]):
            raise DeploymentContractError(f"immutable input missing/size drifted: {name}")
        digest = sha256_file(path)
        if digest != str(spec["sha256"]).lower():
            raise DeploymentContractError(f"immutable input hash drifted: {name}")
        checked[name] = {"bytes": path.stat().st_size, "sha256": digest}

    champion_manifest = read_json(_resolve_input(config["immutable_inputs"]["champion_manifest"]))
    if champion_manifest.get("sha256") != EXPECTED_CHAMPION_SHA:
        raise DeploymentContractError("champion manifest does not bind the expected CSV")
    replay = read_json(_resolve_input(config["immutable_inputs"]["historical_replay_result"]))
    if replay.get("status") != "GO_OFFICIAL_PROBE_LINEAGE_MATCHED":
        raise DeploymentContractError("champion-lineage historical replay is not GO")
    recovery = read_json(_resolve_input(config["immutable_inputs"]["dependency_recovery_result"]))
    if recovery.get("source_gate", {}).get("passed") is not True:
        raise DeploymentContractError("frozen ERA5 source gate is not PASS")

    outputs = config["outputs"]
    artifact = (ROOT / outputs["artifact_dir"]).resolve()
    report = (ROOT / outputs["report"]).resolve()
    candidate_dir = Path(outputs["candidate_dir"]).resolve()
    attempt_lock = artifact.with_name(f"{artifact.name}.attempt.lock")
    if require_output_absent and any(path.exists() for path in (artifact, report, candidate_dir, attempt_lock)):
        raise FileExistsError("one-shot deployment output or attempt lock already exists")

    data_value = os.environ.get("P3_DATA_DIR")
    if not data_value:
        raise DeploymentContractError("P3_DATA_DIR is required")
    data_dir = Path(data_value).resolve()
    for filename in official["files_read"]:
        if not (data_dir / filename).is_file():
            raise DeploymentContractError(f"P3_DATA_DIR is missing {filename}")
    return {
        "status": "PASS",
        "immutable_inputs": checked,
        "environment": environment_receipt(),
        "data_dir_resolved": True,
        "official_values_read": 0,
        "writes": 0,
        "fits": 0,
        "searches": 0,
        "upload_count": 0,
    }


def _full_source_positions(anchors: pd.DataFrame) -> np.ndarray:
    times = pd.to_datetime(anchors["anchor_time"], utc=True, errors="raise")
    years = times.dt.year
    complete = (
        (times - pd.Timedelta(hours=48)).dt.year.eq(years)
        & (times + pd.Timedelta(hours=24)).dt.year.eq(years)
        & years.isin(SOURCE_YEARS)
    )
    positions = np.flatnonzero(complete.to_numpy())
    if not len(positions) or set(years.iloc[positions].astype(int)) != set(SOURCE_YEARS):
        raise DeploymentContractError("full pre-2024 source deployment population is incomplete")
    return positions


def _write_csv_exclusive(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _preserve_inactive_champion_text(
    *, candidate_path: Path, champion_path: Path, inactive_leads: tuple[int, ...] = (3, 6, 9, 12)
) -> None:
    """Copy full inactive CSV rows byte-for-byte from the frozen champion."""
    champion_lines = champion_path.read_text(encoding="utf-8").splitlines(keepends=True)
    candidate_lines = candidate_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(champion_lines) != 1201 or len(candidate_lines) != 1201:
        raise DeploymentContractError("champion/candidate text row count drifted")
    if champion_lines[0] != candidate_lines[0]:
        raise DeploymentContractError("champion/candidate CSV header drifted")
    repaired = [candidate_lines[0]]
    inactive_count = 0
    for champion_line, candidate_line in zip(champion_lines[1:], candidate_lines[1:], strict=True):
        champion_fields = champion_line.rstrip("\r\n").split(",")
        candidate_fields = candidate_line.rstrip("\r\n").split(",")
        if champion_fields[:3] != candidate_fields[:3] or len(champion_fields) != 4:
            raise DeploymentContractError("champion/candidate textual key alignment drifted")
        if int(champion_fields[2]) in inactive_leads:
            repaired.append(champion_line)
            inactive_count += 1
        else:
            repaired.append(candidate_line)
    if inactive_count != 800:
        raise DeploymentContractError("inactive textual support drifted")
    temporary = candidate_path.with_suffix(candidate_path.suffix + ".partial")
    temporary.write_text("".join(repaired), encoding="utf-8", newline="")
    os.replace(temporary, candidate_path)


def _run_independent_validator(candidate_path: Path, data_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts/validate_p3_submission.py"),
        str(candidate_path),
        "--data-dir",
        str(data_dir),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise DeploymentContractError("scripts/validate_p3_submission.py failed")
    payload = json.loads(completed.stdout)
    if payload.get("status") != "passed_local_schema_and_key_validation":
        raise DeploymentContractError("independent validator did not return PASS")
    return {
        "script": "scripts/validate_p3_submission.py",
        "returncode": 0,
        "status": "passed_local_schema_and_key_validation",
        "submission_sha256": payload.get("submission_sha256"),
        "raw_prediction_statistics_persisted": False,
    }


def _report(result: dict[str, Any]) -> str:
    return (
        f"# {EXPERIMENT_ID}\n\n"
        f"- terminal status: `{result['status']}`\n"
        f"- rows/cases: `{result['surface']['rows']} / {result['surface']['cases']}`\n"
        f"- active/inactive: `{result['surface']['active_rows']} / {result['surface']['inactive_rows']}`\n"
        f"- scientific fits/searches: `{result['operations']['scientific_fits']} / {result['operations']['parameter_searches']}`\n"
        f"- official truth rows/upload count: `0 / 0`\n"
        f"- candidate SHA-256: `{result['candidate_sha256']}`\n"
        f"- independent validator: `{result['independent_validator']['status']}`\n\n"
        "No official values, anonymous absolute timestamps, or external evaluation-period observations are recorded here.\n"
    )


def execute_once(config: dict[str, Any]) -> dict[str, Any]:
    preflight = validate_contract(config, require_output_absent=True)
    outputs = config["outputs"]
    artifact = (ROOT / outputs["artifact_dir"]).resolve()
    report = (ROOT / outputs["report"]).resolve()
    candidate_dir = Path(outputs["candidate_dir"]).resolve()
    attempt_lock = artifact.with_name(f"{artifact.name}.attempt.lock")
    atomic_json(
        attempt_lock,
        {
            "experiment_id": EXPERIMENT_ID,
            "status": "ONE_SHOT_OFFICIAL_CANDIDATE_ATTEMPT_CONSUMED",
            "config_sha256": sha256_file(CONFIG),
            "official_truth_rows": 0,
            "upload_count": 0,
        },
    )
    artifact.mkdir(parents=False, exist_ok=False)

    frozen = load_frozen_runner()
    _, _, frozen_paths = frozen._load_contract(ROOT)
    source_hourly, source_provenance = frozen._load_source_hourly(frozen_paths)
    source_cases = build_source_cases(source_hourly, time_column="time", group_column="station")
    source_positions = _full_source_positions(source_cases.anchors)

    train_features_path = _resolve_input(config["immutable_inputs"]["train_features"])
    train_anchors_path = _resolve_input(config["immutable_inputs"]["train_anchors"])
    local_features = frozen._read_local_features(train_features_path)
    anchor_ids = pd.read_parquet(train_anchors_path, columns=["anchor_id"])["anchor_id"].to_numpy(
        dtype=np.int64
    )
    if len(anchor_ids) != 24_360 or len(np.unique(anchor_ids)) != len(anchor_ids):
        raise DeploymentContractError("full local deployment anchor population drifted")
    local_targets = frozen._read_training_targets(train_anchors_path, anchor_ids)
    feature_lookup = local_features.set_index("anchor_id")
    if not feature_lookup.index.is_unique or set(feature_lookup.index.astype(int)) != set(anchor_ids):
        raise DeploymentContractError("full local feature/target key alignment failed")
    local_x = feature_lookup.loc[anchor_ids, list(common_feature_columns())].reset_index(drop=True)

    model = FixedContextTransferRegressor().fit_pretrain(
        source_cases.features.iloc[source_positions].reset_index(drop=True),
        source_cases.log_delta_targets[source_positions],
    )
    model.continue_local(
        local_x,
        frozen._log_delta_targets(local_targets),
        current_hs=local_targets["current_hs"].to_numpy(dtype=np.float64),
    )

    data_dir = Path(os.environ["P3_DATA_DIR"]).resolve()
    context_path = data_dir / "test_context.parquet"
    index_path = data_dir / "test_index.csv"
    context = pd.read_parquet(context_path)
    test_index = pd.read_csv(index_path)
    test_features, case_metadata = build_relative_test_features(context, test_index)
    raw_transfer = model.predict_hs(
        test_features.loc[:, common_feature_columns()],
        current_hs=case_metadata["current_hs"].to_numpy(dtype=np.float64),
    )
    transfer_matrix = frozen._apply_fixed_shrink(
        raw_transfer,
        case_metadata["current_hs"].to_numpy(dtype=np.float64),
    )
    transfer = align_transfer_predictions(test_index, case_metadata, transfer_matrix)

    champion_path = _resolve_input(config["immutable_inputs"]["champion_submission"])
    champion = pd.read_csv(champion_path)
    if not champion[KEYS].equals(test_index[KEYS]):
        raise DeploymentContractError("champion and official index keys/order differ")
    validate_submission(champion, test_index)
    candidate_values, active = make_candidate(
        champion["hs_pred"].to_numpy(dtype=np.float64),
        transfer,
        test_index["lead_h"].to_numpy(dtype=np.int64),
    )
    candidate = champion.copy()
    candidate["hs_pred"] = candidate_values
    validate_submission(candidate, test_index)

    candidate_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = candidate_dir / outputs["candidate_filename"]
    _write_csv_exclusive(candidate, candidate_path)
    _preserve_inactive_champion_text(
        candidate_path=candidate_path,
        champion_path=champion_path,
    )
    reread = pd.read_csv(candidate_path)
    validate_submission(reread, test_index)
    inactive = ~active
    if not np.array_equal(
        reread.loc[inactive, "hs_pred"].to_numpy(dtype=np.float64),
        champion.loc[inactive, "hs_pred"].to_numpy(dtype=np.float64),
    ):
        raise DeploymentContractError("serialized inactive rows are not bit-exact champion")
    candidate_sha = sha256_file(candidate_path)
    validator = _run_independent_validator(candidate_path, data_dir)
    if validator["submission_sha256"] != candidate_sha:
        raise DeploymentContractError("independent validator candidate hash differs")

    candidate_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_NOT_UPLOADED",
        "submission_filename": candidate_path.name,
        "submission_sha256": candidate_sha,
        "champion_sha256": EXPECTED_CHAMPION_SHA,
        "rows": 1200,
        "cases": 200,
        "active_rows_18_24h": 400,
        "inactive_bit_exact_rows_3_6_9_12h": 800,
        "scientific_fits": 2,
        "parameter_searches": 0,
        "official_truth_rows": 0,
        "uploaded": False,
        "upload_count": 0,
    }
    atomic_json(candidate_dir / "MANIFEST.json", candidate_manifest)
    atomic_text(
        candidate_dir / "제출정보.txt",
        "제출물 제목: P3 Champion-matched ERA5 Hs² Residual\n"
        "한줄요약(접근방식): 현 챔피언을 유지하고 18·24시간 예측에만 사전등록된 ERA5 전이모델의 파랑에너지(Hs²) 잔차를 25% 반영했습니다.\n"
        "상태: 로컬 QA 완료, 업로드하지 않음\n",
    )

    result = {
        "schema_version": "p3.champion_matched_era5_hs2_official_deploy.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_NOT_UPLOADED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "deployment_lineage": {
            "source_years": list(SOURCE_YEARS),
            "source_complete_footprint_cases": int(len(source_positions)),
            "local_full_anchor_cases": int(len(anchor_ids)),
            "feature_count": len(common_feature_columns()),
            "seed": 20260824,
            "source_preflight_passed": bool(source_provenance["generic_preflight"]["accepted"]),
        },
        "surface": {
            "rows": int(len(reread)),
            "cases": int(reread["case_id"].nunique()),
            "active_rows": int(active.sum()),
            "inactive_rows": int(inactive.sum()),
            "inactive_champion_bit_exact": True,
        },
        "operations": {
            "scientific_fits": 2,
            "environment_smoke_fits": 0,
            "parameter_searches": 0,
            "official_truth_rows": 0,
            "anonymous_absolute_time_reconstructions": 0,
            "external_evaluation_period_matches": 0,
            "uploads": 0,
        },
        "official_input_hashes": {
            "test_context_sha256": sha256_file(context_path),
            "test_index_sha256": sha256_file(index_path),
        },
        "candidate_sha256": candidate_sha,
        "champion_sha256": sha256_file(champion_path),
        "independent_validator": validator,
        "preflight": preflight,
    }
    result_path = artifact / "result.json"
    atomic_json(result_path, result)
    atomic_text(report, _report(result))
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG),
        "module_sha256": sha256_file(MODULE),
        "runner_sha256": sha256_file(RUNNER),
        "qa_runner_sha256": sha256_file(QA_RUNNER),
        "test_sha256": sha256_file(TEST_FILE),
        "attempt_lock_sha256": sha256_file(attempt_lock),
        "result_sha256": sha256_file(result_path),
        "report_sha256": sha256_file(report),
        "candidate_manifest_sha256": sha256_file(candidate_dir / "MANIFEST.json"),
        "candidate_sha256": candidate_sha,
        "official_values_logged_or_reported": False,
        "official_truth_rows": 0,
        "upload_count": 0,
    }
    atomic_json(artifact / "manifest.json", manifest)
    return result


def finalize_serialization_guard(config: dict[str, Any]) -> dict[str, Any]:
    """Finish the consumed attempt without refitting after a 1-ULP CSV guard stop."""
    preflight = validate_contract(config, require_output_absent=False)
    outputs = config["outputs"]
    artifact = (ROOT / outputs["artifact_dir"]).resolve()
    report = (ROOT / outputs["report"]).resolve()
    candidate_dir = Path(outputs["candidate_dir"]).resolve()
    attempt_lock = artifact.with_name(f"{artifact.name}.attempt.lock")
    candidate_path = candidate_dir / outputs["candidate_filename"]
    if not attempt_lock.is_file() or not artifact.is_dir() or not candidate_path.is_file():
        raise DeploymentContractError("serialization recovery prerequisites are absent")
    if any(artifact.iterdir()) or report.exists():
        raise DeploymentContractError("serialization recovery is allowed only for the empty stopped artifact")
    if any((candidate_dir / name).exists() for name in ("MANIFEST.json", "제출정보.txt")):
        raise DeploymentContractError("candidate finalization already started")

    data_dir = Path(os.environ["P3_DATA_DIR"]).resolve()
    index_path = data_dir / "test_index.csv"
    context_path = data_dir / "test_context.parquet"
    test_index = pd.read_csv(index_path)
    champion_path = _resolve_input(config["immutable_inputs"]["champion_submission"])
    champion = pd.read_csv(champion_path)
    candidate_before = pd.read_csv(candidate_path)
    validate_submission(candidate_before, test_index)
    if not champion[KEYS].equals(test_index[KEYS]) or not candidate_before[KEYS].equals(
        test_index[KEYS]
    ):
        raise DeploymentContractError("serialization recovery key/order drifted")
    active = test_index["lead_h"].isin([18, 24]).to_numpy()
    inactive = ~active
    inactive_delta = np.abs(
        candidate_before.loc[inactive, "hs_pred"].to_numpy(dtype=np.float64)
        - champion.loc[inactive, "hs_pred"].to_numpy(dtype=np.float64)
    )
    if int(np.count_nonzero(inactive_delta)) != 4 or float(inactive_delta.max()) > 1e-12:
        raise DeploymentContractError("stopped candidate is not the exact known 1-ULP serialization case")
    pre_repair_sha = sha256_file(candidate_path)
    _preserve_inactive_champion_text(candidate_path=candidate_path, champion_path=champion_path)
    candidate = pd.read_csv(candidate_path)
    validate_submission(candidate, test_index)
    if not np.array_equal(
        candidate.loc[inactive, "hs_pred"].to_numpy(dtype=np.float64),
        champion.loc[inactive, "hs_pred"].to_numpy(dtype=np.float64),
    ):
        raise DeploymentContractError("inactive rows remain non-exact after textual repair")
    candidate_sha = sha256_file(candidate_path)
    validator = _run_independent_validator(candidate_path, data_dir)
    if validator["submission_sha256"] != candidate_sha:
        raise DeploymentContractError("independent validator candidate hash differs")

    frozen = load_frozen_runner()
    _, _, frozen_paths = frozen._load_contract(ROOT)
    source_hourly, source_provenance = frozen._load_source_hourly(frozen_paths)
    source_cases = build_source_cases(source_hourly, time_column="time", group_column="station")
    source_positions = _full_source_positions(source_cases.anchors)
    train_anchors_path = _resolve_input(config["immutable_inputs"]["train_anchors"])
    local_rows = pq.ParquetFile(train_anchors_path).metadata.num_rows
    if int(local_rows) != 24_360:
        raise DeploymentContractError("full local row count drifted during finalization")

    candidate_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_NOT_UPLOADED",
        "submission_filename": candidate_path.name,
        "submission_sha256": candidate_sha,
        "champion_sha256": EXPECTED_CHAMPION_SHA,
        "rows": 1200,
        "cases": 200,
        "active_rows_18_24h": 400,
        "inactive_bit_exact_rows_3_6_9_12h": 800,
        "scientific_fits": 2,
        "parameter_searches": 0,
        "serialization_only_repairs": 1,
        "official_truth_rows": 0,
        "uploaded": False,
        "upload_count": 0,
    }
    atomic_json(candidate_dir / "MANIFEST.json", candidate_manifest)
    atomic_text(
        candidate_dir / "제출정보.txt",
        "제출물 제목: P3 Champion-matched ERA5 Hs² Residual\n"
        "한줄요약(접근방식): 현 챔피언을 유지하고 18·24시간 예측에만 사전등록된 ERA5 전이모델의 파랑에너지(Hs²) 잔차를 25% 반영했습니다.\n"
        "상태: 로컬 QA 완료, 업로드하지 않음\n",
    )
    result = {
        "schema_version": "p3.champion_matched_era5_hs2_official_deploy.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_NOT_UPLOADED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "deployment_lineage": {
            "source_years": list(SOURCE_YEARS),
            "source_complete_footprint_cases": int(len(source_positions)),
            "local_full_anchor_cases": int(local_rows),
            "feature_count": len(common_feature_columns()),
            "seed": 20260824,
            "source_preflight_passed": bool(source_provenance["generic_preflight"]["accepted"]),
        },
        "surface": {
            "rows": 1200,
            "cases": 200,
            "active_rows": 400,
            "inactive_rows": 800,
            "inactive_champion_bit_exact": True,
        },
        "operations": {
            "scientific_fits": 2,
            "environment_smoke_fits": 0,
            "parameter_searches": 0,
            "serialization_only_repairs": 1,
            "official_truth_rows": 0,
            "anonymous_absolute_time_reconstructions": 0,
            "external_evaluation_period_matches": 0,
            "uploads": 0,
        },
        "official_input_hashes": {
            "test_context_sha256": sha256_file(context_path),
            "test_index_sha256": sha256_file(index_path),
        },
        "serialization_recovery": {
            "reason": "four inactive rows changed by at most one ULP during pandas CSV rewrite",
            "pre_repair_candidate_sha256": pre_repair_sha,
            "model_refits": 0,
            "prediction_regenerations": 0,
            "parameter_changes": 0,
            "inactive_rows_copied_text_exactly_from_champion": 800,
        },
        "candidate_sha256": candidate_sha,
        "champion_sha256": sha256_file(champion_path),
        "independent_validator": validator,
        "preflight": preflight,
    }
    result_path = artifact / "result.json"
    atomic_json(result_path, result)
    atomic_text(report, _report(result))
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG),
        "module_sha256": sha256_file(MODULE),
        "runner_sha256": sha256_file(RUNNER),
        "qa_runner_sha256": sha256_file(QA_RUNNER),
        "test_sha256": sha256_file(TEST_FILE),
        "attempt_lock_sha256": sha256_file(attempt_lock),
        "result_sha256": sha256_file(result_path),
        "report_sha256": sha256_file(report),
        "candidate_manifest_sha256": sha256_file(candidate_dir / "MANIFEST.json"),
        "candidate_sha256": candidate_sha,
        "official_values_logged_or_reported": False,
        "official_truth_rows": 0,
        "upload_count": 0,
    }
    atomic_json(artifact / "manifest.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--finalize-serialization-guard", action="store_true")
    arguments = parser.parse_args()
    config = read_json(CONFIG)
    if arguments.check_only:
        result = validate_contract(config)
    elif arguments.execute:
        result = execute_once(config)
    else:
        result = finalize_serialization_guard(config)
    safe = {
        "status": result["status"],
        "official_values_logged": False,
        "official_truth_rows": 0,
        "upload_count": 0,
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
