"""Prepare the P3 RevIN Patch v1 one-shot without opening outer labels or training."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pyarrow_dataset
import torch

from p3_trajectory import metric_slices, paired_event_bootstrap
from p3_wave.data import audit_p3_data, load_p3_data
from p3_wave.revin_patch import (
    FULL_AUTHORIZATION_TOKEN,
    TwoStreamRevINPatchTransformer,
    assign_storm_episodes_from_wave,
    blend_long_leads,
    bounded_cpu_backward_smoke,
    bounded_training_protocol_smoke,
    build_episode_disjoint_folds_from_ids,
    build_inner_episode_split,
    event_balanced_weights,
    fold_coverage,
    load_preregistration,
    refit_fixed_epoch_and_predict,
    select_epoch_on_inner_split,
    sha256_file,
    validate_preregistration,
    validate_raw_context,
)
from p3_wave.sequences import build_test_sequences, build_train_sequences

DEFAULT_CONFIG = "configs/experiments/p3_revin_patch_v1.json"
DEFAULT_STATUS = "artifacts/status/p3_revin_patch_v1.json"
DEFAULT_FULL_OUTPUT = "artifacts/p3_revin_patch_v1/full_one_shot"
PAIR_KEYS = ["fold", "anchor_id", "station", "lead_h"]


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _status(
    path: Path,
    *,
    state: str,
    phase: str,
    progress: float,
    detail: str,
    elapsed_seconds: float | None = None,
    eta: str | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "title": "P3 RevIN two-stream PatchTST v1 preparation",
        "experiment_id": "p3_revin_patch_v1",
        "status": state,
        "phase": phase,
        "progress": float(progress),
        "detail": detail,
        "updated_at": _now(),
        "elapsed_seconds": None if elapsed_seconds is None else round(elapsed_seconds, 3),
        "eta": eta,
    }
    if result is not None:
        payload["result"] = result
    _atomic_json(path, payload)


def _git_state(root: Path) -> dict[str, Any]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    changed = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"sha": sha, "dirty": bool(changed), "changed_path_count": int(len(changed))}


def _source_hashes(data_dir: Path, config: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for filename, expected in config["source_sha256"].items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"P3 source input is missing: {filename}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"P3 source SHA changed for {filename}: {actual} != {expected}")
        result[filename] = actual
    return result


def _implementation_hashes(root: Path) -> dict[str, str]:
    paths = (
        "src/p3_wave/revin_patch.py",
        "scripts/run_p3_revin_patch_v1.py",
        "configs/experiments/p3_revin_patch_v1.json",
        "tests/test_p3_revin_patch.py",
    )
    return {relative: sha256_file(root / relative) for relative in paths}


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    with temporary.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _verify_blind_prediction_file(path: Path) -> dict[str, Any]:
    required = [
        "fold",
        "seed",
        "anchor_id",
        "station",
        "episode_id",
        "lead_h",
        "current_hs",
        "patch_prediction",
    ]
    frame = pd.read_parquet(path)
    if list(frame.columns) != required:
        raise ValueError(f"blind prediction schema mismatch: {path.name}")
    forbidden = {
        column for column in frame if "target" in column.lower() or "truth" in column.lower()
    }
    if forbidden:
        raise ValueError(f"outer target leaked into blind prediction: {sorted(forbidden)}")
    if frame.duplicated(PAIR_KEYS + ["seed"]).any():
        raise ValueError(f"duplicate blind prediction key: {path.name}")
    if not np.isfinite(frame[["current_hs", "patch_prediction"]].to_numpy()).all():
        raise ValueError(f"non-finite blind prediction: {path.name}")
    return {
        "rows": int(len(frame)),
        "cases": int(frame["anchor_id"].nunique()),
        "sha256": sha256_file(path),
    }


def _seal_blind_prediction_manifest(paths: list[Path], manifest_path: Path) -> dict[str, Any]:
    if len(paths) != 9:
        raise ValueError("exactly 3 outer folds x 3 seeds must be frozen before labels open")
    files: dict[str, Any] = {}
    for path in sorted(paths):
        try:
            relative = path.relative_to(manifest_path.parent).as_posix()
        except ValueError as error:
            raise ValueError("blind prediction is outside the one-shot output directory") from error
        files[relative] = _verify_blind_prediction_file(path)
    payload = {
        "sealed": True,
        "created_at": _now(),
        "expected_fold_seed_files": 9,
        "prediction_files": files,
        "outer_targets_opened_before_seal": False,
    }
    _atomic_json(manifest_path, payload)
    return {"manifest_sha256": sha256_file(manifest_path), **payload}


def _verify_blind_prediction_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("sealed") is not True or payload.get("expected_fold_seed_files") != 9:
        raise ValueError("blind prediction manifest is not sealed for all fold/seed files")
    if payload.get("outer_targets_opened_before_seal") is not False:
        raise ValueError("blind manifest reports premature outer target access")
    files = payload.get("prediction_files", {})
    if len(files) != 9:
        raise ValueError("blind manifest does not contain nine prediction files")
    for relative, record in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("unsafe path in blind prediction manifest")
        path = manifest_path.parent / relative_path
        current = _verify_blind_prediction_file(path)
        if current != record:
            raise ValueError(f"blind prediction changed after sealing: {relative}")
    return payload


@dataclass
class TargetVault:
    """Filtered target access with a single, manifest-gated outer-label opening."""

    path: Path
    access_log: list[dict[str, Any]] = field(default_factory=list)
    outer_open_count: int = 0

    def _read(self, anchor_ids: np.ndarray) -> pd.DataFrame:
        ids = np.asarray(anchor_ids, dtype=np.int64)
        if not len(ids) or len(np.unique(ids)) != len(ids):
            raise ValueError("target request must contain unique anchor ids")
        columns = ["anchor_id", *[f"target_{lead}" for lead in (3, 6, 9, 12, 18, 24)]]
        dataset = pyarrow_dataset.dataset(self.path, format="parquet")
        table = dataset.to_table(
            columns=columns,
            filter=pyarrow_dataset.field("anchor_id").isin(ids.tolist()),
        )
        frame = table.to_pandas().set_index("anchor_id").loc[ids].reset_index()
        if len(frame) != len(ids) or frame["anchor_id"].duplicated().any():
            raise ValueError("filtered target read has incomplete or duplicate coverage")
        if not np.isfinite(frame.drop(columns="anchor_id").to_numpy(dtype=np.float64)).all():
            raise ValueError("filtered target read contains non-finite official labels")
        return frame

    def read_outer_train(
        self,
        anchor_ids: np.ndarray,
        *,
        forbidden_outer_validation_ids: np.ndarray,
        fold: str,
    ) -> pd.DataFrame:
        ids = np.asarray(anchor_ids, dtype=np.int64)
        if np.intersect1d(ids, forbidden_outer_validation_ids).size:
            raise PermissionError("outer validation label requested during train/epoch selection")
        frame = self._read(ids)
        self.access_log.append(
            {"purpose": "outer_train_and_inner_epoch_selection", "fold": fold, "rows": len(frame)}
        )
        return frame

    def open_outer_once(
        self,
        anchor_ids: np.ndarray,
        *,
        blind_manifest_path: Path,
        exposure_receipt_path: Path,
    ) -> pd.DataFrame:
        if self.outer_open_count:
            raise PermissionError("outer validation labels may be opened exactly once")
        manifest = _verify_blind_prediction_manifest(blind_manifest_path)
        receipt = json.loads(exposure_receipt_path.read_text(encoding="utf-8"))
        if receipt.get("outer_validation_labels_opened") is not False:
            raise PermissionError("exposure receipt does not prove the pre-open state")
        if receipt.get("blind_prediction_manifest_sha256") != sha256_file(blind_manifest_path):
            raise PermissionError("exposure receipt does not match the sealed blind manifest")
        if receipt.get("fsync_completed_before_outer_open") is not True:
            raise PermissionError("exposure receipt lacks the pre-open fsync assertion")
        frame = self._read(anchor_ids)
        self.outer_open_count += 1
        self.access_log.append(
            {
                "purpose": "outer_evaluation_after_all_prediction_sha",
                "rows": len(frame),
                "blind_manifest_sha256": sha256_file(blind_manifest_path),
                "exposure_receipt_sha256": sha256_file(exposure_receipt_path),
                "prediction_file_count": len(manifest["prediction_files"]),
            }
        )
        return frame


def _load_anchor_metadata(
    root: Path,
    config: dict[str, Any],
    wave: pd.DataFrame,
) -> pd.DataFrame:
    record = config["frozen_inputs"]["anchor_metadata_cache"]
    path = root / record["path"]
    # Target columns are intentionally not read during preparation.
    anchors = pd.read_parquet(
        path,
        columns=["anchor_id", "station", "anchor_time", "grid_position", "current_hs"],
    )
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True)
    if anchors["anchor_id"].duplicated().any():
        raise ValueError("anchor metadata cache contains duplicate anchor ids")
    if set(anchors["station"].unique()) != {"G-ORS", "I-ORS", "S-ORS"}:
        raise ValueError("anchor metadata station set mismatch")
    if anchors["current_hs"].lt(1.5).any():
        raise ValueError("anchor metadata violates hs >= 1.5m")
    return assign_storm_episodes_from_wave(anchors.reset_index(drop=True), wave)


def _load_frozen_outer_validation_ids(
    root: Path,
    config: dict[str, Any],
    anchors: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Read only frozen incumbent OOF keys, never its predictions or labels."""

    record = config["frozen_inputs"]["incumbent_oof"]
    path = root / record["path"]
    keys = pd.read_parquet(path, columns=["fold", "anchor_id", "station", "lead_h"])
    if keys.duplicated(PAIR_KEYS).any():
        raise ValueError("frozen incumbent OOF contains duplicate keys")
    expected_leads = (3, 6, 9, 12, 18, 24)
    lead_tuple = keys.groupby(["fold", "anchor_id"], observed=True)["lead_h"].agg(tuple)
    if not lead_tuple.map(lambda value: value == expected_leads).all():
        raise ValueError("frozen incumbent OOF key rows do not contain six ordered leads")
    cases = keys[["fold", "anchor_id", "station"]].drop_duplicates()
    anchor_station = anchors.set_index("anchor_id")["station"].astype(str)
    mapped = cases["anchor_id"].map(anchor_station)
    if mapped.isna().any() or not mapped.eq(cases["station"].astype(str)).all():
        raise ValueError("frozen incumbent OOF keys do not match anchor metadata")
    return {
        str(fold): np.sort(group["anchor_id"].to_numpy(dtype=np.int64))
        for fold, group in cases.groupby("fold", observed=True)
    }


def _real_context_smoke(data: Any) -> dict[str, Any]:
    sequences = build_test_sequences(data)
    raw = torch.from_numpy(sequences.values)
    station = torch.from_numpy(sequences.station_code)
    validate_raw_context(raw)
    torch.manual_seed(20260821)
    model = TwoStreamRevINPatchTransformer().cpu().eval()
    with torch.no_grad():
        prediction = model(raw[:4], station[:4])
    if prediction.shape != (4, 6) or not torch.isfinite(prediction).all():
        raise RuntimeError("bounded real-context CPU forward smoke failed")
    return {
        "cases": int(len(raw)),
        "raw_shape": list(raw.shape),
        "wave_native_shape": [int(len(raw)), 145, 4],
        "atmos_native_shape": [int(len(raw)), 289, 6],
        "bounded_forward_batch": 4,
        "bounded_forward_shape": list(prediction.shape),
        "bounded_forward_finite": True,
        "same_case_groups_only": True,
        "absolute_test_time_used": False,
    }


def run_dry_run(
    *,
    root: Path,
    config_path: Path,
    data_dir: Path,
    status_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - started

    config = load_preregistration(config_path)
    _status(
        status_path,
        state="running",
        phase="validate_preregistration_and_frozen_sha",
        progress=5,
        detail="사전등록 고정값과 frozen CatBoost/router/submission SHA 검증 중",
        elapsed_seconds=elapsed(),
        eta="dry-run 약 1~3분",
    )
    prereg = validate_preregistration(config, root=root, verify_frozen_files=True)
    frozen_before = dict(prereg["frozen_sha256"])
    sources_before = _source_hashes(data_dir, config)
    implementation_before = _implementation_hashes(root)

    _status(
        status_path,
        state="running",
        phase="public_data_and_case_coverage",
        progress=25,
        detail="공식 스키마, 200개 익명 case, 두 native cadence coverage 검사 중",
        elapsed_seconds=elapsed(),
        eta="dry-run 약 1~2분",
    )
    data = load_p3_data(data_dir)
    public_audit = audit_p3_data(data)
    real_context = _real_context_smoke(data)

    _status(
        status_path,
        state="running",
        phase="episode_and_78h_split_audit",
        progress=55,
        detail="target 열을 열지 않고 storm episode·78시간 완전 분리 검사 중",
        elapsed_seconds=elapsed(),
        eta="dry-run 약 1분",
    )
    anchors = _load_anchor_metadata(root, config, data.wave)
    frozen_outer_ids = _load_frozen_outer_validation_ids(root, config, anchors)
    folds = build_episode_disjoint_folds_from_ids(
        anchors,
        windows=config["validation"]["windows"],
        validation_ids_by_fold=frozen_outer_ids,
        embargo_hours=int(config["validation"]["embargo_hours"]),
    )
    coverage = fold_coverage(anchors, folds)
    all_outer_ids = np.unique(np.concatenate([fold.validation_ids for fold in folds]))
    inner_split_coverage: dict[str, Any] = {}
    for fold in folds:
        effective_train_ids = np.setdiff1d(fold.train_ids, all_outer_ids)
        inner = build_inner_episode_split(
            anchors,
            effective_train_ids,
            validation_days=int(config["validation"]["inner_validation_days"]),
            embargo_hours=int(config["validation"]["embargo_hours"]),
        )
        inner_split_coverage[fold.name] = {
            "effective_outer_train_cases": int(len(effective_train_ids)),
            "inner_train_cases": int(len(inner.train_ids)),
            "inner_validation_cases": int(len(inner.validation_ids)),
            "outer_validation_union_target_rows_excluded": int(
                len(fold.train_ids) - len(effective_train_ids)
            ),
            "inner_ids_subset_of_outer_train": True,
        }

    _status(
        status_path,
        state="running",
        phase="bounded_cpu_forward_backward",
        progress=75,
        detail="합성 입력에서 bounded CPU forward/backward 및 6-lead head 검사 중",
        elapsed_seconds=elapsed(),
        eta="dry-run 1분 이내",
    )
    cpu_smoke = bounded_cpu_backward_smoke(seed=int(config["seed"]))
    training_protocol_smoke = bounded_training_protocol_smoke(seed=int(config["seed"]))

    frozen_after = validate_preregistration(config, root=root, verify_frozen_files=True)[
        "frozen_sha256"
    ]
    sources_after = _source_hashes(data_dir, config)
    implementation_after = _implementation_hashes(root)
    if frozen_before != frozen_after:
        raise RuntimeError("a frozen P3 artifact changed during dry-run")
    if sources_before != sources_after:
        raise RuntimeError("a source P3 input changed during dry-run")
    if implementation_before != implementation_after:
        raise RuntimeError("a P3 RevIN Patch implementation file changed during dry-run")

    result = {
        "created_at": _now(),
        "mode": "dry-run_only",
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "git": _git_state(root),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torch_cuda_available_observed_but_not_used": bool(torch.cuda.is_available()),
        },
        "preregistration": prereg,
        "public_audit": public_audit,
        "real_case_coverage": real_context,
        "train_anchor_metadata": {
            "rows": int(len(anchors)),
            "episodes": int(anchors.groupby(["station", "episode_id"], observed=True).ngroups),
            "target_columns_read": 0,
        },
        "fold_coverage": coverage,
        "inner_split_coverage": inner_split_coverage,
        "bounded_cpu_smoke": cpu_smoke,
        "bounded_cpu_training_protocol": training_protocol_smoke,
        "estimated_future_gpu_runtime_minutes": config["execution"][
            "estimated_full_gpu_runtime_minutes"
        ],
        "estimated_peak_vram_gb": config["execution"]["estimated_peak_vram_gb"],
        "full_runner_contract": {
            "fixed_seed_fold_prediction_files_before_outer_labels": 9,
            "epoch_selection": "outer-train-only episode-disjoint inner split",
            "outer_label_open_count_allowed": 1,
            "authorization_token_required": True,
            "full_gpu_run_now": False,
        },
        "sha256": {
            "source_before_and_after": sources_after,
            "frozen_before_and_after": frozen_after,
            "implementation_before_and_after": implementation_after,
        },
        "invariants": {
            "full_training_run": False,
            "outer_validation_labels_opened": False,
            "checkpoint_written": False,
            "submission_written_or_uploaded": False,
            "external_observations_used": 0,
            "test_absolute_time_recovered": False,
            "cross_test_case_context_used": False,
            "phase_shift_lattice_used": False,
            "hyperparameter_search_run": False,
            "source_rows_mutated": 0,
            "frozen_artifacts_unchanged": True,
        },
    }
    _status(
        status_path,
        state="ready_pending_root_approval",
        phase="dry_run_complete_full_locked",
        progress=100,
        detail="준비 PASS · full outer/GPU·outer label·checkpoint는 별도 승인 전 잠금",
        elapsed_seconds=elapsed(),
        eta="준비 완료 · full GPU 승인 후 45~90분",
        result=result,
    )
    return result


def _blind_prediction_frame(
    anchors: pd.DataFrame,
    anchor_ids: np.ndarray,
    prediction_delta: np.ndarray,
    *,
    fold: str,
    seed: int,
) -> pd.DataFrame:
    ids = np.asarray(anchor_ids, dtype=np.int64)
    delta = np.asarray(prediction_delta, dtype=np.float64)
    if delta.shape != (len(ids), 6):
        raise ValueError("outer blind prediction must have shape (cases, 6)")
    lookup = anchors.set_index("anchor_id").loc[ids]
    blocks: list[pd.DataFrame] = []
    for column, lead in enumerate((3, 6, 9, 12, 18, 24)):
        current = lookup["current_hs"].to_numpy(dtype=np.float64)
        blocks.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "seed": int(seed),
                    "anchor_id": ids,
                    "station": lookup["station"].astype(str).to_numpy(),
                    "episode_id": lookup["episode_id"].to_numpy(dtype=np.int64),
                    "lead_h": int(lead),
                    "current_hs": current,
                    "patch_prediction": np.clip(current + delta[:, column], 0.0, 30.0),
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def _target_long(frame: pd.DataFrame, fold_by_anchor: dict[int, str]) -> pd.DataFrame:
    result = frame.melt(
        id_vars="anchor_id",
        value_vars=[f"target_{lead}" for lead in (3, 6, 9, 12, 18, 24)],
        var_name="lead_h",
        value_name="target_hs",
    )
    result["lead_h"] = result["lead_h"].str.removeprefix("target_").astype(np.int64)
    result["fold"] = result["anchor_id"].map(fold_by_anchor)
    if result["fold"].isna().any():
        raise ValueError("outer target could not be mapped to an outer fold")
    return result


def _evaluate_gate(
    evaluated: pd.DataFrame,
    bootstrap: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    candidate = metric_slices(evaluated, "candidate_prediction")
    incumbent = metric_slices(evaluated, "incumbent_prediction")
    specification = config["gate"]
    lead_checks = {
        str(lead): candidate["by_lead"][str(lead)] <= incumbent["by_lead"][str(lead)]
        for lead in (18, 24)
    }
    station_limit = float(specification["maximum_station_rmse_degradation"])
    station_checks = {
        station: candidate["by_station"][station]
        <= incumbent["by_station"][station] + station_limit
        for station in candidate["by_station"]
    }
    checks = {
        "candidate_rmse_at_most_0p7701609198910191": candidate["rmse"]
        <= float(specification["maximum_candidate_rmse"]),
        "paired_case_bootstrap_ci90_upper_below_zero": float(bootstrap["ci90"][1]) < 0.0,
        "lead_18_non_degrading": bool(lead_checks["18"]),
        "lead_24_non_degrading": bool(lead_checks["24"]),
        "all_station_degradation_at_most_0p010": bool(all(station_checks.values())),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "lead_checks": lead_checks,
        "station_checks": station_checks,
        "candidate": candidate,
        "incumbent": incumbent,
        "delta_rmse": float(candidate["rmse"] - incumbent["rmse"]),
    }


def run_full_one_shot(
    *,
    root: Path,
    config_path: Path,
    data_dir: Path,
    status_path: Path,
    output_dir: Path,
    authorization_token: str,
) -> dict[str, Any]:
    """Run the preregistered GPU probe; never called by the preparation dry-run."""

    if authorization_token != FULL_AUTHORIZATION_TOKEN:
        raise PermissionError("exact root authorization token is required for full GPU execution")
    if output_dir.exists():
        raise FileExistsError("full one-shot output already exists; refusing overwrite or rerun")
    config = load_preregistration(config_path)
    prereg = validate_preregistration(config, root=root, verify_frozen_files=True)
    dry_status = json.loads(status_path.read_text(encoding="utf-8"))
    if dry_status.get("status") != "ready_pending_root_approval":
        raise PermissionError("a passing dry-run status is required before full execution")
    if dry_status.get("result", {}).get("config_sha256") != sha256_file(config_path):
        raise PermissionError(
            "config changed after dry-run; rerun preparation before full execution"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("full P3 RevIN Patch v1 requires an available CUDA device")

    started = time.perf_counter()
    frozen_before = dict(prereg["frozen_sha256"])
    sources_before = _source_hashes(data_dir, config)
    implementation_before = _implementation_hashes(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    _status(
        status_path,
        state="running_authorized_full",
        phase="build_past_context_and_nested_splits",
        progress=2,
        detail="승인된 full one-shot · outer target vault 잠금 상태로 context 구성 중",
        elapsed_seconds=time.perf_counter() - started,
        eta="약 45~90분",
    )

    data = load_p3_data(data_dir)
    public_audit = audit_p3_data(data)
    anchors = _load_anchor_metadata(root, config, data.wave)
    sequences = build_train_sequences(data, anchors)
    frozen_outer_ids = _load_frozen_outer_validation_ids(root, config, anchors)
    folds = build_episode_disjoint_folds_from_ids(
        anchors,
        windows=config["validation"]["windows"],
        validation_ids_by_fold=frozen_outer_ids,
        embargo_hours=int(config["validation"]["embargo_hours"]),
    )
    all_outer_ids = np.unique(np.concatenate([fold.validation_ids for fold in folds]))
    fold_by_anchor = {
        int(anchor_id): fold.name for fold in folds for anchor_id in fold.validation_ids
    }
    if len(fold_by_anchor) != len(all_outer_ids):
        raise ValueError("an outer validation anchor appears in more than one fold")

    target_record = config["frozen_inputs"]["anchor_metadata_cache"]
    vault = TargetVault(root / target_record["path"])
    anchor_lookup = anchors.set_index("anchor_id")
    training = config["training"]
    seeds = tuple(int(value) for value in training["fixed_seeds"])
    blind_paths: list[Path] = []
    training_receipts: dict[str, Any] = {}
    device = torch.device("cuda")

    for fold_number, fold in enumerate(folds):
        # Strong global blind contract: none of the 182 outer target rows is read as a target
        # for any fold, even when an earlier validation window is chronologically available.
        effective_train_ids = np.setdiff1d(fold.train_ids, all_outer_ids)
        target_frame = vault.read_outer_train(
            effective_train_ids,
            forbidden_outer_validation_ids=all_outer_ids,
            fold=fold.name,
        )
        target_lookup = target_frame.set_index("anchor_id")
        target_columns = [f"target_{lead}" for lead in (3, 6, 9, 12, 18, 24)]
        current = anchor_lookup.loc[effective_train_ids, "current_hs"].to_numpy(dtype=np.float32)
        target_delta = (
            target_lookup.loc[effective_train_ids, target_columns].to_numpy(dtype=np.float32)
            - current[:, None]
        )
        weight = event_balanced_weights(anchors, effective_train_ids).astype(np.float32)
        target_by_id = {
            int(anchor_id): row
            for anchor_id, row in zip(effective_train_ids, target_delta, strict=True)
        }
        weight_by_id = {
            int(anchor_id): float(value)
            for anchor_id, value in zip(effective_train_ids, weight, strict=True)
        }
        inner = build_inner_episode_split(
            anchors,
            effective_train_ids,
            validation_days=int(config["validation"]["inner_validation_days"]),
            embargo_hours=int(config["validation"]["embargo_hours"]),
        )
        fold_receipt: dict[str, Any] = {
            "effective_outer_train_cases": int(len(effective_train_ids)),
            "globally_blinded_outer_cases_removed": int(
                len(fold.train_ids) - len(effective_train_ids)
            ),
            "inner_train_cases": int(len(inner.train_ids)),
            "inner_validation_cases": int(len(inner.validation_ids)),
            "outer_validation_cases": int(len(fold.validation_ids)),
            "seeds": {},
        }

        def rows(
            ids: np.ndarray,
            lookup: dict[int, np.ndarray] = target_by_id,
        ) -> np.ndarray:
            return np.stack([lookup[int(anchor_id)] for anchor_id in ids]).astype(np.float32)

        def weights(
            ids: np.ndarray,
            lookup: dict[int, float] = weight_by_id,
        ) -> np.ndarray:
            return np.asarray([lookup[int(anchor_id)] for anchor_id in ids], dtype=np.float32)

        for seed_number, seed in enumerate(seeds):
            progress = 5 + 9 * (fold_number * len(seeds) + seed_number)
            _status(
                status_path,
                state="running_authorized_full",
                phase=f"{fold.name}_seed_{seed}_inner_epoch_then_refit",
                progress=progress,
                detail="epoch는 outer-train 내부 inner episode split만으로 선택 중",
                elapsed_seconds=time.perf_counter() - started,
                eta="전체 약 45~90분",
            )
            selection = select_epoch_on_inner_split(
                sequences.values[inner.train_ids],
                sequences.station_code[inner.train_ids],
                rows(inner.train_ids),
                weights(inner.train_ids),
                sequences.values[inner.validation_ids],
                sequences.station_code[inner.validation_ids],
                rows(inner.validation_ids),
                seed=seed,
                device=device,
                maximum_epochs=int(training["maximum_epochs"]),
                patience=int(training["patience"]),
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                gradient_clip_norm=float(training["gradient_clip_norm"]),
            )
            prediction_delta = refit_fixed_epoch_and_predict(
                sequences.values[effective_train_ids],
                sequences.station_code[effective_train_ids],
                target_delta,
                weight,
                sequences.values[fold.validation_ids],
                sequences.station_code[fold.validation_ids],
                seed=seed,
                device=device,
                epochs=selection.selected_epoch,
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                gradient_clip_norm=float(training["gradient_clip_norm"]),
            )
            blind = _blind_prediction_frame(
                anchors,
                fold.validation_ids,
                prediction_delta,
                fold=fold.name,
                seed=seed,
            )
            blind_path = output_dir / "blind" / fold.name / f"seed_{seed}.parquet"
            _atomic_parquet(blind_path, blind)
            blind_paths.append(blind_path)
            selection_receipt = {
                "selected_epoch": int(selection.selected_epoch),
                "epochs_ran": int(selection.epochs_ran),
                "best_inner_rmse": float(selection.best_inner_rmse),
                "inner_rmse_history": list(selection.inner_rmse_history),
                "outer_validation_labels_opened": False,
                "blind_prediction_sha256": sha256_file(blind_path),
            }
            fold_receipt["seeds"][str(seed)] = selection_receipt
            receipt_path = output_dir / "epoch_selection" / fold.name / f"seed_{seed}.json"
            _atomic_json(receipt_path, selection_receipt)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        training_receipts[fold.name] = fold_receipt

    manifest_path = output_dir / "blind_prediction_manifest.json"
    blind_manifest = _seal_blind_prediction_manifest(blind_paths, manifest_path)
    exposure_receipt_path = output_dir / "outer_label_exposure_receipt.json"
    exposure_receipt = {
        "created_at": _now(),
        "blind_prediction_manifest_sha256": sha256_file(manifest_path),
        "fold_seed_prediction_file_count": len(blind_paths),
        "outer_validation_labels_opened": False,
        "fsync_completed_before_outer_open": True,
    }
    _atomic_json(exposure_receipt_path, exposure_receipt)
    _status(
        status_path,
        state="running_authorized_full",
        phase="all_fold_seed_predictions_sha_frozen_open_outer_once",
        progress=89,
        detail=f"9개 blind 파일 SHA 고정({blind_manifest['manifest_sha256'][:12]}) 후 outer label 1회 개방",
        elapsed_seconds=time.perf_counter() - started,
        eta="약 2~5분",
    )

    outer_target = vault.open_outer_once(
        all_outer_ids,
        blind_manifest_path=manifest_path,
        exposure_receipt_path=exposure_receipt_path,
    )
    if vault.outer_open_count != 1:
        raise AssertionError("outer label vault open count differs from one")
    blind = pd.concat([pd.read_parquet(path) for path in blind_paths], ignore_index=True)
    seed_count = blind.groupby(PAIR_KEYS, observed=True)["seed"].nunique()
    if not seed_count.eq(3).all():
        raise ValueError("a blind outer row does not have exactly three seed predictions")
    ensemble = (
        blind.groupby(PAIR_KEYS + ["episode_id", "current_hs"], as_index=False, observed=True)[
            "patch_prediction"
        ]
        .mean()
        .sort_values(PAIR_KEYS)
        .reset_index(drop=True)
    )
    incumbent_path = root / config["frozen_inputs"]["incumbent_oof"]["path"]
    incumbent = pd.read_parquet(incumbent_path, columns=[*PAIR_KEYS, "prediction"]).rename(
        columns={"prediction": "incumbent_prediction"}
    )
    evaluated = ensemble.merge(incumbent, on=PAIR_KEYS, how="inner", validate="one_to_one")
    if len(evaluated) != len(ensemble) or len(evaluated) != len(incumbent):
        raise ValueError("blind prediction and frozen incumbent OOF keys do not match exactly")
    targets = _target_long(outer_target, fold_by_anchor)
    evaluated = evaluated.merge(
        targets,
        on=["fold", "anchor_id", "lead_h"],
        how="inner",
        validate="one_to_one",
    ).sort_values(PAIR_KEYS)
    if len(evaluated) != len(incumbent):
        raise ValueError("outer target coverage does not match frozen incumbent OOF")
    lead_tuple = evaluated.groupby(["fold", "anchor_id"], observed=True)["lead_h"].agg(tuple)
    if not lead_tuple.map(lambda value: value == (3, 6, 9, 12, 18, 24)).all():
        raise ValueError("evaluated rows do not contain the six official leads in order")
    incumbent_matrix = evaluated["incumbent_prediction"].to_numpy().reshape(-1, 6)
    patch_matrix = evaluated["patch_prediction"].to_numpy().reshape(-1, 6)
    candidate_matrix = blend_long_leads(incumbent_matrix, patch_matrix, patch_weight=0.2)
    evaluated["candidate_prediction"] = candidate_matrix.reshape(-1)
    protected = evaluated["lead_h"].isin([3, 6, 9])
    if not np.array_equal(
        evaluated.loc[protected, "candidate_prediction"].to_numpy(),
        evaluated.loc[protected, "incumbent_prediction"].to_numpy(),
    ):
        raise AssertionError("protected 3/6/9h incumbent predictions changed")

    bootstrap = paired_event_bootstrap(
        evaluated,
        candidate_column="candidate_prediction",
        baseline_column="incumbent_prediction",
        replicates=int(config["validation"]["bootstrap_replicates"]),
        seed=int(config["seed"]),
    )
    gate = _evaluate_gate(evaluated, bootstrap, config)
    evaluated_path = output_dir / "evaluated_after_outer_open.parquet"
    _atomic_parquet(evaluated_path, evaluated)

    frozen_after = validate_preregistration(config, root=root, verify_frozen_files=True)[
        "frozen_sha256"
    ]
    sources_after = _source_hashes(data_dir, config)
    implementation_after = _implementation_hashes(root)
    if frozen_before != frozen_after or sources_before != sources_after:
        raise RuntimeError("frozen or source P3 artifact changed during full one-shot")
    if implementation_before != implementation_after:
        raise RuntimeError("implementation changed during full one-shot")
    result = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": "gate_passed" if gate["passed"] else "gate_failed",
        "elapsed_seconds": float(time.perf_counter() - started),
        "config_sha256": sha256_file(config_path),
        "git": _git_state(root),
        "public_audit": public_audit,
        "training_receipts": training_receipts,
        "blind_manifest": blind_manifest,
        "pre_outer_open_exposure_receipt": {
            **exposure_receipt,
            "sha256": sha256_file(exposure_receipt_path),
        },
        "target_access_log": vault.access_log,
        "outer_validation_label_open_count": int(vault.outer_open_count),
        "paired_case_bootstrap": bootstrap,
        "gate": gate,
        "invariants": {
            "outer_epoch_or_blend_selection_used_outer_labels": False,
            "all_seed_fold_prediction_sha_fixed_before_outer_open": True,
            "outer_validation_labels_opened_exactly_once": True,
            "fixed_seeds": list(seeds),
            "patch_weight": 0.2,
            "protected_3_6_9_exact": True,
            "checkpoint_written": False,
            "submission_written_or_uploaded": False,
            "external_observations_used": 0,
            "test_absolute_time_recovered": False,
            "phase_shift_lattice_used": False,
            "hyperparameter_search_run": False,
        },
        "sha256": {
            "blind_prediction_manifest": sha256_file(manifest_path),
            "pre_outer_open_exposure_receipt": sha256_file(exposure_receipt_path),
            "evaluated_predictions": sha256_file(evaluated_path),
            "frozen_before_and_after": frozen_after,
            "source_before_and_after": sources_after,
            "implementation_before_and_after": implementation_after,
        },
    }
    metrics_path = output_dir / "metrics.json"
    _atomic_json(metrics_path, result)
    _status(
        status_path,
        state="completed_gate_pass" if gate["passed"] else "completed_gate_fail",
        phase="authorized_full_complete",
        progress=100,
        detail=f"candidate RMSE {gate['candidate']['rmse']:.6f} · gate {'PASS' if gate['passed'] else 'FAIL'}",
        elapsed_seconds=time.perf_counter() - started,
        eta="완료",
        result=result,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--status-path", default=DEFAULT_STATUS)
    parser.add_argument("--output-dir", default=DEFAULT_FULL_OUTPUT)
    parser.add_argument("--authorization-token", default=None)
    parser.add_argument("--mode", choices=("dry-run", "full-one-shot"), default="dry-run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve()
    status_path = (root / args.status_path).resolve()
    output_dir = (root / args.output_dir).resolve()
    data_value = args.data_dir or os.environ.get("P3_DATA_DIR")
    if not data_value:
        raise ValueError("--data-dir or P3_DATA_DIR is required")
    data_dir = Path(data_value).expanduser().resolve()
    try:
        if args.mode == "dry-run":
            result = run_dry_run(
                root=root,
                config_path=config_path,
                data_dir=data_dir,
                status_path=status_path,
            )
        else:
            result = run_full_one_shot(
                root=root,
                config_path=config_path,
                data_dir=data_dir,
                status_path=status_path,
                output_dir=output_dir,
                authorization_token=args.authorization_token or "",
            )
    except Exception as error:
        _status(
            status_path,
            state="failed",
            phase="dry_run_stopped",
            progress=100,
            detail=f"{type(error).__name__}: {error}",
            eta="중단됨",
        )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
