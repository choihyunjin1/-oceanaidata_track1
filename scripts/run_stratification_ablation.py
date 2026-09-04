"""Run the single preregistered 24-hour stratification peer-gate ablation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from p1_qc.config import load_config  # noqa: E402
from p1_qc.data import load_train_test  # noqa: E402
from p1_qc.experiment import RunRecorder, stable_hash, write_json  # noqa: E402
from p1_qc.pipeline import (  # noqa: E402
    load_or_build_features,
    resolve_data_dir,
    run_cross_validation,
)
from p1_qc.stratification import (  # noqa: E402
    PEER_GATE_FEATURES,
    PeerGateConfig,
    append_stratification_peer_gate,
)

EXPERIMENT_NAME = "strat_gate_fixed24h"
GATE_CONFIG = PeerGateConfig(mode="offline", window_hours=24, min_period_fraction=0.5)
BACKEND = "xgboost"
BOOTSTRAP_REPLICATES = 2000
AUGMENTATION = False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="Atomic aggregate-only progress JSON for the local gauge.",
    )
    return parser.parse_args(argv)


def _status_path(value: Path | None) -> Path:
    path = value or Path("artifacts/status/strat_gate_fixed24h.json")
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _write_status(
    path: Path,
    *,
    phase: str,
    progress: int,
    status: str = "running",
    detail: str,
    run_id: str | None = None,
) -> None:
    """Atomically replace the local aggregate progress file."""

    payload: dict[str, Any] = {
        "title": "P1 fixed 24h stratification ablation",
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "phase": phase,
        "progress": progress,
        "detail": detail,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    if run_id is not None:
        payload["run_id"] = run_id
    write_json(path, payload)


def _artifact_root(config: Any) -> Path:
    root = Path(config.paths.artifacts_dir)
    return root if root.is_absolute() else PROJECT_ROOT / root


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    status_path = _status_path(args.status_file)
    config_path = (PROJECT_ROOT / "configs" / "p1.toml").resolve(strict=True)
    data_env = os.environ.get("P1_DATA_DIR")
    if not data_env:
        raise OSError("P1_DATA_DIR must point to the immutable P1_qc_anomaly directory")

    # Only P1_DATA_DIR is admitted from the environment.  This prevents a
    # P1QC_* override from silently changing the preregistered base config.
    config = load_config(config_path, env={"P1_DATA_DIR": data_env})
    if config.mode != "offline" or config.features.mode != "offline":
        raise ValueError("the fixed stratification ablation requires offline configs/p1.toml")
    data_dir = resolve_data_dir(config)
    contract = {
        "experiment": EXPERIMENT_NAME,
        "base_config": config.to_dict(),
        "gate_config": asdict(GATE_CONFIG),
        "gate_features": list(PEER_GATE_FEATURES),
        "backend": BACKEND,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "augmentation": AUGMENTATION,
        "fixed_before_outer_cv": True,
    }
    recorder: RunRecorder | None = None
    progress = 0
    started = time.perf_counter()
    try:
        recorder = RunRecorder(
            EXPERIMENT_NAME,
            contract,
            root=_artifact_root(config),
            seed=config.seed,
        )
        recorder.copy_config(config_path)
        recorder.add_inputs(
            config=config_path,
            train=data_dir / "train.csv",
            test=data_dir / "test.csv",
        )
        progress = 25
        _write_status(
            status_path,
            phase="start",
            progress=progress,
            detail="고정 설정 확인 및 원본 읽기 전용 로드",
            run_id=recorder.run_id,
        )
        train, test = load_train_test(data_dir, audit=True, strict=True)
        base_bundle = load_or_build_features(
            train,
            config,
            kind="train",
            use_cache=True,
        )
        candidate_bundle = append_stratification_peer_gate(
            base_bundle,
            train,
            config=GATE_CONFIG,
            cadence_minutes=config.data.cadence_minutes,
            group_columns=config.data.group_columns,
        )
        feature_contract_sha256 = stable_hash(
            {
                "columns": candidate_bundle.feature_columns,
                "categorical": candidate_bundle.categorical_columns,
                "gate_config": asdict(GATE_CONFIG),
            }
        )
        gate_metadata = {
            "experiment": EXPERIMENT_NAME,
            "config": asdict(GATE_CONFIG),
            "feature_columns": list(PEER_GATE_FEATURES),
            "feature_count": len(PEER_GATE_FEATURES),
            "base_feature_count": len(base_bundle.feature_columns),
            "candidate_feature_count": len(candidate_bundle.feature_columns),
            "feature_contract_sha256": feature_contract_sha256,
            "label_blind": True,
            "outer_labels_used_for_gate_configuration": False,
            "gate_fixed_before_outer_cv": True,
            "calendar_or_month_hard_rules": False,
            "offline_future_dependency_hours": GATE_CONFIG.window_hours / 2,
            "segment_boundary_respected": True,
            "base_feature_cache_requested": True,
        }
        progress = 35
        _write_status(
            status_path,
            phase="feature",
            progress=progress,
            detail="cached base bundle에 고정 4개 label-blind 특징 추가 완료",
            run_id=recorder.run_id,
        )

        # Exactly one CV call.  Gate settings are never changed from its outer
        # results; promotion is a separate downstream decision.
        oof, metrics, selection = run_cross_validation(
            train,
            test,
            candidate_bundle,
            config,
            backend=BACKEND,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            augmentation=AUGMENTATION,
        )
        selection = dict(selection)
        selection["feature_hash"] = feature_contract_sha256
        selection["fixed_gate_ablation"] = {
            "experiment": EXPERIMENT_NAME,
            "config": asdict(GATE_CONFIG),
            "outer_labels_used_for_gate_configuration": False,
            "promotion_decision_performed_by_this_run": False,
        }
        progress = 70
        _write_status(
            status_path,
            phase="cv",
            progress=progress,
            detail="고정 후보의 단일 nested outer CV 완료",
            run_id=recorder.run_id,
        )

        progress = 90
        _write_status(
            status_path,
            phase="save",
            progress=progress,
            detail="OOF 및 aggregate 재현 산출물 저장 중",
            run_id=recorder.run_id,
        )
        oof_path = recorder.path / "oof.parquet"
        oof.to_parquet(oof_path, index=False, compression="zstd")
        recorder.record_file(oof_path)
        recorder.record_json("metrics.json", metrics)
        recorder.record_json("selection.json", selection)
        recorder.record_json("feature_gate.json", gate_metadata)
        runtime_seconds = time.perf_counter() - started
        recorder.finish(
            status="complete",
            micro_f1=metrics["aggregate"]["micro"]["f1"],
            weighted_f1=metrics["aggregate"]["weighted"]["f1"],
            runtime_seconds=runtime_seconds,
            competition_upload=False,
        )
        progress = 100
        _write_status(
            status_path,
            phase="done",
            progress=progress,
            status="complete",
            detail="고정 ablation 산출물 저장 및 SHA 기록 완료",
            run_id=recorder.run_id,
        )
        print(
            json.dumps(
                {
                    "run_id": recorder.run_id,
                    "run_path": str(recorder.path.resolve()),
                    "micro_f1": metrics["aggregate"]["micro"]["f1"],
                    "weighted_f1": metrics["aggregate"]["weighted"]["f1"],
                    "feature_contract_sha256": feature_contract_sha256,
                    "runtime_seconds": runtime_seconds,
                    "uploaded": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        if recorder is not None:
            recorder.finish(
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                runtime_seconds=time.perf_counter() - started,
                competition_upload=False,
            )
        _write_status(
            status_path,
            phase="failed",
            progress=progress,
            status="failed",
            detail=f"{type(exc).__name__}: {exc}",
            run_id=None if recorder is None else recorder.run_id,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
