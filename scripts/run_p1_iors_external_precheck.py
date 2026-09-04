"""Run the preregistered external-only I-ORS 2014--22 to 2023 LOO precheck."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_QUARANTINE = PROJECT_ROOT / "external_data" / "quarantine"
DEFAULT_OPTIONAL_DEPS = DEFAULT_QUARANTINE / "_deps"
for import_path in (SRC_DIR, DEFAULT_OPTIONAL_DEPS):
    if import_path.is_dir() and str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import numpy as np  # noqa: E402

from ocean_external.iors_ctd import (  # noqa: E402
    build_loo_dataset,
    ensure_archive,
    load_json_object,
    read_year_profile,
    sha256_file,
    validate_source_manifest,
    verify_archive,
    verify_official_record,
)
from ocean_external.iors_precheck import (  # noqa: E402
    apply_stop_gate,
    dataset_audit,
    evaluate_precheck,
    fit_quantile_models,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("configs/external_data/i_ors_ctd_v1_1_1.json"),
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/p1_iors_external_loo_precheck_v1.json"),
    )
    parser.add_argument("--quarantine-dir", type=Path, default=Path("external_data/quarantine"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p1_iors_external_precheck_v1")
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path("artifacts/status/p1_iors_external_precheck_v1.json"),
    )
    parser.add_argument(
        "--download", action="store_true", help="Download only the pinned KIOST ZIP"
    )
    parser.add_argument(
        "--skip-live-record-check",
        action="store_true",
        help="Use only pinned archive provenance (actual audited run should not use this)",
    )
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


class Status:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.perf_counter()

    def update(
        self,
        progress: float,
        phase: str,
        detail: str,
        *,
        status: str = "running",
    ) -> None:
        bounded = min(max(float(progress), 0.0), 100.0)
        elapsed = time.perf_counter() - self.started
        remaining = elapsed * (100.0 - bounded) / bounded if bounded > 0 else None
        eta = (
            datetime.now().astimezone() + timedelta(seconds=max(remaining or 0.0, 0.0))
            if remaining is not None
            else None
        )
        _atomic_json(
            self.path,
            {
                "title": "P1 I-ORS external-only LOO precheck",
                "experiment": "p1_iors_external_loo_precheck_v1",
                "status": status,
                "phase": phase,
                "progress": round(bounded, 2),
                "detail": detail,
                "elapsed_seconds": round(elapsed, 3),
                "eta": eta.strftime("%Y-%m-%d %H:%M:%S KST") if eta is not None else "측정 중",
                "updated_at": datetime.now().astimezone().isoformat(),
                "competition_labels_opened": False,
                "competition_outer_validation_opened": False,
                "competition_upload": False,
            },
        )


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        sha = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}
    return {"sha": sha, "dirty": dirty}


def _permission_audit(path: Path, source_id: str) -> dict[str, Any]:
    receipt = load_json_object(path)
    evidence = _resolve(Path(str(receipt["evidence_file"])))
    evidence_sha = sha256_file(evidence)
    expected_sha = str(receipt["evidence_sha256"]).lower()
    checks = {
        "status_approved": receipt.get("status") == "approved",
        "source_allowed": source_id in receipt.get("allowed_sources", []),
        "problem_allowed": "P1" in receipt.get("allowed_problems", []),
        "pretraining_allowed": "pretraining" in receipt.get("allowed_purposes", []),
        "evidence_sha256": evidence_sha == expected_sha,
    }
    if not all(checks.values()):
        raise PermissionError(f"official external-data FAQ permission failed: {checks}")
    return {
        "receipt": str(path.relative_to(PROJECT_ROOT)),
        "receipt_sha256": sha256_file(path),
        "organizer_channel": receipt.get("organizer_channel"),
        "evidence": str(evidence.relative_to(PROJECT_ROOT)),
        "evidence_sha256": evidence_sha,
        "checks": checks,
    }


def _environment() -> dict[str, Any]:
    try:
        import h5py
    except ImportError:
        h5py_version = None
    else:
        h5py_version = h5py.__version__
    try:
        import lightgbm
    except ImportError:
        lightgbm_version = None
    else:
        lightgbm_version = lightgbm.__version__
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "h5py": h5py_version,
        "lightgbm": lightgbm_version,
        "optional_dependency_path": str(DEFAULT_OPTIONAL_DEPS.resolve()),
        "git": _git_state(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_path = _resolve(args.source_manifest)
    experiment_path = _resolve(args.experiment)
    quarantine = _resolve(args.quarantine_dir)
    output_root = _resolve(args.output_dir)
    status = Status(_resolve(args.status_file))
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    result_path: Path | None = None
    try:
        status.update(2, "contract", "외부-only 사전등록·공식 FAQ 권한·경로 검사")
        source = load_json_object(source_path)
        experiment = load_json_object(experiment_path)
        validate_source_manifest(source)
        if experiment.get("status") != "preregistered_external_only":
            raise ValueError("experiment must remain preregistered_external_only")
        if experiment["decision_scope"]["cannot_promote_to_submission"] is not True:
            raise ValueError("precheck cannot be promoted directly to submission")
        permission = _permission_audit(
            PROJECT_ROOT / "configs/external_data/official_faq_permission.json",
            str(source["source_id"]),
        )
        source_contract_sha = sha256_file(source_path)
        experiment_contract_sha = sha256_file(experiment_path)

        status.update(8, "provenance", "KIOST 공식 페이지의 DOI·CC BY 4.0·v1.1.1 검증")
        official_record = (
            {"skipped": True, "reason": "--skip-live-record-check"}
            if args.skip_live_record_check
            else verify_official_record(source)
        )

        status.update(14, "archive", "격리 ZIP 다운로드/재사용 및 고정 SHA256 검사")
        archive_file = ensure_archive(source, quarantine, allow_download=args.download)
        archive_audit = verify_archive(archive_file, source)

        split = experiment["split"]
        target_depth_by_layer = {
            int(layer): float(depth) for layer, depth in experiment["target_grid"]["layers"].items()
        }
        max_distance = float(experiment["target_grid"]["max_mapping_distance_m"])
        years = list(split["fit_years"]) + [int(split["holdout_year"])]
        profiles = []
        for position, year in enumerate(years, start=1):
            progress = 20.0 + 30.0 * position / len(years)
            status.update(
                progress,
                "decode",
                f"{year} OceanSITES TIME·QC1·실제 수심 매핑 ({position}/{len(years)})",
            )
            profiles.append(
                read_year_profile(
                    archive_file,
                    source,
                    year=int(year),
                    target_depth_by_layer=target_depth_by_layer,
                    max_mapping_distance_m=max_distance,
                )
            )

        fit_years = set(int(value) for value in split["fit_years"])
        holdout_year = int(split["holdout_year"])
        fit_profiles = [profile for profile in profiles if profile.year in fit_years]
        holdout_profiles = [profile for profile in profiles if profile.year == holdout_year]
        if {profile.year for profile in fit_profiles} != fit_years or len(holdout_profiles) != 1:
            raise RuntimeError("year split contract failed")

        status.update(54, "features", "target TEMP를 마스킹한 leave-one-layer-out 특징 생성")
        fit_dataset = build_loo_dataset(
            fit_profiles,
            min_peer_temperatures=int(split["min_peer_temperatures"]),
            max_rows_per_year_layer=int(split["max_fit_rows_per_year_layer"]),
        )
        holdout_dataset = build_loo_dataset(
            holdout_profiles,
            min_peer_temperatures=int(split["min_peer_temperatures"]),
            max_rows_per_year_layer=None,
        )
        if sorted(int(item) for item in np.unique(fit_dataset.year)) != sorted(fit_years):
            raise RuntimeError("fit dataset includes an unexpected year")
        if set(int(item) for item in np.unique(holdout_dataset.year)) != {holdout_year}:
            raise RuntimeError("holdout dataset is not 2023-only")

        def model_progress(done: int, total: int, quantile: float) -> None:
            if done >= total:
                status.update(88, "model", "세 개 고정 quantile 모델의 2023 추론 완료")
            else:
                status.update(
                    60 + 27 * done / total,
                    "model",
                    f"LightGBM q={quantile:.1f} 고정 300-tree 학습 ({done + 1}/{total})",
                )

        predictions, model_audit = fit_quantile_models(
            fit_dataset,
            holdout_dataset,
            experiment["model"],
            progress=model_progress,
        )
        status.update(91, "evaluation", "2023 고정 holdout aggregate 지표·중단 gate 계산")
        metrics = evaluate_precheck(holdout_dataset, predictions)
        source_integrity = bool(archive_audit["integrity_verified"]) and not bool(
            official_record.get("skipped", False)
        )
        decision = apply_stop_gate(
            metrics,
            experiment["stop_gate"],
            source_integrity_verified=source_integrity,
        )

        run_id = started_at.strftime("%Y%m%dT%H%M%S%z")
        run_dir = output_root / run_id
        if run_dir.exists():
            raise FileExistsError(f"run output already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        result_path = run_dir / "result.json"
        result = {
            "schema_version": "1.0",
            "experiment_id": experiment["experiment_id"],
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now().astimezone().isoformat(),
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "scope": {
                "competition_labels_opened": False,
                "competition_outer_validation_opened": False,
                "competition_submission_created": False,
                "raw_observational_rows_serialized": False,
            },
            "contracts": {
                "source_manifest": str(source_path.relative_to(PROJECT_ROOT)),
                "source_manifest_sha256": source_contract_sha,
                "experiment": str(experiment_path.relative_to(PROJECT_ROOT)),
                "experiment_sha256": experiment_contract_sha,
                "effective_contract_sha256": _sha256_json(
                    {"source": source, "experiment": experiment}
                ),
            },
            "permission": permission,
            "official_record": official_record,
            "archive": archive_audit,
            "profile_quality": [profile.audit for profile in profiles],
            "fit_dataset": dataset_audit(fit_dataset),
            "holdout_dataset": dataset_audit(holdout_dataset),
            "model": model_audit,
            "metrics": metrics,
            "gate": decision,
            "environment": _environment(),
        }
        _atomic_json(result_path, result)
        receipt = {
            "result": str(result_path.resolve()),
            "result_sha256": sha256_file(result_path),
            "archive_sha256": archive_audit["sha256"],
            "source_manifest_sha256": source_contract_sha,
            "experiment_sha256": experiment_contract_sha,
            "decision": decision["decision"],
            "competition_upload": False,
        }
        receipt_path = run_dir / "receipt.json"
        _atomic_json(receipt_path, receipt)
        status.update(
            100,
            "complete",
            f"{decision['decision']} · result SHA {receipt['result_sha256'][:12]}",
            status="complete",
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        status.update(
            100 if result_path is not None else 0,
            "failed",
            f"{type(exc).__name__}: {exc}",
            status="failed",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
