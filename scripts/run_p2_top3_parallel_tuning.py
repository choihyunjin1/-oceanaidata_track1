"""Orchestrate the three independent P2 GBM tuning workers in parallel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.gbm_tuning import TUNING_FAMILIES
from p2_restore.submission import build_submission, validate_submission


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    for attempt in range(20):
        try:
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def _load_status(path: Path, family: str) -> dict[str, object]:
    if not path.is_file():
        return {"family": family, "status": "starting", "progress": 0.0, "phase": "start"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"family": family, "status": "starting", "progress": 0.0, "phase": "start"}


def _aggregate_status(
    path: Path,
    statuses: list[dict[str, object]],
    started: float,
    *,
    status: str = "running",
) -> None:
    progress = float(np.mean([float(value.get("progress", 0.0)) for value in statuses]))
    elapsed = time.perf_counter() - started
    remaining = elapsed * (100.0 - progress) / progress if progress > 0 else 0.0
    eta = datetime.now().astimezone() + timedelta(seconds=max(remaining, 0.0))
    detail = " | ".join(
        f"{value['family']}: {float(value.get('progress', 0.0)):.1f}% {value.get('phase', '')}"
        for value in statuses
    )
    _write_json(
        path,
        {
            "title": "P2 상위 3개 GBM 병렬 최적화",
            "status": status,
            "progress": 100.0 if status == "complete" else progress,
            "phase": "complete" if status == "complete" else "3개 독립 worker 실행",
            "detail": detail,
            "eta": eta.strftime("%Y-%m-%d %H:%M:%S KST") if progress else "초기 속도 측정 중",
            "updated_at": datetime.now().astimezone().isoformat(),
            "workers": statuses,
        },
    )


def _build_research_pair(
    winner: dict[str, object], data_dir: Path, output_root: Path
) -> dict[str, object]:
    tuned_path = Path(winner["artifacts"]["submission"]["path"])
    deep_path = Path("submissions/p2/P2_DEEP_STACK_V1.csv")
    data = load_p2_data(data_dir)
    tuned = pd.read_csv(tuned_path)
    deep = pd.read_csv(deep_path)
    if not tuned[["station", "layer", "time"]].equals(deep[["station", "layer", "time"]]):
        raise ValueError("tuned and deep submissions have different keys")
    weights = winner["deep_pair"]["fitted_weights_by_layer"]
    prediction = np.empty(len(deep), dtype=float)
    for layer in (2, 3, 4):
        selected = deep["layer"].to_numpy(int) == layer
        weight = float(weights[str(layer)])
        prediction[selected] = (1.0 - weight) * deep.loc[selected, "temp"].to_numpy(
            float
        ) + weight * tuned.loc[selected, "temp"].to_numpy(float)
    submission = build_submission(data.test_index, prediction)
    path = Path("submissions/p2/P2_DEEP_TOP3_TUNED_RESEARCH_V1.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    validation = validate_submission(path, data.test_index)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "validation": validation,
        "family": winner["family"],
        "weights_by_layer": weights,
        "uploaded": False,
        "output_root": str(output_root),
    }


def _finalize_results(
    args: argparse.Namespace,
    contract: dict[str, object],
    data_dir: Path,
    started: float,
) -> int:
    results = []
    for family in TUNING_FAMILIES:
        result_path = args.output_dir / family / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"worker result missing: {family}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("family") != family or result.get("uploaded") is not False:
            raise ValueError(f"invalid worker result: {family}")
        results.append(result)
    ranking = sorted(
        results,
        key=lambda result: (
            result["deep_pair"]["lobo_blend_rmse"],
            result["tuning"]["outer_rmse"],
            result["family"],
        ),
    )
    research_pair = _build_research_pair(ranking[0], data_dir, args.output_dir)
    master = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": contract["experiment_id"],
        "research_only": True,
        "uploaded": False,
        "preregistration_sha256": _sha256(args.preregistration),
        "elapsed_seconds": time.perf_counter() - started,
        "ranking_criterion": "lobo_deep_pair_rmse_then_outer_standalone_rmse",
        "ranking": [
            {
                "rank": rank,
                "family": result["family"],
                "best_inner_rmse": result["tuning"]["best_inner_rmse"],
                "outer_rmse": result["tuning"]["outer_rmse"],
                "lobo_pair_rmse": result["deep_pair"]["lobo_blend_rmse"],
                "lobo_delta_vs_deep": result["deep_pair"]["lobo_delta_vs_deep_lobo"],
                "full_fit_iterations": result["tuning"]["full_fit_iterations"],
                "best_parameters": result["tuning"]["best_parameters"],
            }
            for rank, result in enumerate(ranking, start=1)
        ],
        "research_pair_submission": research_pair,
        "family_result_hashes": {
            result["family"]: _sha256(args.output_dir / result["family"] / "result.json")
            for result in results
        },
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, master)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "result_sha256": _sha256(result_path),
            "preregistration_sha256": master["preregistration_sha256"],
            "research_pair_submission": research_pair,
            "uploaded": False,
        },
    )
    status_root = args.output_dir / "worker_status"
    statuses = [_load_status(status_root / f"{family}.json", family) for family in TUNING_FAMILIES]
    _aggregate_status(args.status_file, statuses, started, status="complete")
    print(json.dumps(master, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/experiments/p2_top3_parallel_tuning_v1.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p2_top3_parallel_tuning_v1")
    )
    parser.add_argument(
        "--status-file", type=Path, default=Path("artifacts/status/p2_top3_tuning.json")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    data_dir = resolve_data_dir(args.data_dir)
    contract = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if tuple(contract.get("families", ())) != TUNING_FAMILIES:
        raise ValueError("top-three tuning family order changed")
    if int(contract["parallelism"]["workers"]) != 3:
        raise ValueError("top-three worker count changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_root = args.output_dir / "worker_status"
    log_root = args.output_dir / "logs"
    status_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    if all((args.output_dir / family / "result.json").is_file() for family in TUNING_FAMILIES):
        return _finalize_results(args, contract, data_dir, started)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = str(contract["parallelism"]["threads_per_worker"])
    processes: dict[str, subprocess.Popen] = {}
    handles = []
    try:
        for family in TUNING_FAMILIES:
            status_path = status_root / f"{family}.json"
            log_path = log_root / f"{family}.log"
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            command = [
                sys.executable,
                "scripts/run_p2_gbm_tuning_worker.py",
                "--family",
                family,
                "--data-dir",
                str(data_dir),
                "--preregistration",
                str(args.preregistration),
                "--output-root",
                str(args.output_dir),
                "--status-file",
                str(status_path),
                "--trials",
                str(contract["search"]["trials_per_family"]),
            ]
            processes[family] = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
        while True:
            statuses = [
                _load_status(status_root / f"{family}.json", family) for family in TUNING_FAMILIES
            ]
            _aggregate_status(args.status_file, statuses, started)
            failed = [
                family for family, process in processes.items() if process.poll() not in (None, 0)
            ]
            if failed:
                for _family, process in processes.items():
                    if process.poll() is None:
                        process.terminate()
                raise RuntimeError(f"P2 tuning worker failed: {failed}; inspect {log_root}")
            if all(process.poll() == 0 for process in processes.values()):
                break
            time.sleep(1)
    finally:
        for handle in handles:
            handle.close()
    return _finalize_results(args, contract, data_dir, started)


if __name__ == "__main__":
    raise SystemExit(main())
