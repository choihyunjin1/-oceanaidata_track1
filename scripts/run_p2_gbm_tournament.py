"""Run the six-arm fixed-budget P2 GBM structure tournament locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.features import build_test_features, build_training_features
from p2_restore.gbm_tournament import (
    GBM_ARM_SPECS,
    align_with_deep_stack,
    arm_names,
    blend_by_layer,
    evaluate_deep_pair,
    fit_gbm_model,
    paired_day_bootstrap,
    run_blocked_arm,
)
from p2_restore.model import VALIDATION_BLOCKS
from p2_restore.research import (
    append_public_dynamics,
    append_public_m2_harmonics,
    select_lean_m2_dynamics,
)
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
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class Progress:
    def __init__(self, path: Path, jobs: int) -> None:
        self.path = path
        self.jobs = jobs
        self.done = 0
        self.started = time.perf_counter()

    def update(
        self,
        phase: str,
        detail: str,
        fraction: float = 0.0,
        *,
        status: str = "running",
    ) -> None:
        position = self.done + min(max(float(fraction), 0.0), 1.0)
        percent = 4.0 + 94.0 * position / max(self.jobs, 1)
        elapsed = time.perf_counter() - self.started
        remaining = elapsed * (self.jobs - position) / position if position > 0 else 0.0
        eta = datetime.now().astimezone() + timedelta(seconds=max(remaining, 0.0))
        _write_json(
            self.path,
            {
                "title": "P2 GBM 계열 비교",
                "status": status,
                "progress": 100.0 if status == "complete" else percent,
                "phase": phase,
                "detail": detail,
                "eta": eta.strftime("%Y-%m-%d %H:%M:%S KST") if position else "초기 속도 측정 중",
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )

    def finish(self) -> None:
        self.done += 1


def _git_state() -> dict[str, object]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "sha": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _validate_contract(path: Path, data_dir: Path) -> dict[str, object]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("experiment_id") != "p2_gbm_family_tournament_v1":
        raise ValueError("unexpected P2 GBM tournament id")
    if contract.get("status") != "authorized_local_structure_screen":
        raise ValueError("P2 GBM tournament is not locally authorized")
    if contract.get("research_only") is not True or contract.get("upload_allowed") is not False:
        raise ValueError("P2 GBM tournament must remain local-only")
    if contract.get("families") != arm_names():
        raise ValueError("P2 GBM family list changed")
    if int(contract["fixed_budget"]["boosting_iterations"]) != 400:
        raise ValueError("P2 GBM fixed iteration budget changed")
    sources = {
        "observations.csv": data_dir / "observations.csv",
        "test_index.csv": data_dir / "test_index.csv",
        "baseline_interp.csv": data_dir / "baseline_interp.csv",
        "frozen_router_oof": Path("artifacts/p2_max_round_convergence_v1/oof.parquet"),
        "deep_stack_oof": Path("artifacts/p2_deep_finalists_v1/stacked_oof.parquet"),
        "deep_stack_submission": Path("submissions/p2/P2_DEEP_STACK_V1.csv"),
    }
    for name, source in sources.items():
        if not source.is_file() or _sha256(source) != contract["sources"][name]:
            raise ValueError(f"P2 GBM source hash mismatch: {name}")
    return contract


def _phase_features(data):
    base = build_training_features(data.observations)
    dynamics = append_public_dynamics(base, data.observations)
    lean = select_lean_m2_dynamics(base, dynamics)
    phase = append_public_m2_harmonics(lean, data.observations)
    return phase


def _phase_test_features(data):
    base = build_test_features(data)
    dynamics = append_public_dynamics(base, data.observations)
    lean = select_lean_m2_dynamics(base, dynamics)
    phase = append_public_m2_harmonics(lean, data.observations)
    return phase


def _load_reusable_validation(
    root: Path, preregistration_sha256: str
) -> tuple[dict[str, object], pd.DataFrame] | None:
    summary_path = root / "validation.json"
    oof_path = root / "oof.parquet"
    if not summary_path.is_file() or not oof_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError(f"cached GBM validation has another contract: {root.name}")
    if summary.get("oof_sha256") != _sha256(oof_path):
        raise ValueError(f"cached GBM OOF hash mismatch: {root.name}")
    return summary["metrics"], pd.read_parquet(oof_path)


def _load_reusable_full(root: Path, preregistration_sha256: str) -> dict[str, object] | None:
    path = root / "full.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError(f"cached GBM full fit has another contract: {root.name}")
    for key in ("model", "submission"):
        item = value[key]
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"cached GBM {key} hash mismatch: {root.name}")
    return value


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(truth)) ** 2)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/experiments/p2_gbm_family_tournament_v1.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p2_gbm_family_tournament_v1")
    )
    parser.add_argument(
        "--status-file", type=Path, default=Path("artifacts/status/p2_gbm_tournament.json")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    progress = Progress(args.status_file, jobs=len(GBM_ARM_SPECS) * 2 + 3)
    progress.update("contract", "입력·OOF·제출 후보 SHA와 6개 구조 계약 검증", 0.0)
    data_dir = resolve_data_dir(args.data_dir)
    contract = _validate_contract(args.preregistration, data_dir)
    preregistration_sha256 = _sha256(args.preregistration)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_p2_data(data_dir)

    progress.finish()
    progress.update("features", "동일 public-only phase 81개 특징 생성", 0.0)
    phase = _phase_features(data)
    test_phase = _phase_test_features(data)
    if len(phase.feature_columns) != 81:
        raise ValueError(f"expected 81 frozen phase features, got {len(phase.feature_columns)}")
    progress.finish()

    deep_stack = pd.read_parquet("artifacts/p2_deep_finalists_v1/stacked_oof.parquet")
    deep_submission = pd.read_csv("submissions/p2/P2_DEEP_STACK_V1.csv")
    if (
        validate_submission(Path("submissions/p2/P2_DEEP_STACK_V1.csv"), data.test_index)["rows"]
        != 26_061
    ):
        raise ValueError("frozen deep-stack submission failed validation")

    summaries: dict[str, dict[str, object]] = {}
    paired_frames: dict[str, pd.DataFrame] = {}
    arm_submissions: dict[str, dict[str, object]] = {}
    for arm_number, spec in enumerate(GBM_ARM_SPECS, start=1):
        root = args.output_dir / spec.name
        root.mkdir(parents=True, exist_ok=True)
        progress.update(
            "blocked validation",
            f"{arm_number}/6 · {spec.name} · 3개 계절 블록",
            0.0,
        )
        reusable = _load_reusable_validation(root, preregistration_sha256)
        if reusable is None:
            arm_started = time.perf_counter()

            def fold_progress(
                position: int,
                total: int,
                block: str,
                arm_position: int = arm_number,
                arm_name: str = spec.name,
            ) -> None:
                progress.update(
                    "blocked validation",
                    f"{arm_position}/6 · {arm_name} · {position}/{total} {block}",
                    (position - 1) / total,
                )

            metrics, oof = run_blocked_arm(phase, spec, progress=fold_progress)
            metrics["elapsed_seconds"] = time.perf_counter() - arm_started
            oof_path = root / "oof.parquet"
            oof.to_parquet(oof_path, index=False, compression="zstd")
            _write_json(
                root / "validation.json",
                {
                    "preregistration_sha256": preregistration_sha256,
                    "metrics": metrics,
                    "oof_sha256": _sha256(oof_path),
                },
            )
        else:
            metrics, oof = reusable
        aligned = align_with_deep_stack(deep_stack, oof)
        pair = evaluate_deep_pair(aligned)
        paired = aligned.copy()
        paired["fitted_pair_prediction"] = pair.pop("fitted_prediction")
        paired["lobo_pair_prediction"] = pair.pop("lobo_prediction")
        bootstrap = paired_day_bootstrap(
            paired,
            paired["fitted_pair_prediction"].to_numpy(float),
            replicates=int(contract["comparison"]["bootstrap"]["replicates"]),
            seed=int(contract["comparison"]["bootstrap"]["seed"]),
        )
        paired_path = root / "paired_oof.parquet"
        paired.to_parquet(paired_path, index=False, compression="zstd")
        summaries[spec.name] = {
            **metrics,
            "pair_with_deep": pair,
            "bootstrap_vs_deep": bootstrap,
            "oof_sha256": _sha256(root / "oof.parquet"),
            "paired_oof_sha256": _sha256(paired_path),
        }
        paired_frames[spec.name] = paired
        progress.finish()

        progress.update("full fit", f"{arm_number}/6 · {spec.name} · 전체 학습과 test 추론", 0.0)
        reusable_full = _load_reusable_full(root, preregistration_sha256)
        if reusable_full is None:
            model = fit_gbm_model(phase, spec, seed=20260816)
            prediction = model.predict(test_phase)
            model_path = root / "model.joblib"
            joblib.dump(model, model_path, compress=3)
            restored = joblib.load(model_path)
            roundtrip = float(np.max(np.abs(restored.predict(test_phase) - prediction)))
            if roundtrip > 1e-12:
                raise RuntimeError(f"saved GBM model did not reproduce: {spec.name} {roundtrip}")
            submission_path = Path("submissions/p2") / f"P2_GBM_{spec.name.upper()}_V1.csv"
            submission_path.parent.mkdir(parents=True, exist_ok=True)
            build_submission(data.test_index, prediction).to_csv(
                submission_path, index=False, encoding="utf-8", lineterminator="\n"
            )
            validation = validate_submission(submission_path, data.test_index)
            reusable_full = {
                "preregistration_sha256": preregistration_sha256,
                "model": {"path": model_path.as_posix(), "sha256": _sha256(model_path)},
                "submission": {
                    "path": submission_path.as_posix(),
                    "sha256": _sha256(submission_path),
                    **validation,
                },
                "roundtrip_max_abs_error": roundtrip,
            }
            _write_json(root / "full.json", reusable_full)
        arm_submissions[spec.name] = reusable_full
        progress.finish()

    progress.update("selection", "standalone·Deep pair·LOBO·bootstrap 순위 검산", 0.0)
    ranking = sorted(
        (
            {
                "arm": name,
                "standalone_rmse": float(values["rmse"]),
                "fitted_pair_rmse": float(values["pair_with_deep"]["fitted_blend_rmse"]),
                "lobo_pair_rmse": float(values["pair_with_deep"]["lobo_blend_rmse"]),
                "fitted_delta_vs_deep": float(values["pair_with_deep"]["fitted_delta_vs_deep"]),
                "lobo_delta_vs_deep": float(values["pair_with_deep"]["lobo_delta_vs_deep_lobo"]),
            }
            for name, values in summaries.items()
        ),
        key=lambda row: (row["lobo_pair_rmse"], row["fitted_pair_rmse"], row["standalone_rmse"]),
    )
    selected = ranking[0]["arm"]
    selected_pair = summaries[selected]["pair_with_deep"]
    selected_submission = pd.read_csv(arm_submissions[selected]["submission"]["path"])
    keys = ["station", "layer", "time"]
    if not deep_submission[keys].equals(selected_submission[keys]):
        raise ValueError("deep and GBM submission keys differ")
    test_mix = pd.DataFrame(
        {
            "layer": deep_submission["layer"].to_numpy(int),
            "deep_prediction": deep_submission["temp"].to_numpy(float),
            "gbm_prediction": selected_submission["temp"].to_numpy(float),
        }
    )
    hybrid_prediction = blend_by_layer(
        test_mix,
        selected_pair["fitted_weights_by_layer"],
        reference_column="deep_prediction",
    )
    hybrid_path = Path("submissions/p2/P2_DEEP_GBM_RESEARCH_V1.csv")
    build_submission(data.test_index, hybrid_prediction).to_csv(
        hybrid_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    hybrid_validation = validate_submission(hybrid_path, data.test_index)
    selected_paired = paired_frames[selected]
    selected_bootstrap_lobo = paired_day_bootstrap(
        selected_paired,
        selected_paired["lobo_pair_prediction"].to_numpy(float),
        reference_column="deep_lobo_prediction",
        replicates=int(contract["comparison"]["bootstrap"]["replicates"]),
        seed=int(contract["comparison"]["bootstrap"]["seed"]),
    )
    progress.finish()

    truth = deep_stack["truth"].to_numpy(float)
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": contract["experiment_id"],
        "research_only": True,
        "uploaded": False,
        "elapsed_seconds": time.perf_counter() - started,
        "preregistration_sha256": preregistration_sha256,
        "git": _git_state(),
        "source_sha256": contract["sources"],
        "feature_contract": {
            "rows": len(phase.frame),
            "features": len(phase.feature_columns),
            "target_layers": list(map(int, sorted(phase.frame["layer"].unique()))),
            "target_layer_temp_or_psal_features": 0,
        },
        "oof_contract": {
            "rows": len(deep_stack),
            "blocks": list(VALIDATION_BLOCKS),
            "deep_stack_rmse": _rmse(truth, deep_stack["prediction"].to_numpy(float)),
            "deep_stack_lobo_rmse": _rmse(truth, deep_stack["lobo_prediction"].to_numpy(float)),
        },
        "arms": summaries,
        "ranking": ranking,
        "selected_for_parameter_search": selected,
        "eligible_for_parameter_search": sorted(
            {
                min(ranking, key=lambda row: row["standalone_rmse"])["arm"],
                *(row["arm"] for row in ranking if float(row["lobo_delta_vs_deep"]) < 0.0),
            }
        ),
        "selected_lobo_bootstrap_vs_deep_lobo": selected_bootstrap_lobo,
        "arm_full_outputs": arm_submissions,
        "hybrid_research_candidate": {
            "selected_arm": selected,
            "weights_by_layer": selected_pair["fitted_weights_by_layer"],
            "path": hybrid_path.as_posix(),
            "sha256": _sha256(hybrid_path),
            **hybrid_validation,
        },
        "decision": "STRUCTURE_SCREEN_COMPLETE_PARAMETER_SEARCH_NEXT_NO_UPLOAD",
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "result_sha256": _sha256(result_path),
            "preregistration_sha256": preregistration_sha256,
            "hybrid_submission_sha256": _sha256(hybrid_path),
            "uploaded": False,
        },
    )
    progress.update(
        "complete",
        f"6개 계열 완료 · 다음 튜닝 후보 {selected} · LOBO pair RMSE {ranking[0]['lobo_pair_rmse']:.6f} · 업로드 없음",
        1.0,
        status="complete",
    )
    print(
        json.dumps(
            {
                "ranking": ranking,
                "selected_for_parameter_search": selected,
                "eligible_for_parameter_search": result["eligible_for_parameter_search"],
                "hybrid_research_candidate": result["hybrid_research_candidate"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
