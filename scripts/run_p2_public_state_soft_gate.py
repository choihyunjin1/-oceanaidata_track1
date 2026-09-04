"""Run the authorized full P2 public-state soft-gate experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.deep_data import build_panel
from p2_restore.deep_training import predict_full_checkpoint
from p2_restore.regime_gate import (
    STATE_FEATURES,
    build_public_state_features,
    nested_lobo_soft_gate,
    predict_soft_gate,
    soft_gate_weights,
)
from p2_restore.submission import build_submission, validate_submission

CONTRIBUTORS = (
    "router_400",
    "depth_query_bitcn",
    "lsti_style",
    "timemixerpp_style",
    "moment_units_scratch",
)
OOF_PATHS = {
    "router_400": Path("artifacts/p2_max_round_convergence_v1/oof.parquet"),
    "depth_query_bitcn": Path(
        "artifacts/p2_deep_model_tournament_v1/models/depth_query_bitcn/oof.parquet"
    ),
    "moment_units_scratch": Path(
        "artifacts/p2_deep_model_tournament_v1/models/moment_units_scratch/oof.parquet"
    ),
    "lsti_style": Path("artifacts/p2_deep_finalists_v1/lsti_style/ensemble_oof.parquet"),
    "timemixerpp_style": Path(
        "artifacts/p2_deep_finalists_v1/timemixerpp_style/ensemble_oof.parquet"
    ),
}


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
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.perf_counter()

    def update(self, progress: float, phase: str, detail: str, *, status: str = "running") -> None:
        elapsed = time.perf_counter() - self.started
        bounded = min(max(progress, 0.1), 100.0)
        remaining = elapsed * (100.0 - bounded) / bounded if bounded < 100 else 0.0
        eta = datetime.now().astimezone() + timedelta(seconds=max(remaining, 0.0))
        _write_json(
            self.path,
            {
                "title": "P2 공개층 물리상태 Soft Gate 풀테스트",
                "status": status,
                "progress": bounded,
                "phase": phase,
                "detail": detail,
                "eta": eta.strftime("%Y-%m-%d %H:%M:%S KST"),
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )


def _validate_contract(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != "p2_public_state_soft_gate_v1":
        raise ValueError("unexpected public-state gate experiment id")
    if value.get("status") != "authorized_local_full_test":
        raise ValueError("public-state full test is not authorized")
    if value.get("research_only") is not True or value.get("upload_allowed") is not False:
        raise ValueError("public-state experiment must remain local-only")
    if tuple(value.get("contributors", ())) != CONTRIBUTORS:
        raise ValueError("frozen contributor list changed")
    if tuple(value.get("public_state_features", ())) != STATE_FEATURES:
        raise ValueError("public-state feature contract changed")
    hashes = value["frozen_inputs"]
    checks = {
        "router_oof_sha256": OOF_PATHS["router_400"],
        "depth_query_oof_sha256": OOF_PATHS["depth_query_bitcn"],
        "moment_oof_sha256": OOF_PATHS["moment_units_scratch"],
        "lsti_oof_sha256": OOF_PATHS["lsti_style"],
        "timemixer_oof_sha256": OOF_PATHS["timemixerpp_style"],
        "deep_stacked_oof_sha256": Path("artifacts/p2_deep_finalists_v1/stacked_oof.parquet"),
        "deep_result_sha256": Path("artifacts/p2_deep_finalists_v1/result.json"),
        "deep_submission_sha256": Path("submissions/p2/P2_DEEP_STACK_V1.csv"),
    }
    for key, source in checks.items():
        if _sha256(source) != hashes[key]:
            raise ValueError(f"frozen input changed: {source}")
    return value


def _load_oof_stack() -> pd.DataFrame:
    base = pd.read_parquet(OOF_PATHS["router_400"])[
        ["time", "layer", "truth", "block", "router_400"]
    ].copy()
    base["time"] = pd.to_datetime(base["time"], utc=True)
    keys = ["time", "layer", "block"]
    for name in CONTRIBUTORS[1:]:
        current = pd.read_parquet(OOF_PATHS[name])[
            ["time", "layer", "truth", "block", "prediction"]
        ]
        current["time"] = pd.to_datetime(current["time"], utc=True)
        current = current.rename(columns={"prediction": name, "truth": "truth_check"})
        base = base.merge(current, on=keys, how="inner", validate="one_to_one")
        if not np.allclose(base.pop("truth_check"), base["truth"], rtol=0, atol=0):
            raise ValueError(f"OOF truth differs for {name}")
    if len(base) != 69_850 or base[keys].duplicated().any():
        raise ValueError("frozen contributor OOF grain changed")
    return base


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((np.asarray(prediction) - np.asarray(truth)) ** 2) ** 0.5)


def _metrics(frame: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    truth = frame["truth"].to_numpy(float)
    result: dict[str, object] = {
        "rows": len(frame),
        "baseline_rmse": _rmse(truth, baseline),
        "candidate_rmse": _rmse(truth, candidate),
    }
    result["delta_rmse"] = result["candidate_rmse"] - result["baseline_rmse"]
    result["by_block"] = {}
    for block in sorted(frame["block"].unique()):
        keep = frame["block"].to_numpy() == block
        result["by_block"][block] = {
            "rows": int(keep.sum()),
            "baseline_rmse": _rmse(truth[keep], baseline[keep]),
            "candidate_rmse": _rmse(truth[keep], candidate[keep]),
        }
        result["by_block"][block]["delta_rmse"] = (
            result["by_block"][block]["candidate_rmse"] - result["by_block"][block]["baseline_rmse"]
        )
    result["by_layer"] = {}
    for layer in (2, 3, 4):
        keep = frame["layer"].to_numpy(int) == layer
        result["by_layer"][str(layer)] = {
            "rows": int(keep.sum()),
            "baseline_rmse": _rmse(truth[keep], baseline[keep]),
            "candidate_rmse": _rmse(truth[keep], candidate[keep]),
        }
        result["by_layer"][str(layer)]["delta_rmse"] = (
            result["by_layer"][str(layer)]["candidate_rmse"]
            - result["by_layer"][str(layer)]["baseline_rmse"]
        )
    return result


def _paired_day_bootstrap(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    replicates: int = 2000,
) -> dict[str, object]:
    day = (
        pd.to_datetime(frame["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )
    unique = np.unique(day)
    blocks = [np.flatnonzero(day == value) for value in unique]
    truth = frame["truth"].to_numpy(float)
    rng = np.random.default_rng(20260816)
    deltas = np.empty(replicates)
    for number in range(replicates):
        chosen = rng.integers(0, len(blocks), len(blocks))
        index = np.concatenate([blocks[position] for position in chosen])
        deltas[number] = _rmse(truth[index], candidate[index]) - _rmse(
            truth[index], baseline[index]
        )
    return {
        "replicates": replicates,
        "kst_days": len(unique),
        "delta_rmse": _rmse(truth, candidate) - _rmse(truth, baseline),
        "ci90_low": float(np.quantile(deltas, 0.05)),
        "ci90_high": float(np.quantile(deltas, 0.95)),
        "probability_improved": float(np.mean(deltas < 0)),
    }


def _panel_test_prediction(
    index: pd.DataFrame, panel_times: pd.DatetimeIndex, prediction: np.ndarray
) -> np.ndarray:
    positions = panel_times.get_indexer(pd.to_datetime(index["time"], utc=True))
    if (positions < 0).any():
        raise ValueError("test_index time is absent from the deep panel")
    return prediction[positions, index["layer"].to_numpy(int) - 2]


def _full_test_components(
    data: object, deep_result: dict[str, object], progress: Progress
) -> pd.DataFrame:
    panel = build_panel(data.observations)
    values: dict[str, np.ndarray] = {
        "router_400": pd.read_csv("submissions/p2/P2_SCORE_ROUTER_ROUND400.csv")["temp"].to_numpy(
            float
        )
    }
    model_names = ("depth_query_bitcn", "lsti_style", "timemixerpp_style", "moment_units_scratch")
    total = sum(len(deep_result["full_models"][name]) for name in model_names)
    completed = 0
    for name in model_names:
        predictions = []
        for entry in deep_result["full_models"][name]:
            checkpoint = Path(entry["checkpoint"])
            if _sha256(checkpoint) != entry["checkpoint_sha256"]:
                raise ValueError(f"deep checkpoint hash changed: {checkpoint}")
            progress.update(
                50 + 28 * completed / total,
                "동결 deep checkpoint 재추론",
                f"{name} · seed {entry['seed']} ({completed + 1}/{total})",
            )
            panel_prediction = predict_full_checkpoint(checkpoint, panel)
            predictions.append(
                _panel_test_prediction(data.test_index, panel.times, panel_prediction)
            )
            completed += 1
        values[name] = np.mean(predictions, axis=0)
    return pd.DataFrame(
        {
            "time": pd.to_datetime(data.test_index["time"], utc=True),
            "layer": data.test_index["layer"].to_numpy(int),
            **values,
        }
    )


def _weighted_stack(frame: pd.DataFrame, weights: dict[str, dict[str, float]]) -> np.ndarray:
    prediction = np.full(len(frame), np.nan)
    for layer in (2, 3, 4):
        keep = frame["layer"].to_numpy(int) == layer
        vector = np.array([weights[str(layer)][name] for name in CONTRIBUTORS])
        prediction[keep] = frame.loc[keep, CONTRIBUTORS].to_numpy(float) @ vector
    return prediction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/experiments/p2_public_state_soft_gate_v1.json"),
    )
    parser.add_argument(
        "--status-file", type=Path, default=Path("artifacts/status/p2_public_state_soft_gate.json")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    progress = Progress(args.status_file)
    progress.update(2, "계약·해시 확인", "동결 입력 8개와 실행 계약 검증")
    contract = _validate_contract(args.preregistration)
    data_dir = resolve_data_dir(args.data_dir)
    data = load_p2_data(data_dir)
    output_dir = Path(contract["outputs"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    progress.update(8, "OOF 정렬", "5개 동결 contributor · 69,850행 exact-key merge")
    stack = _load_oof_stack()
    public_features = build_public_state_features(data.observations, stack[["time", "layer"]])
    if (
        not stack[["time", "layer"]]
        .reset_index(drop=True)
        .equals(public_features[["time", "layer"]].reset_index(drop=True))
    ):
        raise ValueError("public-state OOF features are not aligned")
    stack = pd.concat(
        [
            stack.reset_index(drop=True),
            public_features.loc[:, STATE_FEATURES].reset_index(drop=True),
        ],
        axis=1,
    )

    progress.update(18, "Nested soft-gate 선택", "3 outer × 7 정규화 · outer label 선택 금지")
    regularization_grid = tuple(float(value) for value in contract["gate"]["regularization_grid"])
    nested = nested_lobo_soft_gate(
        stack,
        regularization_grid=regularization_grid,
        prediction_columns=CONTRIBUTORS,
    )
    frozen = pd.read_parquet("artifacts/p2_deep_finalists_v1/stacked_oof.parquet")
    frozen["time"] = pd.to_datetime(frozen["time"], utc=True)
    frozen = stack[["time", "layer", "block"]].merge(
        frozen[["time", "layer", "block", "lobo_prediction"]],
        on=["time", "layer", "block"],
        validate="one_to_one",
    )
    baseline_error = float(
        np.max(np.abs(nested.baseline_prediction - frozen["lobo_prediction"].to_numpy(float)))
    )
    if baseline_error > 1e-6:
        raise ValueError(f"frozen LOBO stack reconstruction failed: {baseline_error}")
    metrics = _metrics(stack, nested.baseline_prediction, nested.prediction)
    bootstrap = _paired_day_bootstrap(stack, nested.baseline_prediction, nested.prediction)

    progress.update(
        46,
        "Outer 평가 완료",
        f"candidate {metrics['candidate_rmse']:.6f} · deep LOBO {metrics['baseline_rmse']:.6f}",
    )
    deep_result_path = Path("artifacts/p2_deep_finalists_v1/result.json")
    deep_result = json.loads(deep_result_path.read_text(encoding="utf-8"))
    test_components = _full_test_components(data, deep_result, progress)
    frozen_submission = pd.read_csv("submissions/p2/P2_DEEP_STACK_V1.csv")
    reconstructed = _weighted_stack(test_components, deep_result["weights_by_layer"])
    test_reproduction_error = float(
        np.max(np.abs(reconstructed - frozen_submission["temp"].to_numpy(float)))
    )
    if test_reproduction_error > 1e-5:
        raise ValueError(f"frozen deep submission reconstruction failed: {test_reproduction_error}")

    progress.update(82, "Hidden 공개층 gate 추론", "26,061행 · target temp/psal 미사용")
    test_features = build_public_state_features(
        data.observations, test_components[["time", "layer"]]
    )
    test_stack = pd.concat(
        [
            test_components.reset_index(drop=True),
            test_features.loc[:, STATE_FEATURES].reset_index(drop=True),
        ],
        axis=1,
    )
    test_prediction = predict_soft_gate(nested.final_gate, test_stack)
    submission_path = Path(contract["outputs"]["submission"])
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    build_submission(data.test_index, test_prediction).to_csv(
        submission_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    submission_validation = validate_submission(submission_path, data.test_index)

    progress.update(88, "산출물 저장", "OOF·gate model·test components·submission")
    oof_output = stack[["time", "layer", "truth", "block"]].copy()
    oof_output["baseline_prediction"] = nested.baseline_prediction
    oof_output["prediction"] = nested.prediction
    oof_path = output_dir / "oof.parquet"
    oof_output.to_parquet(oof_path, index=False, compression="zstd")
    components_path = output_dir / "test_components.parquet"
    test_components.to_parquet(components_path, index=False, compression="zstd")
    model_path = output_dir / "gate_model.joblib"
    joblib.dump(nested.final_gate, model_path, compress=3)
    restored_gate = joblib.load(model_path)
    roundtrip_error = float(
        np.max(np.abs(predict_soft_gate(restored_gate, test_stack) - test_prediction))
    )
    weights = soft_gate_weights(nested.final_gate, stack)
    weight_summary = {
        str(layer): {
            name: float(weights[stack["layer"].to_numpy(int) == layer, number].mean())
            for number, name in enumerate(CONTRIBUTORS)
        }
        for layer in (2, 3, 4)
    }
    decision = (
        "PROVISIONAL_CHALLENGER_NO_UPLOAD"
        if metrics["candidate_rmse"] < metrics["baseline_rmse"]
        else "REJECT_KEEP_DEEP_STACK"
    )
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": contract["experiment_id"],
        "research_only": True,
        "uploaded": False,
        "elapsed_seconds": time.perf_counter() - started,
        "preregistration_sha256": _sha256(args.preregistration),
        "target_layer_temp_or_psal_used_by_gate": False,
        "external_values_used": False,
        "contributors": list(CONTRIBUTORS),
        "features": list(STATE_FEATURES),
        "selected_regularization_by_outer": nested.selected_regularization,
        "inner_scores": nested.inner_scores,
        "final_regularization": nested.final_regularization,
        "baseline_reproduction_max_abs_error": baseline_error,
        "test_deep_reproduction_max_abs_error": test_reproduction_error,
        "model_roundtrip_max_abs_error": roundtrip_error,
        "metrics": metrics,
        "paired_kst_day_bootstrap": bootstrap,
        "mean_weights_by_layer": weight_summary,
        "submission_validation": submission_validation,
        "decision": decision,
        "artifacts": {
            "oof": {"path": oof_path.as_posix(), "sha256": _sha256(oof_path)},
            "test_components": {
                "path": components_path.as_posix(),
                "sha256": _sha256(components_path),
            },
            "model": {"path": model_path.as_posix(), "sha256": _sha256(model_path)},
            "submission": {
                "path": submission_path.as_posix(),
                "sha256": _sha256(submission_path),
            },
        },
    }
    result_path = output_dir / "result.json"
    _write_json(result_path, result)
    _write_json(
        output_dir / "manifest.json",
        {
            "result_sha256": _sha256(result_path),
            "preregistration_sha256": _sha256(args.preregistration),
            "oof_sha256": _sha256(oof_path),
            "model_sha256": _sha256(model_path),
            "submission_sha256": _sha256(submission_path),
            "uploaded": False,
        },
    )
    progress.update(
        100,
        "완료",
        f"{decision} · ΔRMSE {metrics['delta_rmse']:+.6f}℃ · 업로드 없음",
        status="complete",
    )
    print(json.dumps({"status": "passed", "decision": decision, "metrics": metrics}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
