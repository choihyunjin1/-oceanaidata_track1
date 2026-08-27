"""Freeze the adaptive P2 layer-routed, physically projected soft-gate candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.profile_projection import project_profiles_vectorized, public_endpoint_frame
from p2_restore.submission import build_submission, validate_submission

SEED = 20260816


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(prediction)) ** 2)))


def _metrics(frame: pd.DataFrame, base: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    truth = frame["truth"].to_numpy(float)

    def cut(mask: np.ndarray) -> dict[str, float | int]:
        baseline = _rmse(truth[mask], base[mask])
        current = _rmse(truth[mask], candidate[mask])
        return {
            "rows": int(mask.sum()),
            "baseline_rmse": baseline,
            "candidate_rmse": current,
            "delta_rmse": current - baseline,
        }

    return {
        **cut(np.ones(len(frame), dtype=bool)),
        "by_block": {
            str(block): cut(frame["block"].eq(block).to_numpy())
            for block in frame["block"].unique()
        },
        "by_layer": {str(layer): cut(frame["layer"].eq(layer).to_numpy()) for layer in (2, 3, 4)},
    }


def _bootstrap(
    frame: pd.DataFrame,
    base: np.ndarray,
    candidate: np.ndarray,
    replicates: int = 2000,
) -> dict[str, object]:
    truth = frame["truth"].to_numpy(float)
    day = (
        pd.to_datetime(frame["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )
    unique = np.unique(day)
    blocks = [np.flatnonzero(day == value) for value in unique]
    rng = np.random.default_rng(SEED)
    delta = np.empty(replicates)
    for number in range(replicates):
        rows = np.concatenate(
            [blocks[index] for index in rng.integers(0, len(blocks), len(blocks))]
        )
        delta[number] = _rmse(truth[rows], candidate[rows]) - _rmse(truth[rows], base[rows])
    return {
        "replicates": replicates,
        "kst_days": len(unique),
        "delta_rmse": _rmse(truth, candidate) - _rmse(truth, base),
        "ci90_low": float(np.quantile(delta, 0.05)),
        "ci90_high": float(np.quantile(delta, 0.95)),
        "probability_improved": float(np.mean(delta < 0)),
    }


def _validate_contract(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != "p2_extrapolated_soft_gate_v1":
        raise ValueError("unexpected extrapolated-gate experiment id")
    if value.get("status") != "authorized_local_adaptive_score_optimization":
        raise ValueError("extrapolated-gate experiment is not locally authorized")
    if value.get("research_only") is not True or value.get("upload_allowed") is not False:
        raise ValueError("extrapolated-gate candidate must remain local-only")
    if value.get("adaptive_after_outer_exposure") is not True:
        raise ValueError("adaptive provenance must not be removed")
    if float(value["factor_scout"]["selected"]) != 2.0:
        raise ValueError("the frozen extrapolation factor changed")
    if value["layer_route"] != {"2": "raw_expert", "3": "base", "4": "raw_expert"}:
        raise ValueError("the frozen layer route changed")
    return value


def _compose(
    frame: pd.DataFrame,
    base: np.ndarray,
    raw: np.ndarray,
    endpoints: pd.DataFrame,
    factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    routed_input = np.asarray(base, dtype=float).copy()
    use_raw = frame["layer"].isin((2, 4)).to_numpy()
    routed_input[use_raw] = np.asarray(raw, dtype=float)[use_raw]
    routed = project_profiles_vectorized(frame, routed_input, endpoints).prediction
    final = project_profiles_vectorized(
        frame, base + factor * (routed - base), endpoints
    ).prediction
    return routed, final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/experiments/p2_extrapolated_soft_gate_v1.json"),
    )
    parser.add_argument(
        "--base-oof",
        type=Path,
        default=Path("artifacts/p2_physical_profile_projection_v1/oof.parquet"),
    )
    parser.add_argument(
        "--raw-oof",
        type=Path,
        default=Path("artifacts/p2_safe_residual_gate_v1/oof.parquet"),
    )
    parser.add_argument(
        "--base-submission",
        type=Path,
        default=Path("submissions/p2/P2_PHYSICAL_PROFILE_PROJECTION_V1.csv"),
    )
    parser.add_argument(
        "--raw-submission",
        type=Path,
        default=Path("submissions/p2/P2_PUBLIC_STATE_SOFT_GATE_V1.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p2_extrapolated_soft_gate_v1")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    contract = _validate_contract(args.contract)
    data = load_p2_data(resolve_data_dir(args.data_dir))
    endpoints = public_endpoint_frame(data.observations)
    base_oof = pd.read_parquet(args.base_oof)
    raw_oof = pd.read_parquet(args.raw_oof)
    base_oof["time"] = pd.to_datetime(base_oof["time"], utc=True)
    raw_oof["time"] = pd.to_datetime(raw_oof["time"], utc=True)
    frame = base_oof.merge(
        raw_oof[["time", "layer", "block", "raw_lambda10_prediction"]],
        on=["time", "layer", "block"],
        validate="one_to_one",
    )
    if len(frame) != 69_850 or frame.duplicated(["time", "layer"]).any():
        raise ValueError("OOF alignment or grain changed")
    base = frame["prediction"].to_numpy(float)
    raw = frame["raw_lambda10_prediction"].to_numpy(float)
    routed, candidate = _compose(frame, base, raw, endpoints, factor=2.0)
    metrics = _metrics(frame, base, candidate)
    bootstrap = _bootstrap(frame, base, candidate)

    base_submission = pd.read_csv(args.base_submission)
    raw_submission = pd.read_csv(args.raw_submission)
    key_columns = ["station", "layer", "time"]
    if not base_submission[key_columns].equals(raw_submission[key_columns]):
        raise ValueError("base and raw submission keys differ")
    if not all(
        np.array_equal(
            base_submission[column].astype(str).to_numpy(),
            data.test_index[column].astype(str).to_numpy(),
        )
        for column in key_columns
    ):
        raise ValueError("submission keys differ from test_index")
    _, test_prediction = _compose(
        base_submission,
        base_submission["temp"].to_numpy(float),
        raw_submission["temp"].to_numpy(float),
        endpoints,
        factor=2.0,
    )
    submission_path = Path(contract["submission"]["path"])
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    build_submission(data.test_index, test_prediction).to_csv(
        submission_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    validation = validate_submission(submission_path, data.test_index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.output_dir / "oof.parquet"
    saved = frame[["time", "layer", "truth", "block", "prediction"]].rename(
        columns={"prediction": "baseline"}
    )
    saved["raw_prediction"] = raw
    saved["routed_prediction"] = routed
    saved["prediction"] = candidate
    saved.to_parquet(oof_path, index=False, compression="zstd")
    decision = (
        "PROMOTE_LOCAL_SCORE_CANDIDATE_NO_UPLOAD"
        if metrics["delta_rmse"] < 0
        else "REJECT_KEEP_PHYSICAL_PROJECTION_NO_UPLOAD"
    )
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": contract["experiment_id"],
        "research_only": True,
        "adaptive_after_outer_exposure": True,
        "fresh_holdout_claimed": False,
        "external_values_used": False,
        "uploaded": False,
        "elapsed_seconds": time.perf_counter() - started,
        "factor": 2.0,
        "layer_route": contract["layer_route"],
        "metrics": metrics,
        "paired_kst_day_bootstrap": bootstrap,
        "decision": decision,
        "artifacts": {
            "oof": {"path": oof_path.as_posix(), "sha256": _sha256(oof_path)},
            "submission": {
                "path": submission_path.as_posix(),
                "sha256": _sha256(submission_path),
                **validation,
            },
        },
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "contract_sha256": _sha256(args.contract),
            "base_oof_sha256": _sha256(args.base_oof),
            "raw_oof_sha256": _sha256(args.raw_oof),
            "base_submission_sha256": _sha256(args.base_submission),
            "raw_submission_sha256": _sha256(args.raw_submission),
            "result_sha256": _sha256(result_path),
            "oof_sha256": _sha256(oof_path),
            "submission_sha256": _sha256(submission_path),
            "uploaded": False,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
