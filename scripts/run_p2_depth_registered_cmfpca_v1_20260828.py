"""Execute the single preregistered P2 depth-registered CMFPCA experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy
import sklearn

from p2_restore.data import KEYS, load_p2_data
from p2_restore.depth_registered_cmfpca import (
    ConditionalMFPCA,
    build_layer_identity_panel,
    evaluate_promotion_gate,
    paired_kst_day_bootstrap,
    predict_layer_identity_oas,
    prepare_complete_target_profiles,
    rmse,
)
from p2_restore.profile_projection import project_profiles_vectorized, public_endpoint_frame
from p2_restore.submission import build_submission, validate_submission

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/experiments/p2_depth_registered_cmfpca_v1_20260828.json"
DEFAULT_ARTIFACT = REPO / "artifacts/p2_depth_registered_cmfpca_v1_20260828"
OFFICIAL_GAP_START = pd.Timestamp("2025-09-01T00:00:00+09:00").tz_convert("UTC")
OFFICIAL_GAP_STOP = pd.Timestamp("2025-11-01T00:00:00+09:00").tz_convert("UTC")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--base-csv", type=Path, default=None)
    parser.add_argument("--ready-root", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def resolve_optional_path(argument: Path | None, environment: str) -> Path | None:
    raw = argument if argument is not None else os.environ.get(environment)
    return Path(raw).expanduser().resolve() if raw else None


def git_snapshot() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status_short": status}


def source_score_input_contract(frame: pd.DataFrame, test_index: pd.DataFrame) -> dict[str, object]:
    required = KEYS + ["temp"]
    require(list(frame.columns) == required, "source score.py submission columns differ")
    require(len(frame) == len(test_index), "source score.py submission row count differs")
    require(not frame[KEYS].isna().any().any(), "source score.py key null")
    require(not frame.duplicated(KEYS).any(), "source score.py duplicate key")
    left = test_index[KEYS].merge(frame[KEYS], how="outer", indicator=True, validate="one_to_one")
    require(left["_merge"].eq("both").all(), "source score.py key set differs")
    values = pd.to_numeric(frame["temp"], errors="coerce").to_numpy(float)
    require(np.isfinite(values).all(), "source score.py non-finite prediction")
    require(np.all((values >= -5.0) & (values <= 45.0)), "source score.py range failure")
    return {
        "replicated_submission_side_only": True,
        "answer_file_read": False,
        "rows": int(len(frame)),
        "columns": required,
        "key_set_match": True,
        "finite": True,
        "range_c": [float(values.min()), float(values.max())],
    }


def fold_metric(frame: pd.DataFrame, group: str) -> dict[str, object]:
    reference = rmse(frame["truth"].to_numpy(), frame["oas_alpha20"].to_numpy())
    candidate = rmse(frame["truth"].to_numpy(), frame["cmfpca_alpha20"].to_numpy())
    return {
        "rows": int(len(frame)),
        "oas_alpha20_rmse": reference,
        "cmfpca_alpha20_rmse": candidate,
        "delta_rmse": candidate - reference,
        "group": group,
    }


def main() -> None:
    args = parse_args()
    require(args.execute, "this one-shot experiment requires --execute")
    config_path = args.config.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    require(not artifact_dir.exists(), f"append-only artifact already exists: {artifact_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["experiment_id"] == "p2_depth_registered_cmfpca_v1_20260828", "config drift")
    require(config["leakage_contract"]["official_upload_authorized"] is False, "upload flag")
    data_dir = resolve_optional_path(args.data_dir, "P2_DATA_DIR")
    require(data_dir is not None, "set P2_DATA_DIR or pass --data-dir")
    base_csv = resolve_optional_path(args.base_csv, "P2_CMFPCA_BASE_CSV")
    ready_root = resolve_optional_path(args.ready_root, "P2_SUBMISSION_READY_ROOT")

    input_pins = config["input_pins"]
    for filename, key in (
        ("observations.csv", "observations_sha256"),
        ("test_index.csv", "test_index_sha256"),
        ("sample_submission.csv", "sample_submission_sha256"),
        ("score.py", "score_py_sha256"),
    ):
        require(sha256(data_dir / filename) == input_pins[key], f"input pin changed: {filename}")
    anchor_path = REPO / config["validation_anchor"]["path"]
    require(sha256(anchor_path) == config["validation_anchor"]["sha256"], "OOF anchor pin changed")

    started = datetime.now(ZoneInfo("Asia/Seoul"))
    data = load_p2_data(data_dir)
    observations = data.observations.copy()
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    profiles = prepare_complete_target_profiles(
        observations,
        coefficient_ridge=float(config["model"]["coefficient_ridge"]),
    )
    layer_panel, _, _ = build_layer_identity_panel(observations)
    endpoints = public_endpoint_frame(observations)
    anchor = pd.read_parquet(anchor_path)
    anchor["time"] = pd.to_datetime(anchor["time"], utc=True)
    required_anchor = {"time", "layer", "truth", "block", config["validation_anchor"]["prediction_column"]}
    require(required_anchor.issubset(anchor.columns), "validation anchor schema differs")
    require(not anchor.duplicated(["time", "layer"]).any(), "validation anchor keys duplicate")

    observation_targets = observations.loc[
        observations["layer"].isin([2, 3, 4]),
        ["station", "time", "layer", "nominal_depth", "temp"],
    ].rename(columns={"temp": "observation_truth"})
    require(not observation_targets.duplicated(["time", "layer"]).any(), "observation target keys duplicate")
    model_config = config["model"]
    fold_frames: list[pd.DataFrame] = []
    fit_receipts: dict[str, object] = {}

    for fold_name, (start_text, stop_text) in config["folds"].items():
        start = pd.Timestamp(start_text).tz_convert("UTC")
        stop = pd.Timestamp(stop_text).tz_convert("UTC")
        fold_anchor = anchor.loc[anchor["block"].astype(str) == fold_name].copy()
        require(len(fold_anchor) > 0, f"anchor lacks fold {fold_name}")
        query = fold_anchor.loc[:, ["time", "layer", "truth", config["validation_anchor"]["prediction_column"]]].merge(
            observation_targets,
            on=["time", "layer"],
            how="inner",
            validate="one_to_one",
        )
        require(len(query) == len(fold_anchor), f"observation alignment lost rows in {fold_name}")
        require(
            np.max(np.abs(query["truth"].to_numpy(float) - query["observation_truth"].to_numpy(float)))
            <= 1e-12,
            f"anchor truth does not reproduce source observations in {fold_name}",
        )
        query = query.reset_index(drop=True)
        train_mask = (profiles.times < start) | (profiles.times >= stop)
        model = ConditionalMFPCA.fit(
            profiles,
            train_mask,
            variance_threshold=float(model_config["variance_threshold"]),
            rank_cap=int(model_config["rank_cap"]),
            noise_floor=float(model_config["noise_floor"]),
        )
        prediction_cmfpca = model.predict(observations, query)
        prediction_oas, oas_receipts = predict_layer_identity_oas(
            layer_panel,
            query,
            exclude_start=start,
            exclude_stop=stop,
        )
        base = query[config["validation_anchor"]["prediction_column"]].to_numpy(float)
        alpha = float(model_config["blend_weight"])
        oas_blend = base + alpha * (prediction_oas - base)
        cmfpca_blend = base + alpha * (prediction_cmfpca - base)
        projection_frame = query.loc[:, ["station", "time", "layer"]]
        projected_oas = project_profiles_vectorized(projection_frame, oas_blend, endpoints)
        projected_cmfpca = project_profiles_vectorized(projection_frame, cmfpca_blend, endpoints)
        scored = query.loc[:, ["station", "time", "layer", "truth", "nominal_depth"]].copy()
        scored["fold"] = fold_name
        scored["anchor"] = base
        scored["oas_raw"] = prediction_oas
        scored["cmfpca_raw"] = prediction_cmfpca
        scored["oas_alpha20"] = projected_oas.prediction
        scored["cmfpca_alpha20"] = projected_cmfpca.prediction
        fold_frames.append(scored)
        fit_receipts[fold_name] = {
            "cmfpca": model.receipt(),
            "layer_id_oas_bins": oas_receipts,
            "oas_projection": projected_oas.diagnostics(),
            "cmfpca_projection": projected_cmfpca.diagnostics(),
        }

    oof = pd.concat(fold_frames, ignore_index=True)
    aggregate = fold_metric(oof, "aggregate")
    by_fold = {
        str(name): fold_metric(group, str(name)) for name, group in oof.groupby("fold", sort=True)
    }
    by_layer = {
        str(int(name)): fold_metric(group, str(int(name)))
        for name, group in oof.groupby("layer", sort=True)
    }
    bootstrap = paired_kst_day_bootstrap(
        oof,
        reference="oas_alpha20",
        candidate="cmfpca_alpha20",
        replicates=int(model_config["bootstrap_replicates"]),
        seed=int(model_config["bootstrap_seed"]),
    )
    gate = evaluate_promotion_gate(
        aggregate_delta=float(aggregate["delta_rmse"]),
        bootstrap_ci90_high=float(bootstrap["ci90_high"]),
        fold_deltas={name: float(value["delta_rmse"]) for name, value in by_fold.items()},
        layer_deltas={name: float(value["delta_rmse"]) for name, value in by_layer.items()},
        thresholds=config["gate"],
    )

    candidate: pd.DataFrame | None = None
    candidate_diagnostics: dict[str, object] | None = None
    ready_dir: Path | None = None
    if gate["passed"]:
        require(base_csv is not None, "passing gate requires --base-csv or P2_CMFPCA_BASE_CSV")
        require(ready_root is not None, "passing gate requires --ready-root or P2_SUBMISSION_READY_ROOT")
        require(sha256(base_csv) == input_pins["deployment_base_u_sha256"], "deployment U pin changed")
        ready_dir = ready_root / config["candidate_on_pass"]["ready_directory_name"]
        require(not ready_dir.exists(), f"append-only ready directory exists: {ready_dir}")
        deployment_mask = (profiles.times < OFFICIAL_GAP_START) | (profiles.times >= OFFICIAL_GAP_STOP)
        deployment_model = ConditionalMFPCA.fit(
            profiles,
            deployment_mask,
            variance_threshold=float(model_config["variance_threshold"]),
            rank_cap=int(model_config["rank_cap"]),
            noise_floor=float(model_config["noise_floor"]),
        )
        raw = deployment_model.predict(observations, data.test_index)
        base = pd.read_csv(base_csv, dtype={"station": "string", "time": "string"})
        require(list(base.columns) == KEYS + ["temp"], "deployment U schema differs")
        require(base[KEYS].equals(data.test_index[KEYS]), "deployment U key/order differs")
        base_values = base["temp"].to_numpy(float)
        blended = base_values + float(model_config["blend_weight"]) * (raw - base_values)
        projected = project_profiles_vectorized(data.test_index, blended, endpoints)
        candidate = build_submission(data.test_index, projected.prediction)
        validate_submission(candidate, data.test_index)
        score_contract = source_score_input_contract(candidate, data.test_index)
        second_projection = project_profiles_vectorized(
            data.test_index, candidate["temp"].to_numpy(float), endpoints
        )
        idempotence = float(
            np.max(np.abs(second_projection.prediction - candidate["temp"].to_numpy(float)))
        )
        require(
            idempotence <= float(config["candidate_on_pass"]["pava_idempotence_tolerance"]),
            "candidate PAVA is not idempotent",
        )
        difference = candidate["temp"].to_numpy(float) - base_values
        candidate_diagnostics = {
            "deployment_model": deployment_model.receipt(),
            "projection": projected.diagnostics(),
            "pava_idempotence_max_abs": idempotence,
            "source_score_py_submission_contract": score_contract,
            "difference_vs_base": {
                "changed_rows": int(np.sum(np.abs(difference) > 1e-12)),
                "rms": float(np.sqrt(np.mean(difference**2))),
                "maximum_absolute": float(np.max(np.abs(difference))),
                "by_layer": {
                    str(layer): {
                        "rows": int(np.sum(data.test_index["layer"].to_numpy(int) == layer)),
                        "changed_rows": int(
                            np.sum(
                                (data.test_index["layer"].to_numpy(int) == layer)
                                & (np.abs(difference) > 1e-12)
                            )
                        ),
                        "rms": float(
                            np.sqrt(
                                np.mean(
                                    difference[data.test_index["layer"].to_numpy(int) == layer] ** 2
                                )
                            )
                        ),
                    }
                    for layer in (2, 3, 4)
                },
            },
        }

    completed = datetime.now(ZoneInfo("Asia/Seoul"))
    decision = "PASS_BUILD_LOCAL_CANDIDATE_NO_UPLOAD" if gate["passed"] else "FAIL_GATE_NO_CANDIDATE"
    result = {
        "schema_version": "p2.depth_registered_cmfpca.result.20260828.v1",
        "experiment_id": config["experiment_id"],
        "execution_count": 1,
        "decision": decision,
        "historical_exposure": config["historical_exposure"],
        "reference": (
            "Fold-local frozen public-only current_blend50 anchor; both layer-ID OAS20 and "
            "depth-registered CMFPCA20 use the identical anchor and endpoint/PAVA projection."
        ),
        "metrics": {"aggregate": aggregate, "by_fold": by_fold, "by_layer": by_layer},
        "paired_kst_day_bootstrap": bootstrap,
        "gate": gate,
        "fit_receipts": fit_receipts,
        "candidate_diagnostics": candidate_diagnostics,
        "leakage_audit": {
            "answer_file_read": False,
            "hidden_answer_or_mirror_read": False,
            "official_gap_target_value_reads": 0,
            "target_temp_psal_masked_together": True,
            "official_upload_performed": False,
            "post_result_parameter_search_performed": False,
        },
        "runtime": {
            "started_at_kst": started.isoformat(),
            "completed_at_kst": completed.isoformat(),
            "elapsed_seconds": (completed - started).total_seconds(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "git": git_snapshot(),
        },
    }

    artifact_dir.mkdir(parents=True)
    oof_path = artifact_dir / "oof.parquet"
    result_path = artifact_dir / "result.json"
    oof.to_parquet(oof_path, index=False)
    if candidate is not None and ready_dir is not None:
        candidate_dir = artifact_dir / "candidate"
        candidate_dir.mkdir()
        ready_dir.mkdir(parents=True)
        canonical = candidate_dir / "P2_submission.csv"
        ready = ready_dir / "P2_submission.csv"
        candidate.to_csv(canonical, index=False, encoding="utf-8", lineterminator="\n")
        candidate.to_csv(ready, index=False, encoding="utf-8", lineterminator="\n")
        require(canonical.read_bytes() == ready.read_bytes(), "ready candidate copy differs")
        validate_submission(canonical, data.test_index)
        result["candidate"] = {
            "canonical_path": str(canonical),
            "ready_path": str(ready),
            "bytes": canonical.stat().st_size,
            "sha256": sha256(canonical),
        }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    tracked_sources = [
        config_path,
        REPO / "src/p2_restore/depth_registered_cmfpca.py",
        Path(__file__).resolve(),
        REPO / "scripts/qa_p2_depth_registered_cmfpca_v1_20260828.py",
        REPO / "tests/test_p2_depth_registered_cmfpca.py",
    ]
    manifest = {
        "schema_version": "p2.depth_registered_cmfpca.manifest.20260828.v1",
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "sources": {str(path.relative_to(REPO)): sha256(path) for path in tracked_sources},
        "inputs": {
            "observations": {"path": str(data_dir / "observations.csv"), "sha256": sha256(data_dir / "observations.csv")},
            "test_index": {"path": str(data_dir / "test_index.csv"), "sha256": sha256(data_dir / "test_index.csv")},
            "sample_submission": {"path": str(data_dir / "sample_submission.csv"), "sha256": sha256(data_dir / "sample_submission.csv")},
            "score_py": {"path": str(data_dir / "score.py"), "sha256": sha256(data_dir / "score.py")},
            "validation_anchor": {"path": str(anchor_path), "sha256": sha256(anchor_path)},
        },
        "outputs": {
            "result": {"path": str(result_path), "sha256": sha256(result_path)},
            "oof": {"path": str(oof_path), "sha256": sha256(oof_path), "rows": int(len(oof))},
            "candidate": result.get("candidate"),
        },
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "metrics": result["metrics"], "gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
