#!/usr/bin/env python3
"""Run the preregistered P3 perfect-future-wind one-shot experiment.

Only historical train data and already-sealed historical OOF artifacts are read.
The runner never resolves, lists, or opens any official test/sample/submission file.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/p3_perfect_future_wind_oracle_20260829_v1.json"
KEYS = ["fold", "anchor_id", "station"]
ROW_KEYS = [*KEYS, "lead_h"]


class ContractError(RuntimeError):
    """Raised when the frozen one-shot contract is not exactly satisfied."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ContractError(f"path escapes repository root: {relative}") from exc
    return candidate


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    _exclusive_bytes(path, encoded + b"\n")


def exclusive_parquet(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    _exclusive_bytes(path, buffer.getvalue())


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, encoding="utf-8"
    ).strip()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "experiment_id",
        "base_commit",
        "data_boundary",
        "inputs",
        "surface",
        "oracle",
        "conditional_future_wind",
        "conditional_mos",
        "one_shot",
    }
    if set(config) < required:
        raise ContractError(f"config is missing keys: {sorted(required - set(config))}")
    boundary = config["data_boundary"]
    if boundary != {
        "historical_train_only": True,
        "allowed_source_file": "train_atmos.csv",
        "forbidden_source_names": [
            "test_context.parquet",
            "test_index.csv",
            "sample_submission.csv",
            "baseline_persistence.csv",
            "score.py",
        ],
        "official_values_read_allowed": False,
        "absolute_official_time_reconstruction_allowed": False,
        "external_period_matching_allowed": False,
        "csv_output_allowed": False,
        "upload_allowed": False,
        "source_mutation_allowed": False,
    }:
        raise ContractError("data boundary changed")
    surface = config["surface"]
    if surface["expected_cases"] != 179 or surface["expected_rows"] != 1074:
        raise ContractError("179-case surface changed")
    if surface["active_leads_h"] != [18, 24]:
        raise ContractError("active leads changed")
    if surface["exact_no_op_leads_h"] != [3, 6, 9, 12]:
        raise ContractError("exact no-op leads changed")
    if surface["embargo_hours"] != 78:
        raise ContractError("78h embargo changed")
    if surface["frozen_kma_prediction_column"] != "candidate_final":
        raise ContractError("frozen KMA prediction column changed")
    return config


def _input_paths(config: dict[str, Any], data_dir: Path) -> dict[str, Path]:
    paths = {
        name: _contained(ROOT, item["path"])
        for name, item in config["inputs"].items()
        if name != "train_atmos"
    }
    paths["train_atmos"] = (data_dir.resolve() / "train_atmos.csv").resolve()
    if paths["train_atmos"].parent != data_dir.resolve():
        raise ContractError("historical atmosphere path escaped the explicit data root")
    return paths


def _output_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {name: _contained(ROOT, value) for name, value in config["one_shot"].items()}


def _verify_hashes(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ContractError(f"required historical input is missing: {name}")
        actual = sha256_file(path)
        expected = config["inputs"][name]["sha256"]
        if actual != expected:
            raise ContractError(f"historical input hash changed: {name}")
        observed[name] = actual
    return observed


def _read_key_frame(path: Path) -> pd.DataFrame:
    schema = set(pq.read_schema(path).names)
    if not set(KEYS).issubset(schema):
        raise ContractError(f"historical OOF key schema changed: {path.name}")
    frame = pd.read_parquet(path, columns=KEYS).drop_duplicates().reset_index(drop=True)
    if frame.duplicated(KEYS).any():
        raise ContractError(f"historical OOF keys are duplicated: {path.name}")
    return frame


def _case_key_set(frame: pd.DataFrame) -> set[tuple[str, int, str]]:
    return set(
        frame.loc[:, KEYS].itertuples(index=False, name=None)  # type: ignore[arg-type]
    )


def _build_membership(config: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    kma = _read_key_frame(paths["kma_oof"])
    champion = _read_key_frame(paths["champion_oof"])
    validation = _read_key_frame(paths["validation_keys"])
    membership = (
        kma.merge(champion, on=KEYS, validate="one_to_one")
        .merge(validation, on=KEYS, validate="one_to_one")
        .sort_values(KEYS, kind="mergesort")
        .reset_index(drop=True)
    )
    expected = int(config["surface"]["expected_cases"])
    if len(membership) != expected:
        raise ContractError(f"historical OOF intersection is {len(membership)}, expected {expected}")
    triple = _case_key_set(kma) & _case_key_set(champion) & _case_key_set(validation)
    if _case_key_set(membership) != triple:
        raise ContractError("historical OOF intersection is not exact")
    if _case_key_set(kma) & _case_key_set(champion) != triple:
        raise ContractError("KMA/champion intersection differs from the fixed triple intersection")
    if _case_key_set(kma) & _case_key_set(validation) != triple:
        raise ContractError("KMA/validation intersection differs from the fixed triple intersection")
    if set(membership["fold"]) != set(config["surface"]["folds"]):
        raise ContractError("fold set changed")
    if set(membership["station"]) != set(config["surface"]["stations"]):
        raise ContractError("station set changed")
    return membership


def _future_delta_frame(
    cases: pd.DataFrame,
    features: pd.DataFrame,
    atmos_path: Path,
    leads: list[int],
) -> pd.DataFrame:
    expected_columns = ["station", "time", "wspd", "gust", "wdir", "airt", "relh", "caph"]
    observed_columns = pd.read_csv(atmos_path, nrows=0).columns.tolist()
    if observed_columns != expected_columns:
        raise ContractError("historical train_atmos schema changed")
    atmos = pd.read_csv(atmos_path, usecols=["station", "time", "wspd", "wdir"])
    atmos["time"] = pd.to_datetime(atmos["time"], utc=True, errors="raise")
    if atmos.duplicated(["station", "time"]).any():
        raise ContractError("historical train_atmos contains duplicate keys")
    radians = np.deg2rad(atmos["wdir"].to_numpy(dtype=np.float64))
    speed = atmos["wspd"].to_numpy(dtype=np.float64)
    atmos["future_u"] = -speed * np.sin(radians)
    atmos["future_v"] = -speed * np.cos(radians)

    current = features.loc[
        :, ["anchor_id", "station", "wspd_current", "wdir_sin_current", "wdir_cos_current"]
    ].copy()
    current["current_u"] = -current["wspd_current"] * current["wdir_sin_current"]
    current["current_v"] = -current["wspd_current"] * current["wdir_cos_current"]
    base = cases.loc[:, [*KEYS, "anchor_time"]].merge(
        current.loc[:, ["anchor_id", "station", "current_u", "current_v"]],
        on=["anchor_id", "station"],
        validate="one_to_one",
    )
    blocks: list[pd.DataFrame] = []
    for lead in leads:
        block = base.copy()
        block["lead_h"] = int(lead)
        block["time"] = pd.to_datetime(block["anchor_time"], utc=True) + pd.Timedelta(
            hours=int(lead)
        )
        block = block.merge(
            atmos.loc[:, ["station", "time", "future_u", "future_v"]],
            on=["station", "time"],
            how="left",
            validate="one_to_one",
        )
        observed = block.loc[
            :, ["current_u", "current_v", "future_u", "future_v"]
        ].notna().all(axis=1)
        block[f"observed_{lead}h"] = observed
        block[f"delta_u_{lead}h"] = np.where(
            observed, block["future_u"] - block["current_u"], 0.0
        )
        block[f"delta_v_{lead}h"] = np.where(
            observed, block["future_v"] - block["current_v"], 0.0
        )
        blocks.append(
            block.loc[
                :,
                [
                    *KEYS,
                    f"delta_u_{lead}h",
                    f"delta_v_{lead}h",
                    f"observed_{lead}h",
                ],
            ]
        )
    result = blocks[0]
    for block in blocks[1:]:
        result = result.merge(block, on=KEYS, validate="one_to_one")
    delta_columns = [f"delta_{axis}_{lead}h" for lead in leads for axis in ("u", "v")]
    if not np.isfinite(result[delta_columns].to_numpy(dtype=np.float64)).all():
        raise ContractError("completed future-wind delta path is not finite")
    return result


def load_label_free_surface(
    config: dict[str, Any], paths: dict[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    membership = _build_membership(config, paths)
    kma_columns = [
        *ROW_KEYS,
        "incumbent_final",
        "calibrated_source",
        "candidate_final",
    ]
    if not set(kma_columns).issubset(pq.read_schema(paths["kma_oof"]).names):
        raise ContractError("frozen KMA label-free schema changed")
    kma = pd.read_parquet(paths["kma_oof"], columns=kma_columns).merge(
        membership, on=KEYS, validate="many_to_one"
    )
    kma = kma.sort_values(ROW_KEYS, kind="mergesort").reset_index(drop=True)
    surface = config["surface"]
    if len(kma) != int(surface["expected_rows"]) or kma.duplicated(ROW_KEYS).any():
        raise ContractError("frozen 179-case KMA row surface changed")
    expected_leads = tuple(surface["leads_h"])
    per_case = kma.groupby(KEYS, sort=False, observed=True)["lead_h"].agg(tuple)
    if not per_case.map(lambda value: tuple(sorted(value)) == expected_leads).all():
        raise ContractError("a frozen KMA case is not six-lead complete")
    short = kma["lead_h"].isin(surface["exact_no_op_leads_h"])
    if not np.array_equal(
        kma.loc[short, "candidate_final"].to_numpy(dtype=np.float64),
        kma.loc[short, "incumbent_final"].to_numpy(dtype=np.float64),
    ):
        raise ContractError("frozen KMA artifact is not exact no-op at short leads")
    if not np.isfinite(kma["candidate_final"].to_numpy(dtype=np.float64)).all():
        raise ContractError("frozen KMA prediction is not finite")

    feature_columns = list(
        dict.fromkeys(
            [
                *config["oracle"]["control_wave_features"],
                *config["conditional_future_wind"]["context_features"],
            ]
        )
    )
    feature_schema = set(pq.read_schema(paths["features"]).names)
    required_features = {"anchor_id", "station", *feature_columns}
    if not required_features.issubset(feature_schema):
        raise ContractError(
            f"historical feature schema is missing: {sorted(required_features - feature_schema)}"
        )
    features = pd.read_parquet(
        paths["features"], columns=["anchor_id", "station", *feature_columns]
    ).merge(membership.loc[:, KEYS], on=["anchor_id", "station"], validate="one_to_one")
    if len(features) != int(surface["expected_cases"]):
        raise ContractError("historical feature intersection changed")

    anchor_schema = set(pq.read_schema(paths["anchors"]).names)
    if not {"anchor_id", "station", "anchor_time"}.issubset(anchor_schema):
        raise ContractError("historical anchor metadata schema changed")
    anchors = pd.read_parquet(
        paths["anchors"], columns=["anchor_id", "station", "anchor_time"]
    ).merge(membership.loc[:, KEYS], on=["anchor_id", "station"], validate="one_to_one")
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True, errors="raise")
    if len(anchors) != int(surface["expected_cases"]):
        raise ContractError("historical anchor metadata intersection changed")

    cases = (
        membership.merge(
            anchors.loc[:, ["anchor_id", "station", "anchor_time"]],
            on=["anchor_id", "station"],
            validate="one_to_one",
        )
        .merge(features.drop(columns="fold"), on=["anchor_id", "station"], validate="one_to_one")
        .sort_values(KEYS, kind="mergesort")
        .reset_index(drop=True)
    )
    future = _future_delta_frame(cases, features, paths["train_atmos"], surface["leads_h"])
    cases = cases.merge(future, on=KEYS, validate="one_to_one")
    rows = kma.merge(cases, on=KEYS, validate="many_to_one")
    coverage = {
        "eligible_actual_vector_rows_by_lead": {
            str(lead): int(cases[f"observed_{lead}h"].sum()) for lead in surface["leads_h"]
        },
        "eligible_actual_vector_cases_all_six": int(
            cases[[f"observed_{lead}h" for lead in surface["leads_h"]]].all(axis=1).sum()
        ),
        "fallback_rule": config["oracle"]["missing_future_rule"],
    }
    return rows, cases, coverage


def preflight(config_path: Path, data_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    if _git_head(ROOT) != config["base_commit"]:
        raise ContractError("repository HEAD differs from the preregistered base commit")
    paths = _input_paths(config, data_dir)
    hashes = _verify_hashes(config, paths)
    outputs = _output_paths(config)
    consumed = [name for name, path in outputs.items() if path.exists()]
    if consumed:
        raise ContractError(f"one-shot output already exists: {sorted(consumed)}")
    rows, cases, coverage = load_label_free_surface(config, paths)
    summary = {
        "status": "PREFLIGHT_PASS",
        "base_commit_exact": True,
        "historical_input_hashes_exact": True,
        "cases": int(cases["anchor_id"].nunique()),
        "rows": int(len(rows)),
        "fold_case_counts": {
            str(key): int(value)
            for key, value in cases.groupby("fold", observed=True).size().sort_index().items()
        },
        "station_case_counts": {
            str(key): int(value)
            for key, value in cases.groupby("station", observed=True).size().sort_index().items()
        },
        "future_vector_coverage": coverage,
        "official_files_opened": 0,
        "csv_outputs_written": 0,
        "attempt_lock_created": False,
    }
    state = {
        "config": config,
        "config_path": config_path,
        "input_paths": paths,
        "input_hashes": hashes,
        "output_paths": outputs,
        "label_free_rows": rows,
        "label_free_cases": cases,
    }
    return summary, state


def _attach_historical_truth(
    config: dict[str, Any], paths: dict[str, Path], rows: pd.DataFrame
) -> pd.DataFrame:
    leads = config["surface"]["leads_h"]
    target_columns = [f"target_{lead}" for lead in leads]
    schema = set(pq.read_schema(paths["anchors"]).names)
    if not {"anchor_id", *target_columns}.issubset(schema):
        raise ContractError("historical Hs target vault schema changed")
    truth = pd.read_parquet(paths["anchors"], columns=["anchor_id", *target_columns])
    truth = truth.loc[truth["anchor_id"].isin(rows["anchor_id"].unique())]
    truth = truth.melt(
        id_vars="anchor_id", var_name="target_name", value_name="target_hs"
    )
    truth["lead_h"] = truth["target_name"].str.removeprefix("target_").astype(int)
    truth = truth.drop(columns="target_name")
    result = rows.merge(truth, on=["anchor_id", "lead_h"], validate="one_to_one")
    if len(result) != int(config["surface"]["expected_rows"]):
        raise ContractError("historical Hs truth did not attach exactly")
    if not np.isfinite(result["target_hs"].to_numpy(dtype=np.float64)).all():
        raise ContractError("historical Hs truth is not finite")
    return result


def purge_embargo(
    train: pd.DataFrame, validation: pd.DataFrame, embargo_hours: int
) -> pd.DataFrame:
    train_cases = train.loc[:, [*KEYS, "anchor_time"]].drop_duplicates(KEYS)
    validation_cases = validation.loc[:, [*KEYS, "anchor_time"]].drop_duplicates(KEYS)
    keep = np.ones(len(train_cases), dtype=bool)
    embargo = pd.Timedelta(hours=int(embargo_hours))
    for position, row in enumerate(train_cases.itertuples(index=False)):
        times = validation_cases.loc[
            validation_cases["station"].eq(str(row.station)), "anchor_time"
        ]
        if len(times) and (times - pd.Timestamp(row.anchor_time)).abs().min() < embargo:
            keep[position] = False
    kept_keys = train_cases.loc[keep, KEYS]
    result = train.merge(kept_keys, on=KEYS, validate="many_to_one")
    if len(result) == 0:
        raise ContractError("78h embargo removed the entire training surface")
    return result


def _wave_design(
    frame: pd.DataFrame, feature_columns: list[str], include_future: bool
) -> pd.DataFrame:
    columns: dict[str, Any] = {
        "base_prediction": frame["candidate_final"].to_numpy(dtype=np.float64),
        "lead_is_24h": frame["lead_h"].eq(24).to_numpy(dtype=np.float64),
    }
    for station in ("G-ORS", "I-ORS", "S-ORS"):
        columns[f"station_{station}"] = frame["station"].eq(station).to_numpy(dtype=np.float64)
    for name in feature_columns:
        columns[name] = frame[name].to_numpy(dtype=np.float64)
    if include_future:
        for name in [
            "delta_u_3h",
            "delta_v_3h",
            "delta_u_6h",
            "delta_v_6h",
            "delta_u_9h",
            "delta_v_9h",
            "delta_u_12h",
            "delta_v_12h",
            "delta_u_18h",
            "delta_v_18h",
            "delta_u_24h",
            "delta_v_24h",
        ]:
            columns[name] = frame[name].to_numpy(dtype=np.float64)
    return pd.DataFrame(columns, index=frame.index)


def _fit_wave_ridge(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    alpha: float,
    include_future: bool,
) -> np.ndarray:
    train_active = train.loc[train["lead_h"].isin([18, 24])]
    validation_active = validation.loc[validation["lead_h"].isin([18, 24])]
    x_train = _wave_design(train_active, feature_columns, include_future)
    x_validation = _wave_design(validation_active, feature_columns, include_future)
    y_train = (
        train_active["target_hs"].to_numpy(dtype=np.float64)
        - train_active["candidate_final"].to_numpy(dtype=np.float64)
    )
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=float(alpha)),
    )
    pipeline.fit(x_train, y_train)
    correction = pipeline.predict(x_validation)
    base = validation_active["candidate_final"].to_numpy(dtype=np.float64)
    return np.clip(base + correction, 0.0, 30.0)


def select_control_alpha(
    outer_train: pd.DataFrame,
    feature_columns: list[str],
    alphas: list[float],
    embargo_hours: int,
) -> tuple[float, dict[str, float]]:
    folds = sorted(outer_train["fold"].unique().tolist())
    if len(folds) != 2:
        raise ContractError("control-only inner selection requires exactly two outer-train folds")
    scores: dict[str, float] = {}
    for alpha in alphas:
        squared: list[np.ndarray] = []
        for fold in folds:
            inner_validation = outer_train.loc[outer_train["fold"].eq(fold)]
            inner_train = outer_train.loc[~outer_train["fold"].eq(fold)]
            inner_train = purge_embargo(inner_train, inner_validation, embargo_hours)
            prediction = _fit_wave_ridge(
                inner_train,
                inner_validation,
                feature_columns,
                float(alpha),
                include_future=False,
            )
            target = inner_validation.loc[
                inner_validation["lead_h"].isin([18, 24]), "target_hs"
            ].to_numpy(dtype=np.float64)
            squared.append(np.square(prediction - target))
        scores[str(float(alpha))] = float(np.sqrt(np.mean(np.concatenate(squared))))
    selected = min((score, float(alpha)) for alpha, score in ((k, v) for k, v in scores.items()))[1]
    return selected, scores


def make_oracle_predictions(
    config: dict[str, Any], evaluated_rows: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    surface = config["surface"]
    oracle = config["oracle"]
    feature_columns = oracle["control_wave_features"]
    predictions: list[pd.DataFrame] = []
    fold_records: dict[str, Any] = {}
    for fold in surface["folds"]:
        validation = evaluated_rows.loc[evaluated_rows["fold"].eq(fold)].copy()
        outer_train = evaluated_rows.loc[~evaluated_rows["fold"].eq(fold)].copy()
        before_cases = int(outer_train["anchor_id"].nunique())
        outer_train = purge_embargo(outer_train, validation, surface["embargo_hours"])
        after_cases = int(outer_train["anchor_id"].nunique())
        alpha, scores = select_control_alpha(
            outer_train,
            feature_columns,
            [float(value) for value in oracle["ridge_alphas"]],
            int(surface["embargo_hours"]),
        )
        control_active = _fit_wave_ridge(
            outer_train, validation, feature_columns, alpha, include_future=False
        )
        treatment_active = _fit_wave_ridge(
            outer_train, validation, feature_columns, alpha, include_future=True
        )
        blind = validation.loc[:, ROW_KEYS].copy()
        blind["control_prediction"] = validation["candidate_final"].to_numpy(dtype=np.float64)
        blind["treatment_prediction"] = validation["candidate_final"].to_numpy(
            dtype=np.float64
        )
        active = validation["lead_h"].isin(surface["active_leads_h"]).to_numpy()
        if int(active.sum()) != len(control_active) or len(control_active) != len(treatment_active):
            raise ContractError("active oracle prediction alignment changed")
        blind.loc[active, "control_prediction"] = control_active
        blind.loc[active, "treatment_prediction"] = treatment_active
        short = blind["lead_h"].isin(surface["exact_no_op_leads_h"])
        base_short = validation.loc[short, "candidate_final"].to_numpy(dtype=np.float64)
        if not np.array_equal(
            blind.loc[short, "control_prediction"].to_numpy(dtype=np.float64), base_short
        ) or not np.array_equal(
            blind.loc[short, "treatment_prediction"].to_numpy(dtype=np.float64), base_short
        ):
            raise ContractError("oracle short-lead exact no-op failed before seal")
        predictions.append(blind)
        fold_records[str(fold)] = {
            "selected_control_only_alpha": float(alpha),
            "control_only_inner_rmse_by_alpha": scores,
            "outer_train_cases_before_embargo": before_cases,
            "outer_train_cases_after_embargo": after_cases,
            "validation_cases": int(validation["anchor_id"].nunique()),
        }
    result = pd.concat(predictions, ignore_index=True).sort_values(
        ROW_KEYS, kind="mergesort"
    ).reset_index(drop=True)
    if len(result) != int(surface["expected_rows"]) or result.duplicated(ROW_KEYS).any():
        raise ContractError("oracle blind prediction surface changed")
    return result, fold_records


def write_prediction_seal(
    stage: str,
    prediction_path: Path,
    seal_path: Path,
    prediction_frame: pd.DataFrame,
    state: dict[str, Any],
    attempt_sha: str,
) -> dict[str, Any]:
    forbidden = {"target_hs", "target_3", "target_6", "target_9", "target_12", "target_18", "target_24"}
    if forbidden & set(prediction_frame.columns):
        raise ContractError("blind prediction frame exposes historical Hs truth")
    exclusive_parquet(prediction_path, prediction_frame)
    prediction_sha = sha256_file(prediction_path)
    manifest = {
        "sealed": True,
        "stage": stage,
        "prediction_sha256": prediction_sha,
        "prediction_rows": int(len(prediction_frame)),
        "prediction_cases": int(prediction_frame["anchor_id"].nunique()),
        "prediction_columns": prediction_frame.columns.tolist(),
        "target_columns_present": False,
        "attempt_lock_sha256": attempt_sha,
        "config_sha256": sha256_file(state["config_path"]),
        "script_sha256": sha256_file(Path(__file__)),
        "historical_input_sha256": state["input_hashes"],
        "official_files_opened": 0,
    }
    exclusive_json(seal_path, manifest)
    persisted = json.loads(seal_path.read_text(encoding="utf-8"))
    if persisted != manifest or sha256_file(prediction_path) != prediction_sha:
        raise ContractError("blind prediction seal failed immediate reload verification")
    reloaded = pd.read_parquet(prediction_path)
    if reloaded.columns.tolist() != prediction_frame.columns.tolist() or len(reloaded) != len(
        prediction_frame
    ):
        raise ContractError("blind prediction parquet failed immediate reload verification")
    return {"prediction_sha256": prediction_sha, "seal_sha256": sha256_file(seal_path)}


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def paired_case_bootstrap_wave(
    frame: pd.DataFrame, replicates: int, seed: int
) -> dict[str, Any]:
    ordered = frame.sort_values(["anchor_id", "lead_h"], kind="mergesort")
    groups = list(ordered.groupby("anchor_id", sort=True, observed=True))
    if not groups or any(len(group) != 6 for _, group in groups):
        raise ContractError("wave bootstrap requires complete six-lead cases")
    control_ss = np.array(
        [np.square(group["control_prediction"] - group["target_hs"]).sum() for _, group in groups]
    )
    treatment_ss = np.array(
        [
            np.square(group["treatment_prediction"] - group["target_hs"]).sum()
            for _, group in groups
        ]
    )
    generator = np.random.default_rng(int(seed))
    draws = generator.integers(0, len(groups), size=(int(replicates), len(groups)))
    denominator = float(len(groups) * 6)
    deltas = np.sqrt(treatment_ss[draws].sum(axis=1) / denominator) - np.sqrt(
        control_ss[draws].sum(axis=1) / denominator
    )
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "ci90_lower_m": float(np.quantile(deltas, 0.05)),
        "ci90_upper_m": float(np.quantile(deltas, 0.95)),
        "median_delta_rmse_m": float(np.median(deltas)),
    }


def evaluate_wave_gate(
    blind: pd.DataFrame,
    truth_rows: pd.DataFrame,
    gate: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    evaluated = blind.merge(
        truth_rows.loc[:, [*ROW_KEYS, "target_hs"]], on=ROW_KEYS, validate="one_to_one"
    )
    control = evaluated["control_prediction"].to_numpy(dtype=np.float64)
    treatment = evaluated["treatment_prediction"].to_numpy(dtype=np.float64)
    target = evaluated["target_hs"].to_numpy(dtype=np.float64)
    overall_control = _rmse(control, target)
    overall_treatment = _rmse(treatment, target)
    delta = overall_treatment - overall_control

    def slices(column: str) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for name, group in evaluated.groupby(column, sort=True, observed=True):
            c = _rmse(group["control_prediction"].to_numpy(), group["target_hs"].to_numpy())
            t = _rmse(
                group["treatment_prediction"].to_numpy(), group["target_hs"].to_numpy()
            )
            result[str(name)] = {
                "control_rmse_m": c,
                "treatment_rmse_m": t,
                "delta_rmse_m": t - c,
            }
        return result

    by_fold = slices("fold")
    by_station = slices("station")
    by_lead = slices("lead_h")
    station_lead: dict[str, dict[str, float]] = {}
    for (station, lead), group in evaluated.groupby(
        ["station", "lead_h"], sort=True, observed=True
    ):
        c = _rmse(group["control_prediction"].to_numpy(), group["target_hs"].to_numpy())
        t = _rmse(group["treatment_prediction"].to_numpy(), group["target_hs"].to_numpy())
        station_lead[f"{station}|{int(lead)}"] = {
            "control_rmse_m": c,
            "treatment_rmse_m": t,
            "delta_rmse_m": t - c,
        }
    boot = paired_case_bootstrap_wave(
        evaluated, int(bootstrap["replicates"]), int(bootstrap["seed"])
    )
    improved_folds = sum(item["delta_rmse_m"] < 0.0 for item in by_fold.values())
    improved_stations = sum(item["delta_rmse_m"] < 0.0 for item in by_station.values())
    worst = max(item["delta_rmse_m"] for item in station_lead.values())
    checks = {
        "pooled_delta_at_most_threshold": delta
        <= float(gate["pooled_six_lead_delta_rmse_m_max"]),
        "bootstrap_ci90_upper_strictly_below_zero": boot["ci90_upper_m"]
        < float(gate["paired_case_bootstrap_ci90_upper_strictly_below_m"]),
        "minimum_improved_folds": improved_folds >= int(gate["minimum_improved_folds"]),
        "minimum_improved_stations": improved_stations
        >= int(gate["minimum_improved_stations"]),
        "lead_18_non_degrade_or_improve": by_lead["18"]["delta_rmse_m"]
        <= float(gate["lead_18_delta_rmse_m_max"]),
        "lead_24_non_degrade_or_improve": by_lead["24"]["delta_rmse_m"]
        <= float(gate["lead_24_delta_rmse_m_max"]),
        "worst_station_by_lead_within_limit": worst
        <= float(gate["worst_station_by_lead_delta_rmse_m_max"]),
    }
    return {
        "overall": {
            "control_rmse_m": overall_control,
            "treatment_rmse_m": overall_treatment,
            "delta_rmse_m": delta,
        },
        "by_fold": by_fold,
        "by_station": by_station,
        "by_lead": by_lead,
        "by_station_lead": station_lead,
        "improved_fold_count": int(improved_folds),
        "improved_station_count": int(improved_stations),
        "worst_station_lead_delta_rmse_m": float(worst),
        "paired_whole_case_bootstrap": boot,
        "gate_checks": checks,
        "gate_pass": bool(all(checks.values())),
    }


def _wind_design(frame: pd.DataFrame, context_features: list[str]) -> pd.DataFrame:
    columns: dict[str, Any] = {}
    for station in ("G-ORS", "I-ORS", "S-ORS"):
        columns[f"station_{station}"] = frame["station"].eq(station).to_numpy(dtype=np.float64)
    for name in context_features:
        columns[name] = frame[name].to_numpy(dtype=np.float64)
    return pd.DataFrame(columns, index=frame.index)


def _wind_targets(frame: pd.DataFrame, leads: list[int]) -> np.ndarray:
    columns = [f"delta_{axis}_{lead}h" for lead in leads for axis in ("u", "v")]
    return frame[columns].to_numpy(dtype=np.float64)


def _fit_wind_ridge(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    context_features: list[str],
    leads: list[int],
    alpha: float,
) -> np.ndarray:
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=float(alpha)),
    )
    pipeline.fit(_wind_design(train, context_features), _wind_targets(train, leads))
    return np.asarray(pipeline.predict(_wind_design(validation, context_features)), dtype=np.float64)


def select_wind_alpha(
    outer_train: pd.DataFrame,
    context_features: list[str],
    leads: list[int],
    alphas: list[float],
    embargo_hours: int,
) -> tuple[float, dict[str, float]]:
    folds = sorted(outer_train["fold"].unique().tolist())
    if len(folds) != 2:
        raise ContractError("future-wind inner selection requires exactly two folds")
    scores: dict[str, float] = {}
    for alpha in alphas:
        losses: list[np.ndarray] = []
        for fold in folds:
            validation = outer_train.loc[outer_train["fold"].eq(fold)]
            train = outer_train.loc[~outer_train["fold"].eq(fold)]
            train = purge_embargo(train, validation, embargo_hours)
            prediction = _fit_wind_ridge(
                train, validation, context_features, leads, float(alpha)
            )
            losses.append(np.square(prediction - _wind_targets(validation, leads)).ravel())
        scores[str(float(alpha))] = float(np.mean(np.concatenate(losses)))
    selected = min((score, float(alpha)) for alpha, score in ((k, v) for k, v in scores.items()))[1]
    return selected, scores


def _prediction_columns(leads: list[int]) -> list[str]:
    return [f"pred_delta_{axis}_{lead}h" for lead in leads for axis in ("u", "v")]


def make_wind_predictions(
    config: dict[str, Any], cases: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    surface = config["surface"]
    wind = config["conditional_future_wind"]
    leads = surface["leads_h"]
    prediction_columns = _prediction_columns(leads)
    outer_predictions: list[pd.DataFrame] = []
    nested_predictions: dict[str, pd.DataFrame] = {}
    records: dict[str, Any] = {}
    for outer_fold in surface["folds"]:
        validation = cases.loc[cases["fold"].eq(outer_fold)].copy()
        outer_train = cases.loc[~cases["fold"].eq(outer_fold)].copy()
        outer_train = purge_embargo(outer_train, validation, surface["embargo_hours"])
        alpha, scores = select_wind_alpha(
            outer_train,
            wind["context_features"],
            leads,
            [float(value) for value in wind["ridge_alphas"]],
            int(surface["embargo_hours"]),
        )
        prediction = _fit_wind_ridge(
            outer_train, validation, wind["context_features"], leads, alpha
        )
        block = validation.loc[:, KEYS].copy()
        block[prediction_columns] = prediction
        outer_predictions.append(block)

        nested_blocks: list[pd.DataFrame] = []
        for inner_fold in sorted(outer_train["fold"].unique().tolist()):
            inner_validation = outer_train.loc[outer_train["fold"].eq(inner_fold)]
            inner_train = outer_train.loc[~outer_train["fold"].eq(inner_fold)]
            inner_train = purge_embargo(
                inner_train, inner_validation, int(surface["embargo_hours"])
            )
            inner_prediction = _fit_wind_ridge(
                inner_train,
                inner_validation,
                wind["context_features"],
                leads,
                alpha,
            )
            nested = inner_validation.loc[:, KEYS].copy()
            nested[prediction_columns] = inner_prediction
            nested_blocks.append(nested)
        nested_frame = pd.concat(nested_blocks, ignore_index=True)
        if len(nested_frame) != outer_train["anchor_id"].nunique():
            raise ContractError("nested future-wind OOF coverage changed")
        nested_predictions[str(outer_fold)] = nested_frame
        records[str(outer_fold)] = {
            "selected_forcing_only_alpha": float(alpha),
            "forcing_only_inner_mse_by_alpha": scores,
            "outer_train_cases_after_embargo": int(outer_train["anchor_id"].nunique()),
            "validation_cases": int(validation["anchor_id"].nunique()),
        }
    result = pd.concat(outer_predictions, ignore_index=True).sort_values(
        KEYS, kind="mergesort"
    ).reset_index(drop=True)
    if len(result) != int(surface["expected_cases"]) or result.duplicated(KEYS).any():
        raise ContractError("future-wind blind prediction surface changed")
    return result, nested_predictions, records


def _wind_skill(group: pd.DataFrame, leads: list[int]) -> dict[str, float]:
    actual_columns = [f"delta_{axis}_{lead}h" for lead in leads for axis in ("u", "v")]
    prediction_columns = [f"pred_delta_{axis}_{lead}h" for lead in leads for axis in ("u", "v")]
    actual = group[actual_columns].to_numpy(dtype=np.float64)
    prediction = group[prediction_columns].to_numpy(dtype=np.float64)
    persistence_mse = float(np.mean(np.square(actual)))
    prediction_mse = float(np.mean(np.square(prediction - actual)))
    skill = float(1.0 - prediction_mse / persistence_mse) if persistence_mse > 0.0 else -np.inf
    return {
        "persistence_mse": persistence_mse,
        "prediction_mse": prediction_mse,
        "skill": skill,
    }


def evaluate_wind_gate(
    blind: pd.DataFrame,
    cases: pd.DataFrame,
    gate: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    evaluated = blind.merge(cases, on=KEYS, validate="one_to_one")
    active = [18, 24]
    overall = _wind_skill(evaluated, active)
    by_fold = {
        str(name): _wind_skill(group, active)
        for name, group in evaluated.groupby("fold", sort=True, observed=True)
    }
    by_station = {
        str(name): _wind_skill(group, active)
        for name, group in evaluated.groupby("station", sort=True, observed=True)
    }
    actual_columns = [f"delta_{axis}_{lead}h" for lead in active for axis in ("u", "v")]
    prediction_columns = [
        f"pred_delta_{axis}_{lead}h" for lead in active for axis in ("u", "v")
    ]
    actual = evaluated[actual_columns].to_numpy(dtype=np.float64)
    prediction = evaluated[prediction_columns].to_numpy(dtype=np.float64)
    persistence_ss = np.square(actual).sum(axis=1)
    prediction_ss = np.square(prediction - actual).sum(axis=1)
    generator = np.random.default_rng(int(bootstrap["seed"]))
    draws = generator.integers(
        0, len(evaluated), size=(int(bootstrap["replicates"]), len(evaluated))
    )
    base = persistence_ss[draws].sum(axis=1)
    candidate = prediction_ss[draws].sum(axis=1)
    skills = np.where(base > 0.0, 1.0 - candidate / base, -np.inf)
    boot = {
        "replicates": int(bootstrap["replicates"]),
        "seed": int(bootstrap["seed"]),
        "ci90_lower": float(np.quantile(skills, 0.05)),
        "ci90_upper": float(np.quantile(skills, 0.95)),
        "median_skill": float(np.median(skills)),
    }
    positive_folds = sum(item["skill"] > 0.0 for item in by_fold.values())
    positive_stations = sum(item["skill"] > 0.0 for item in by_station.values())
    worst_station = min(item["skill"] for item in by_station.values())
    checks = {
        "pooled_skill_at_least_five_percent": overall["skill"]
        >= float(gate["pooled_18_24_wind_vector_mse_skill_min"]),
        "bootstrap_ci90_lower_strictly_above_zero": boot["ci90_lower"]
        > float(gate["paired_case_bootstrap_ci90_lower_strictly_above"]),
        "minimum_positive_skill_folds": positive_folds
        >= int(gate["minimum_positive_skill_folds"]),
        "minimum_positive_skill_stations": positive_stations
        >= int(gate["minimum_positive_skill_stations"]),
        "worst_station_within_limit": worst_station >= float(gate["worst_station_skill_min"]),
    }
    return {
        "overall_18_24": overall,
        "by_fold": by_fold,
        "by_station": by_station,
        "positive_fold_count": int(positive_folds),
        "positive_station_count": int(positive_stations),
        "worst_station_skill": float(worst_station),
        "paired_whole_case_bootstrap": boot,
        "gate_checks": checks,
        "gate_pass": bool(all(checks.values())),
    }


def _replace_actual_with_predicted_wind(
    frame: pd.DataFrame, wind_predictions: pd.DataFrame, leads: list[int]
) -> pd.DataFrame:
    prediction_columns = _prediction_columns(leads)
    merged = frame.merge(
        wind_predictions.loc[:, [*KEYS, *prediction_columns]], on=KEYS, validate="many_to_one"
    )
    for lead in leads:
        for axis in ("u", "v"):
            merged[f"delta_{axis}_{lead}h"] = merged[f"pred_delta_{axis}_{lead}h"]
    return merged.drop(columns=prediction_columns)


def make_mos_predictions(
    config: dict[str, Any],
    evaluated_rows: pd.DataFrame,
    oracle_blind: pd.DataFrame,
    wind_blind: pd.DataFrame,
    nested_wind: dict[str, pd.DataFrame],
    oracle_fold_records: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    surface = config["surface"]
    feature_columns = config["oracle"]["control_wave_features"]
    predictions: list[pd.DataFrame] = []
    records: dict[str, Any] = {}
    for fold in surface["folds"]:
        validation = evaluated_rows.loc[evaluated_rows["fold"].eq(fold)].copy()
        outer_train = evaluated_rows.loc[~evaluated_rows["fold"].eq(fold)].copy()
        outer_train = purge_embargo(outer_train, validation, surface["embargo_hours"])
        train_predicted = _replace_actual_with_predicted_wind(
            outer_train, nested_wind[str(fold)], surface["leads_h"]
        )
        validation_predicted = _replace_actual_with_predicted_wind(
            validation,
            wind_blind.loc[wind_blind["fold"].eq(fold)],
            surface["leads_h"],
        )
        alpha = float(oracle_fold_records[str(fold)]["selected_control_only_alpha"])
        treatment_active = _fit_wave_ridge(
            train_predicted,
            validation_predicted,
            feature_columns,
            alpha,
            include_future=True,
        )
        blind = validation.loc[:, ROW_KEYS].copy()
        oracle_control = oracle_blind.loc[
            oracle_blind["fold"].eq(fold), [*ROW_KEYS, "control_prediction"]
        ]
        blind = blind.merge(oracle_control, on=ROW_KEYS, validate="one_to_one")
        blind["treatment_prediction"] = blind["control_prediction"].to_numpy(dtype=np.float64)
        active = blind["lead_h"].isin(surface["active_leads_h"]).to_numpy()
        blind.loc[active, "treatment_prediction"] = treatment_active
        short = blind["lead_h"].isin(surface["exact_no_op_leads_h"])
        if not np.array_equal(
            blind.loc[short, "control_prediction"].to_numpy(dtype=np.float64),
            blind.loc[short, "treatment_prediction"].to_numpy(dtype=np.float64),
        ):
            raise ContractError("predicted-wind MOS short-lead exact no-op failed")
        predictions.append(blind)
        records[str(fold)] = {
            "reused_control_only_alpha": alpha,
            "nested_cross_fit_train_cases": int(train_predicted["anchor_id"].nunique()),
            "validation_cases": int(validation["anchor_id"].nunique()),
        }
    result = pd.concat(predictions, ignore_index=True).sort_values(
        ROW_KEYS, kind="mergesort"
    ).reset_index(drop=True)
    if len(result) != int(surface["expected_rows"]):
        raise ContractError("predicted-wind MOS blind surface changed")
    return result, records


def execute(state: dict[str, Any], preflight_summary: dict[str, Any]) -> dict[str, Any]:
    config = state["config"]
    outputs = state["output_paths"]
    lock_payload = {
        "experiment_id": config["experiment_id"],
        "base_commit": config["base_commit"],
        "config_sha256": sha256_file(state["config_path"]),
        "script_sha256": sha256_file(Path(__file__)),
        "historical_input_sha256": state["input_hashes"],
        "scientific_attempt_ordinal": 1,
        "result_based_rerun_allowed": False,
        "official_files_allowed": False,
        "csv_output_allowed": False,
        "upload_allowed": False,
    }
    exclusive_json(outputs["attempt_lock"], lock_payload)
    attempt_sha = sha256_file(outputs["attempt_lock"])
    try:
        evaluated_rows = _attach_historical_truth(
            config, state["input_paths"], state["label_free_rows"]
        )
        oracle_blind, oracle_fold_records = make_oracle_predictions(config, evaluated_rows)
        oracle_seals = write_prediction_seal(
            "all_oracle_predictions_before_designated_wave_metric",
            outputs["oracle_predictions"],
            outputs["oracle_seal"],
            oracle_blind,
            state,
            attempt_sha,
        )
        oracle_metrics = evaluate_wave_gate(
            pd.read_parquet(outputs["oracle_predictions"]),
            evaluated_rows,
            config["oracle"]["gate"],
            config["oracle"]["bootstrap"],
        )
        result: dict[str, Any] = {
            "experiment_id": config["experiment_id"],
            "base_commit": config["base_commit"],
            "status": "",
            "attempts_consumed": 1,
            "preflight": preflight_summary,
            "data_boundary": {
                "historical_train_only": True,
                "official_files_opened": 0,
                "official_values_read": 0,
                "official_absolute_times_reconstructed": 0,
                "external_period_matches": 0,
                "csv_outputs_written": 0,
                "uploads": 0,
                "source_mutations": 0,
            },
            "seals": {"attempt_lock_sha256": attempt_sha, "oracle": oracle_seals},
            "oracle": {
                "executed": True,
                "fold_training": oracle_fold_records,
                "metrics": oracle_metrics,
            },
            "conditional_future_wind": {"executed": False},
            "conditional_mos": {"executed": False},
        }
        if not oracle_metrics["gate_pass"]:
            result["status"] = "CLOSE_PREDICTED_FUTURE_WIND_AND_MOS_FAMILY"
            result["conditional_future_wind"]["reason"] = "oracle_gate_failed"
            result["conditional_mos"]["reason"] = "oracle_gate_failed"
            exclusive_json(outputs["result"], result)
            return result

        wind_blind, nested_wind, wind_records = make_wind_predictions(
            config, state["label_free_cases"]
        )
        wind_seals = write_prediction_seal(
            "all_future_wind_predictions_before_designated_forcing_metric",
            outputs["wind_predictions"],
            outputs["wind_seal"],
            wind_blind,
            state,
            attempt_sha,
        )
        wind_metrics = evaluate_wind_gate(
            pd.read_parquet(outputs["wind_predictions"]),
            state["label_free_cases"],
            config["conditional_future_wind"]["gate"],
            config["conditional_future_wind"]["bootstrap"],
        )
        result["seals"]["future_wind"] = wind_seals
        result["conditional_future_wind"] = {
            "executed": True,
            "fold_training": wind_records,
            "metrics": wind_metrics,
        }
        if not wind_metrics["gate_pass"]:
            result["status"] = "CLOSE_DEPLOYABLE_FUTURE_WIND_AND_MOS_FAMILY"
            result["conditional_mos"]["reason"] = "future_wind_gate_failed"
            exclusive_json(outputs["result"], result)
            return result

        mos_blind, mos_records = make_mos_predictions(
            config,
            evaluated_rows,
            oracle_blind,
            wind_blind,
            nested_wind,
            oracle_fold_records,
        )
        mos_seals = write_prediction_seal(
            "all_predicted_wind_mos_predictions_before_designated_wave_metric",
            outputs["mos_predictions"],
            outputs["mos_seal"],
            mos_blind,
            state,
            attempt_sha,
        )
        mos_metrics = evaluate_wave_gate(
            pd.read_parquet(outputs["mos_predictions"]),
            evaluated_rows,
            config["conditional_mos"]["gate"],
            config["conditional_mos"]["bootstrap"],
        )
        result["seals"]["mos"] = mos_seals
        result["conditional_mos"] = {
            "executed": True,
            "fold_training": mos_records,
            "metrics": mos_metrics,
        }
        result["status"] = (
            "LOCAL_CANDIDATE_ONLY_OFFICIAL_PAIRED_AB_REQUIRED"
            if mos_metrics["gate_pass"]
            else "REJECT_PREDICTED_WIND_MOS"
        )
        exclusive_json(outputs["result"], result)
        return result
    except Exception as exc:
        failure = {
            "experiment_id": config["experiment_id"],
            "attempt_lock_sha256": attempt_sha,
            "technical_failure": True,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback_sha256": hashlib.sha256(traceback.format_exc().encode()).hexdigest(),
            "rerun_allowed": False,
        }
        if not outputs["technical_failure_receipt"].exists():
            exclusive_json(outputs["technical_failure_receipt"], failure)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_dir = args.data_dir
    if data_dir is None:
        explicit = os.environ.get("P3_DATA_DIR")
        if not explicit:
            raise ContractError("P3_DATA_DIR or --data-dir is required")
        data_dir = Path(explicit)
    summary, state = preflight(args.config.resolve(), data_dir.resolve())
    if args.preflight:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    result = execute(state, summary)
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "status": result["status"],
                "attempts_consumed": result["attempts_consumed"],
                "oracle_gate_pass": result["oracle"]["metrics"]["gate_pass"],
                "conditional_future_wind_executed": result["conditional_future_wind"][
                    "executed"
                ],
                "conditional_mos_executed": result["conditional_mos"]["executed"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
