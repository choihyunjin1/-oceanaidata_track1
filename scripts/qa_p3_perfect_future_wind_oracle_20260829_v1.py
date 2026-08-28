#!/usr/bin/env python3
"""Independent aggregate-only QA for the sealed P3 future-wind oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/p3_perfect_future_wind_oracle_20260829_v1.json"
QA_PATH = ROOT / "reports/p3_perfect_future_wind_oracle_20260829_v1/independent_qa.json"
ROW_KEYS = ["fold", "anchor_id", "station", "lead_h"]


class QAError(RuntimeError):
    """Raised when independent replay differs from the sealed aggregate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise QAError(f"path escapes repository: {relative}") from exc
    return path


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("QA write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def attach_truth(config: dict[str, Any], blind: pd.DataFrame) -> pd.DataFrame:
    anchors_path = contained(config["inputs"]["anchors"]["path"])
    if sha256_file(anchors_path) != config["inputs"]["anchors"]["sha256"]:
        raise QAError("historical anchor vault hash changed")
    leads = config["surface"]["leads_h"]
    columns = [f"target_{lead}" for lead in leads]
    truth = pd.read_parquet(anchors_path, columns=["anchor_id", *columns])
    truth = truth.loc[truth["anchor_id"].isin(blind["anchor_id"].unique())]
    truth = truth.melt(id_vars="anchor_id", var_name="name", value_name="target_hs")
    truth["lead_h"] = truth.pop("name").str.removeprefix("target_").astype(int)
    return blind.merge(truth, on=["anchor_id", "lead_h"], validate="one_to_one")


def bootstrap(frame: pd.DataFrame, replicates: int, seed: int) -> dict[str, float | int]:
    groups = list(
        frame.sort_values(["anchor_id", "lead_h"], kind="mergesort").groupby(
            "anchor_id", sort=True, observed=True
        )
    )
    if len(groups) != 179 or any(len(group) != 6 for _, group in groups):
        raise QAError("bootstrap block surface differs from 179 complete six-lead cases")
    control = np.array(
        [np.square(group["control_prediction"] - group["target_hs"]).sum() for _, group in groups]
    )
    treatment = np.array(
        [
            np.square(group["treatment_prediction"] - group["target_hs"]).sum()
            for _, group in groups
        ]
    )
    generator = np.random.default_rng(seed)
    draw = generator.integers(0, len(groups), size=(replicates, len(groups)))
    scale = float(len(groups) * 6)
    delta = np.sqrt(treatment[draw].sum(axis=1) / scale) - np.sqrt(
        control[draw].sum(axis=1) / scale
    )
    return {
        "replicates": replicates,
        "seed": seed,
        "ci90_lower_m": float(np.quantile(delta, 0.05)),
        "ci90_upper_m": float(np.quantile(delta, 0.95)),
        "median_delta_rmse_m": float(np.median(delta)),
    }


def slice_metrics(frame: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for name, group in frame.groupby(column, sort=True, observed=True):
        control = rmse(group["control_prediction"].to_numpy(), group["target_hs"].to_numpy())
        treatment = rmse(
            group["treatment_prediction"].to_numpy(), group["target_hs"].to_numpy()
        )
        output[str(name)] = {
            "control_rmse_m": control,
            "treatment_rmse_m": treatment,
            "delta_rmse_m": treatment - control,
        }
    return output


def exact_nested_close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            exact_nested_close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            exact_nested_close(a, b, tolerance) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=tolerance))
    return left == right


def run_qa(config_path: Path, qa_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    outputs = {name: contained(path) for name, path in config["one_shot"].items()}
    result = json.loads(outputs["result"].read_text(encoding="utf-8"))
    seal = json.loads(outputs["oracle_seal"].read_text(encoding="utf-8"))
    blind = pd.read_parquet(outputs["oracle_predictions"])
    expected_columns = [*ROW_KEYS, "control_prediction", "treatment_prediction"]
    checks: dict[str, bool] = {
        "result_status_closes_family": result["status"]
        == "CLOSE_PREDICTED_FUTURE_WIND_AND_MOS_FAMILY",
        "oracle_gate_failed": result["oracle"]["metrics"]["gate_pass"] is False,
        "future_wind_not_executed": result["conditional_future_wind"]["executed"] is False,
        "mos_not_executed": result["conditional_mos"]["executed"] is False,
        "conditional_artifacts_absent": not any(
            outputs[name].exists()
            for name in ("wind_predictions", "wind_seal", "mos_predictions", "mos_seal")
        ),
        "prediction_hash_matches_result": sha256_file(outputs["oracle_predictions"])
        == result["seals"]["oracle"]["prediction_sha256"],
        "prediction_hash_matches_seal": sha256_file(outputs["oracle_predictions"])
        == seal["prediction_sha256"],
        "seal_hash_matches_result": sha256_file(outputs["oracle_seal"])
        == result["seals"]["oracle"]["seal_sha256"],
        "attempt_hash_matches_result": sha256_file(outputs["attempt_lock"])
        == result["seals"]["attempt_lock_sha256"],
        "blind_schema_exact": blind.columns.tolist() == expected_columns,
        "blind_has_no_target": not any(column.startswith("target") for column in blind.columns),
        "blind_rows_exact": len(blind) == 1074,
        "blind_cases_exact": blind["anchor_id"].nunique() == 179,
        "blind_keys_unique": not blind.duplicated(ROW_KEYS).any(),
    }
    per_case_leads = blind.groupby(
        ["fold", "anchor_id", "station"], sort=False, observed=True
    )["lead_h"].agg(lambda values: tuple(sorted(values)))
    checks["six_lead_cases_exact"] = per_case_leads.map(
        lambda value: value == (3, 6, 9, 12, 18, 24)
    ).all()

    kma_path = contained(config["inputs"]["kma_oof"]["path"])
    if sha256_file(kma_path) != config["inputs"]["kma_oof"]["sha256"]:
        raise QAError("frozen KMA OOF hash changed")
    kma = pd.read_parquet(kma_path, columns=[*ROW_KEYS, "candidate_final"])
    short = blind.loc[blind["lead_h"].isin([3, 6, 9, 12])].merge(
        kma, on=ROW_KEYS, validate="one_to_one"
    )
    checks["short_control_exact_frozen_kma"] = np.array_equal(
        short["control_prediction"].to_numpy(), short["candidate_final"].to_numpy()
    )
    checks["short_treatment_exact_frozen_kma"] = np.array_equal(
        short["treatment_prediction"].to_numpy(), short["candidate_final"].to_numpy()
    )

    evaluated = attach_truth(config, blind)
    control = rmse(evaluated["control_prediction"].to_numpy(), evaluated["target_hs"].to_numpy())
    treatment = rmse(
        evaluated["treatment_prediction"].to_numpy(), evaluated["target_hs"].to_numpy()
    )
    replay: dict[str, Any] = {
        "overall": {
            "control_rmse_m": control,
            "treatment_rmse_m": treatment,
            "delta_rmse_m": treatment - control,
        },
        "by_fold": slice_metrics(evaluated, "fold"),
        "by_station": slice_metrics(evaluated, "station"),
        "by_lead": slice_metrics(evaluated, "lead_h"),
    }
    by_station_lead: dict[str, dict[str, float]] = {}
    for (station, lead), group in evaluated.groupby(
        ["station", "lead_h"], sort=True, observed=True
    ):
        c_value = rmse(
            group["control_prediction"].to_numpy(), group["target_hs"].to_numpy()
        )
        t_value = rmse(
            group["treatment_prediction"].to_numpy(), group["target_hs"].to_numpy()
        )
        by_station_lead[f"{station}|{int(lead)}"] = {
            "control_rmse_m": c_value,
            "treatment_rmse_m": t_value,
            "delta_rmse_m": t_value - c_value,
        }
    replay["by_station_lead"] = by_station_lead
    replay["improved_fold_count"] = sum(
        value["delta_rmse_m"] < 0.0 for value in replay["by_fold"].values()
    )
    replay["improved_station_count"] = sum(
        value["delta_rmse_m"] < 0.0 for value in replay["by_station"].values()
    )
    replay["worst_station_lead_delta_rmse_m"] = max(
        value["delta_rmse_m"] for value in by_station_lead.values()
    )
    replay["paired_whole_case_bootstrap"] = bootstrap(evaluated, 5000, 20260829)
    gate = config["oracle"]["gate"]
    replay["gate_checks"] = {
        "pooled_delta_at_most_threshold": replay["overall"]["delta_rmse_m"]
        <= gate["pooled_six_lead_delta_rmse_m_max"],
        "bootstrap_ci90_upper_strictly_below_zero": replay[
            "paired_whole_case_bootstrap"
        ]["ci90_upper_m"]
        < gate["paired_case_bootstrap_ci90_upper_strictly_below_m"],
        "minimum_improved_folds": replay["improved_fold_count"]
        >= gate["minimum_improved_folds"],
        "minimum_improved_stations": replay["improved_station_count"]
        >= gate["minimum_improved_stations"],
        "lead_18_non_degrade_or_improve": replay["by_lead"]["18"]["delta_rmse_m"]
        <= gate["lead_18_delta_rmse_m_max"],
        "lead_24_non_degrade_or_improve": replay["by_lead"]["24"]["delta_rmse_m"]
        <= gate["lead_24_delta_rmse_m_max"],
        "worst_station_by_lead_within_limit": replay[
            "worst_station_lead_delta_rmse_m"
        ]
        <= gate["worst_station_by_lead_delta_rmse_m_max"],
    }
    replay["gate_pass"] = all(replay["gate_checks"].values())
    checks["independent_metric_replay_exact"] = exact_nested_close(
        replay, result["oracle"]["metrics"]
    )
    checks["all_seven_gate_checks_fail"] = not any(replay["gate_checks"].values())
    checks = {name: bool(value) for name, value in checks.items()}
    qa = {
        "experiment_id": config["experiment_id"],
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "independent_replay": replay,
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "qa_script_sha256": sha256_file(Path(__file__)),
            "result_sha256": sha256_file(outputs["result"]),
            "oracle_prediction_sha256": sha256_file(outputs["oracle_predictions"]),
            "oracle_seal_sha256": sha256_file(outputs["oracle_seal"]),
        },
        "data_boundary": {
            "historical_only": True,
            "official_files_opened": 0,
            "csv_outputs_written": 0,
            "models_refit": 0,
        },
    }
    exclusive_json(qa_path, qa)
    if qa["status"] != "PASS":
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise QAError(f"independent QA failed: {failed}")
    return qa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=QA_PATH)
    args = parser.parse_args()
    qa = run_qa(args.config.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "experiment_id": qa["experiment_id"],
                "status": qa["status"],
                "checks": len(qa["checks"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
