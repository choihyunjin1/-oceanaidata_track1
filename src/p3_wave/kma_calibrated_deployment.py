"""Secondary P3 deployment of the fixed KMA long-lead calibration.

This module deliberately contains no model-selection logic.  It preserves the
validated v2 structure: one fixed Ridge calibrator for 18 h, one for 24 h, and
an alpha-0.4 blend into the frozen incumbent.  Test features are derived only
from the relative 48-hour history belonging to the same anonymous case.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p3_wave.kma_calibrated_longlead_blend import (
    ACTIVE_LEADS,
    NO_OP_LEADS,
    RidgeAffineCalibrator,
)
from p3_wave.kma_source_meta import (
    DIRECTION_VARIABLES,
    HISTORY_ROWS,
    LEADS,
    META_COLUMNS,
    SCALAR_VARIABLES,
    compact_source_feature_columns,
    summarize_common_history,
)

EXPERIMENT_ID = "p3_kma_calibrated_longlead_deployment_v1"
DEPLOYMENT_ALPHA = 0.4
EXPECTED_CASES = 200
EXPECTED_ROWS = EXPECTED_CASES * len(LEADS)
EXPECTED_FULL_TRAIN_ANCHORS = 24_360
EXPECTED_REUSED_META = 20_952
EXPECTED_GENERATED_META = EXPECTED_FULL_TRAIN_ANCHORS - EXPECTED_REUSED_META
KEY_COLUMNS = ("case_id", "station", "lead_h")
SUBMISSION_COLUMNS = (*KEY_COLUMNS, "hs_pred")


class KMADeploymentError(RuntimeError):
    """Fail-closed error for the fixed secondary deployment path."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_columns_sha256() -> str:
    payload = json.dumps(list(compact_source_feature_columns()), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_deployment_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise KMADeploymentError("deployment config must be a JSON object")
    validate_deployment_config(config)
    return config


def validate_deployment_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "1.0" or config.get("experiment_id") != EXPERIMENT_ID:
        raise KMADeploymentError("unexpected deployment config identity")
    objective = config["objective"]
    if objective.get("classification") != "low_confidence_secondary_research_candidate":
        raise KMADeploymentError("secondary-candidate classification changed")
    if (
        objective.get("promotion_claimed") is not False
        or objective.get("upload_allowed") is not False
    ):
        raise KMADeploymentError("this generation cannot claim promotion or upload authority")

    evidence = config["validation_evidence"]
    if evidence.get("required_decision") != "NO_GO_EXACT_INCUMBENT":
        raise KMADeploymentError("v2 NO-GO caveat is missing")
    if evidence.get("deployment_alpha_rule") != "median_of_three_inner_selected_fold_alphas":
        raise KMADeploymentError("deployment-alpha rule changed")
    if float(evidence.get("deployment_alpha")) != DEPLOYMENT_ALPHA:
        raise KMADeploymentError("deployment alpha changed")
    if sorted(float(value) for value in evidence.get("inner_selected_fold_alphas", ())) != [
        0.2,
        0.4,
        0.4,
    ]:
        raise KMADeploymentError("v2 fold alpha evidence changed")

    source = config["sealed_source_reuse"]
    if source.get("fit_new_source_model") is not False:
        raise KMADeploymentError("a new source model fit is prohibited")
    if source["source_model"].get("expected_feature_count") != 447:
        raise KMADeploymentError("source feature count changed")
    if source["source_model"].get("expected_output_count") != len(LEADS):
        raise KMADeploymentError("source output count changed")
    if source["source_feature_medians"].get("expected_count") != 447:
        raise KMADeploymentError("source median count changed")
    if source.get("feature_columns_sha256") != feature_columns_sha256():
        raise KMADeploymentError("source feature schema changed")
    for forbidden_count in (
        "period_features",
        "source_station_or_proxy_features",
        "absolute_calendar_features",
        "missingness_fingerprint_features",
    ):
        if source.get(forbidden_count) != 0:
            raise KMADeploymentError(f"forbidden source input enabled: {forbidden_count}")

    refit = config["full_refit"]
    expected_refit = {
        "single_refit_generation": True,
        "model_count": 2,
        "one_per_active_lead": True,
        "active_leads": list(ACTIVE_LEADS),
        "family": "Ridge",
        "ridge_alpha": 10.0,
        "fit_intercept": False,
        "solver": "cholesky",
        "standardize": False,
        "hyperparameter_grid_size": 0,
        "expected_full_train_anchor_count": EXPECTED_FULL_TRAIN_ANCHORS,
        "expected_reused_source_meta_count": EXPECTED_REUSED_META,
        "expected_generated_source_meta_count": EXPECTED_GENERATED_META,
    }
    for name, expected in expected_refit.items():
        if refit.get(name) != expected:
            raise KMADeploymentError(f"full-refit contract changed: {name}")
    expected_design = ["source_residual", "station_G-ORS", "station_I-ORS", "station_S-ORS"]
    if refit.get("design_columns") != expected_design:
        raise KMADeploymentError("Ridge design changed")

    inference = config["test_inference"]
    expected_inference = {
        "history_hours": 48,
        "relative_grid_minutes": 30,
        "history_rows_including_anchor": HISTORY_ROWS,
        "lead_order": list(LEADS),
        "active_leads": list(ACTIVE_LEADS),
        "byte_exact_no_op_leads": list(NO_OP_LEADS),
        "deployment_alpha": DEPLOYMENT_ALPHA,
        "unsupported_case_policy": "exact_incumbent_no_op_on_all_leads",
        "absolute_timestamp_reconstruction": False,
        "external_test_join": False,
        "same_case_context_only": True,
        "expected_case_count": EXPECTED_CASES,
        "expected_row_count": EXPECTED_ROWS,
    }
    for name, expected in expected_inference.items():
        if inference.get(name) != expected:
            raise KMADeploymentError(f"test-inference contract changed: {name}")

    execution = config["execution"]
    if execution.get("run_once") is not True:
        raise KMADeploymentError("run-once contract is missing")
    if execution.get("source_model_fit_count") != 0 or execution.get("ridge_model_fit_count") != 2:
        raise KMADeploymentError("model-fit count changed")
    if execution.get("submission_upload_allowed") is not False:
        raise KMADeploymentError("upload must remain prohibited")
    if execution.get("incumbent_overwrite_allowed") is not False:
        raise KMADeploymentError("incumbent overwrite must remain prohibited")
    prohibitions = config["prohibitions"]
    if not all(bool(value) for value in prohibitions.values()):
        raise KMADeploymentError("a required prohibition is disabled")
    incumbent = config["frozen_inputs"]["incumbent_submission"]["path"]
    output = config["artifacts"]["submission_path"]
    if Path(incumbent).as_posix().lower() == Path(output).as_posix().lower():
        raise KMADeploymentError("candidate output would overwrite the incumbent")


def _relative_asof(
    query_steps: np.ndarray,
    observations: pd.DataFrame,
    columns: Sequence[str],
    *,
    tolerance_minutes: int,
) -> pd.DataFrame:
    selected = observations.loc[:, ["step_minute", *columns]].copy()
    selected["step_minute"] = pd.to_numeric(selected["step_minute"], errors="raise").astype("int64")
    selected = selected.sort_values("step_minute").rename(columns={"step_minute": "_obs_step"})
    if selected["_obs_step"].duplicated().any():
        raise KMADeploymentError("relative context contains duplicate observation steps")
    query = pd.DataFrame({"_query_step": np.asarray(query_steps, dtype=np.int64)})
    merged = pd.merge_asof(
        query,
        selected,
        left_on="_query_step",
        right_on="_obs_step",
        direction="backward",
        tolerance=int(tolerance_minutes),
        allow_exact_matches=True,
    )
    available = merged["_obs_step"].notna()
    if available.any():
        delay = merged.loc[available, "_query_step"] - merged.loc[available, "_obs_step"]
        if delay.lt(0).any() or delay.gt(int(tolerance_minutes)).any():
            raise AssertionError("relative as-of join used future or stale observations")
    return merged.loc[:, list(columns)].reset_index(drop=True)


def extract_relative_test_history(case: pd.DataFrame) -> pd.DataFrame:
    """Map one anonymous 10-minute case to the frozen causal 30-minute surface."""

    required = {
        "case_id",
        "station",
        "step_minute",
        *SCALAR_VARIABLES,
        *DIRECTION_VARIABLES,
    }
    if not required <= set(case.columns):
        raise KMADeploymentError(
            f"test case is missing columns: {sorted(required - set(case.columns))}"
        )
    ordered = case.sort_values("step_minute").reset_index(drop=True)
    expected_steps = np.arange(-2880, 1, 10, dtype=np.int64)
    if len(ordered) != len(expected_steps) or not np.array_equal(
        ordered["step_minute"].to_numpy(dtype=np.int64), expected_steps
    ):
        raise KMADeploymentError("anonymous case does not have the official relative grid")
    if ordered["case_id"].nunique() != 1 or ordered["station"].nunique() != 1:
        raise KMADeploymentError("relative history crossed a case or station boundary")
    query_steps = np.arange(-2880, 1, 30, dtype=np.int64)
    if len(query_steps) != HISTORY_ROWS:
        raise AssertionError("30-minute relative grid length changed")
    wave_slots = ordered.loc[ordered["step_minute"].mod(20).eq(0)]
    wave = _relative_asof(
        query_steps,
        wave_slots,
        ("hs", "hmax", "wvdir"),
        tolerance_minutes=19,
    )
    atmosphere = _relative_asof(
        query_steps,
        ordered,
        ("wspd", "gust", "wdir", "airt", "relh", "caph"),
        tolerance_minutes=9,
    )
    history = pd.concat([wave, atmosphere], axis=1)
    return history.loc[:, [*SCALAR_VARIABLES, *DIRECTION_VARIABLES]]


def build_test_source_features(
    context: pd.DataFrame,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Create one 447-feature row per case without absolute timestamps or cross-case joins."""

    if context.empty:
        raise KMADeploymentError("test context is empty")
    case_order = context["case_id"].drop_duplicates().astype(str).tolist()
    if len(case_order) != EXPECTED_CASES:
        raise KMADeploymentError("official test context must contain exactly 200 cases")
    rows: list[dict[str, Any]] = []
    columns = compact_source_feature_columns()
    for number, case_id in enumerate(case_order, start=1):
        case = context.loc[context["case_id"].astype(str).eq(case_id)]
        history = extract_relative_test_history(case)
        summary = summarize_common_history(history)
        current = pd.to_numeric(
            case.loc[case["step_minute"].eq(0), "hs"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if current.shape != (1,):
            raise KMADeploymentError("test case has no unique current hs")
        station = str(case["station"].iloc[0])
        raw = np.asarray([summary[name] for name in columns], dtype=np.float64)
        supported = bool(np.isfinite(current[0]) and not np.isinf(raw).any())
        row: dict[str, Any] = {
            "case_id": case_id,
            "station": station,
            "current_hs": float(current[0]),
            "source_supported": supported,
        }
        row.update(summary)
        rows.append(row)
        if progress is not None and (number == len(case_order) or number % 25 == 0):
            progress(number, len(case_order))
    result = pd.DataFrame(
        rows,
        columns=["case_id", "station", "current_hs", "source_supported", *columns],
    )
    if result["case_id"].duplicated().any() or len(result) != EXPECTED_CASES:
        raise KMADeploymentError("test source feature keys are invalid")
    return result


def combine_full_training_meta(
    anchors: pd.DataFrame,
    reused_meta: pd.DataFrame,
    generated_meta: pd.DataFrame,
) -> pd.DataFrame:
    expected_meta_columns = ["anchor_id", *META_COLUMNS]
    for name, frame in (("reused", reused_meta), ("generated", generated_meta)):
        if list(frame.columns) != expected_meta_columns:
            raise KMADeploymentError(f"{name} source meta schema changed")
        if frame["anchor_id"].duplicated().any():
            raise KMADeploymentError(f"{name} source meta contains duplicate anchor ids")
    if len(anchors) != EXPECTED_FULL_TRAIN_ANCHORS or anchors["anchor_id"].duplicated().any():
        raise KMADeploymentError("full training anchor contract changed")
    if len(reused_meta) != EXPECTED_REUSED_META or len(generated_meta) != EXPECTED_GENERATED_META:
        raise KMADeploymentError("reused/generated source-meta counts changed")
    overlap = np.intersect1d(reused_meta["anchor_id"], generated_meta["anchor_id"])
    if len(overlap):
        raise KMADeploymentError("reused and generated source meta overlap")
    combined = pd.concat([reused_meta, generated_meta], ignore_index=True)
    combined = combined.sort_values("anchor_id").reset_index(drop=True)
    expected_ids = anchors["anchor_id"].sort_values().to_numpy(dtype=np.int64)
    if not np.array_equal(combined["anchor_id"].to_numpy(dtype=np.int64), expected_ids):
        raise KMADeploymentError("full source meta does not cover every training anchor")
    values = combined.loc[:, list(META_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 30.0).any():
        raise KMADeploymentError("full training source meta is invalid")
    return combined


def build_full_ridge_frame(anchors: pd.DataFrame, source_meta: pd.DataFrame) -> pd.DataFrame:
    required = {
        "anchor_id",
        "station",
        "current_hs",
        *[f"target_{lead}" for lead in ACTIVE_LEADS],
    }
    if not required <= set(anchors.columns):
        raise KMADeploymentError("training anchors lack Ridge targets")
    anchor_columns = [
        "anchor_id",
        "station",
        "current_hs",
        *[f"target_{lead}" for lead in ACTIVE_LEADS],
    ]
    merged = anchors.loc[:, anchor_columns].merge(
        source_meta,
        on="anchor_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != EXPECTED_FULL_TRAIN_ANCHORS:
        raise KMADeploymentError("Ridge full-refit frame lost training anchors")
    rows: list[pd.DataFrame] = []
    for lead in ACTIVE_LEADS:
        part = merged.loc[:, ["anchor_id", "station", "current_hs"]].copy()
        part["lead_h"] = int(lead)
        part["source_prediction"] = merged[f"kma_source_hs_pred_{lead}h"].to_numpy(dtype=np.float64)
        part["target_hs"] = merged[f"target_{lead}"].to_numpy(dtype=np.float64)
        rows.append(part)
    result = pd.concat(rows, ignore_index=True)
    numeric = result.loc[:, ["current_hs", "source_prediction", "target_hs"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all():
        raise KMADeploymentError("Ridge full-refit frame contains non-finite values")
    return result


def calibrators_to_payload(
    calibrators: Mapping[int, RidgeAffineCalibrator],
) -> dict[str, Any]:
    if set(calibrators) != set(ACTIVE_LEADS):
        raise KMADeploymentError("exactly two per-lead calibrators are required")
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "single_refit_generation": True,
        "model_count": 2,
        "one_per_active_lead": True,
        "active_leads": list(ACTIVE_LEADS),
        "calibrators": {str(lead): calibrators[lead].to_dict() for lead in ACTIVE_LEADS},
    }


def calibrators_from_payload(payload: Mapping[str, Any]) -> dict[int, RidgeAffineCalibrator]:
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise KMADeploymentError("saved calibrator experiment id changed")
    if payload.get("single_refit_generation") is not True:
        raise KMADeploymentError("saved generation flag changed")
    if payload.get("model_count") != 2 or payload.get("one_per_active_lead") is not True:
        raise KMADeploymentError("saved calibrator count/structure changed")
    raw = payload.get("calibrators", {})
    if set(raw) != {str(lead) for lead in ACTIVE_LEADS}:
        raise KMADeploymentError("saved calibrator leads changed")
    result = {int(lead): RidgeAffineCalibrator.from_dict(value) for lead, value in raw.items()}
    for lead, calibrator in result.items():
        if lead != calibrator.lead_h or calibrator.fit_rows != EXPECTED_FULL_TRAIN_ANCHORS:
            raise KMADeploymentError("saved per-lead calibrator metadata changed")
    return result


def validate_candidate_submission(
    candidate: pd.DataFrame,
    test_index: pd.DataFrame,
    incumbent: pd.DataFrame,
    *,
    supported_cases: Sequence[str],
) -> dict[str, Any]:
    if list(candidate.columns) != list(SUBMISSION_COLUMNS):
        raise KMADeploymentError("candidate submission schema changed")
    if list(test_index.columns) != list(KEY_COLUMNS):
        raise KMADeploymentError("test index schema changed")
    if list(incumbent.columns) != list(SUBMISSION_COLUMNS):
        raise KMADeploymentError("incumbent submission schema changed")
    if len(candidate) != EXPECTED_ROWS or not candidate[list(KEY_COLUMNS)].equals(test_index):
        raise KMADeploymentError("candidate keys/order differ from the official test index")
    if not incumbent[list(KEY_COLUMNS)].equals(test_index):
        raise KMADeploymentError("incumbent keys/order differ from the official test index")
    if candidate.duplicated(list(KEY_COLUMNS)).any():
        raise KMADeploymentError("candidate contains duplicate keys")
    values = candidate["hs_pred"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 30.0).any():
        raise KMADeploymentError("candidate predictions are outside [0, 30]")
    leads = candidate["lead_h"].to_numpy(dtype=np.int64)
    candidate_values = candidate["hs_pred"].to_numpy(dtype=np.float64)
    incumbent_values = incumbent["hs_pred"].to_numpy(dtype=np.float64)
    no_op = np.isin(leads, NO_OP_LEADS)
    if not np.array_equal(candidate_values[no_op], incumbent_values[no_op]):
        raise KMADeploymentError("short-lead predictions are not exactly the incumbent")
    supported = {str(value) for value in supported_cases}
    unsupported = ~candidate["case_id"].astype(str).isin(supported).to_numpy()
    if not np.array_equal(candidate_values[unsupported], incumbent_values[unsupported]):
        raise KMADeploymentError("unsupported cases are not exact incumbent no-ops")
    per_case = candidate.groupby("case_id", sort=False, observed=True)["lead_h"].agg(tuple)
    if (
        len(per_case) != EXPECTED_CASES
        or not per_case.map(lambda value: tuple(value) == LEADS).all()
    ):
        raise KMADeploymentError("candidate does not contain the official six leads per case")
    return {
        "rows": int(len(candidate)),
        "cases": int(candidate["case_id"].nunique()),
        "supported_cases": int(len(supported)),
        "unsupported_cases": int(EXPECTED_CASES - len(supported)),
        "no_op_rows": int(no_op.sum() + (unsupported & ~no_op).sum()),
        "modified_active_rows": int((~no_op & ~unsupported).sum()),
        "prediction_min": float(values.min()),
        "prediction_max": float(values.max()),
    }


def render_submission_preserving_noop_lines(
    incumbent_bytes: bytes,
    candidate: pd.DataFrame,
    *,
    supported_cases: Sequence[str],
) -> bytes:
    """Rewrite only supported 18h/24h values and copy every no-op line byte-for-byte."""

    try:
        text = incumbent_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise KMADeploymentError("incumbent submission is not UTF-8") from exc
    lines = text.splitlines(keepends=True)
    if len(lines) != len(candidate) + 1:
        raise KMADeploymentError("incumbent byte row count changed")
    header_without_eol = lines[0].rstrip("\r\n")
    if next(csv.reader([header_without_eol])) != list(SUBMISSION_COLUMNS):
        raise KMADeploymentError("incumbent byte header changed")
    supported = {str(value) for value in supported_cases}
    output = [lines[0]]
    for line, row in zip(lines[1:], candidate.itertuples(index=False), strict=True):
        ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        body = line[: -len(ending)] if ending else line
        parsed = next(csv.reader([body]))
        if len(parsed) != 4:
            raise KMADeploymentError("incumbent CSV line is malformed")
        key = (parsed[0], parsed[1], int(parsed[2]))
        expected = (str(row.case_id), str(row.station), int(row.lead_h))
        if key != expected:
            raise KMADeploymentError("candidate row order differs from incumbent bytes")
        modify = int(row.lead_h) in ACTIVE_LEADS and str(row.case_id) in supported
        if not modify:
            output.append(line)
            continue
        prefix = body.rsplit(",", 1)[0]
        output.append(f"{prefix},{format(float(row.hs_pred), '.17g')}{ending}")
    return "".join(output).encode("utf-8")


def count_byte_exact_noop_lines(
    incumbent_bytes: bytes,
    candidate_bytes: bytes,
    candidate: pd.DataFrame,
    *,
    supported_cases: Sequence[str],
) -> int:
    original = incumbent_bytes.splitlines(keepends=True)
    rendered = candidate_bytes.splitlines(keepends=True)
    if len(original) != len(rendered) or len(rendered) != len(candidate) + 1:
        raise KMADeploymentError("rendered submission line count changed")
    supported = {str(value) for value in supported_cases}
    exact = 0
    for index, row in enumerate(candidate.itertuples(index=False), start=1):
        no_op = int(row.lead_h) in NO_OP_LEADS or str(row.case_id) not in supported
        if no_op:
            if original[index] != rendered[index]:
                raise KMADeploymentError("a required no-op CSV line changed bytes")
            exact += 1
    return exact
