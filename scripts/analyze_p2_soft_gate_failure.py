"""Diagnose the failed P2 public-state soft gate without retuning a candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from run_p2_public_state_soft_gate import CONTRIBUTORS, _load_oof_stack
from scipy.stats import spearmanr, wasserstein_distance

from p2_restore.data import load_p2_data
from p2_restore.regime_gate import (
    STATE_FEATURES,
    RobustStateTransform,
    build_public_state_features,
    fit_soft_gate,
    predict_simplex_baseline,
    predict_soft_gate,
    soft_gate_weights,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(truth)) ** 2)))


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _inner_noop_rmse(frame: pd.DataFrame, outer: str) -> float:
    blocks = [block for block in sorted(frame["block"].unique()) if block != outer]
    squared_error = 0.0
    count = 0
    for held in blocks:
        fit_block = next(block for block in blocks if block != held)
        train = frame["block"].to_numpy() == fit_block
        validation = frame["block"].to_numpy() == held
        prediction = predict_simplex_baseline(frame.loc[train], frame.loc[validation], CONTRIBUTORS)
        error = prediction - frame.loc[validation, "truth"].to_numpy(float)
        squared_error += float(error @ error)
        count += len(error)
    return float(np.sqrt(squared_error / count))


def _decomposition_rows(
    frame: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray, grouping: list[str]
) -> list[dict[str, object]]:
    working = frame.loc[:, grouping].copy()
    error = baseline - frame["truth"].to_numpy(float)
    adjustment = candidate - baseline
    working["baseline_error_sq"] = error**2
    working["candidate_error_sq"] = (error + adjustment) ** 2
    working["alignment_sse"] = 2.0 * error * adjustment
    working["movement_sse"] = adjustment**2
    working["abs_adjustment"] = np.abs(adjustment)
    total_delta = float((working["candidate_error_sq"] - working["baseline_error_sq"]).sum())
    rows: list[dict[str, object]] = []
    for key, group in working.groupby(grouping, sort=True, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        base_sse = float(group["baseline_error_sq"].sum())
        candidate_sse = float(group["candidate_error_sq"].sum())
        alignment = float(group["alignment_sse"].sum())
        movement = float(group["movement_sse"].sum())
        row = {name: value for name, value in zip(grouping, values, strict=True)}
        row.update(
            {
                "rows": int(len(group)),
                "baseline_rmse": float(np.sqrt(base_sse / len(group))),
                "candidate_rmse": float(np.sqrt(candidate_sse / len(group))),
                "delta_rmse": float(
                    np.sqrt(candidate_sse / len(group)) - np.sqrt(base_sse / len(group))
                ),
                "alignment_mse": alignment / len(group),
                "movement_mse": movement / len(group),
                "delta_mse": (candidate_sse - base_sse) / len(group),
                "delta_sse": candidate_sse - base_sse,
                "delta_sse_share": (
                    (candidate_sse - base_sse) / total_delta if abs(total_delta) > 1e-15 else None
                ),
                "adjustment_rmse": float(np.sqrt(np.mean(group["abs_adjustment"] ** 2))),
                "adjustment_p95_abs": float(np.quantile(group["abs_adjustment"], 0.95)),
            }
        )
        if not np.isclose(alignment + movement, candidate_sse - base_sse, atol=1e-8):
            raise AssertionError("error decomposition does not reconcile")
        rows.append(row)
    return rows


def _state_labels(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    reference = frame.loc[frame["block"].eq("2024_sep_oct"), "abs_t1_t5"].dropna()
    q33, q67 = np.quantile(reference.to_numpy(float), [1 / 3, 2 / 3])
    values = frame["abs_t1_t5"].to_numpy(float)
    labels = np.full(len(frame), "missing", dtype=object)
    finite = np.isfinite(values)
    labels[finite & (values <= q33)] = "low"
    labels[finite & (values > q33) & (values < q67)] = "transition"
    labels[finite & (values >= q67)] = "high"
    return labels, {"q33": float(q33), "q67": float(q67)}


def _regularization_diagnostics(
    frame: pd.DataFrame,
    stored: pd.DataFrame,
    selected_by_outer: dict[str, float],
    grid: list[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]], np.ndarray]:
    rows: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    reconstructed = np.full(len(frame), np.nan)
    for outer in sorted(frame["block"].unique()):
        validation = frame["block"].to_numpy() == outer
        train = ~validation
        truth = frame.loc[validation, "truth"].to_numpy(float)
        baseline = predict_simplex_baseline(frame.loc[train], frame.loc[validation], CONTRIBUTORS)
        noop_rmse = _rmse(truth, baseline)
        inner_noop = _inner_noop_rmse(frame, outer)
        rows.append(
            {
                "outer_block": outer,
                "candidate": "no_op",
                "regularization": None,
                "rmse": noop_rmse,
                "delta_vs_no_op": 0.0,
                "selected": False,
            }
        )
        predictions: dict[float, np.ndarray] = {}
        for regularization in grid:
            gate = fit_soft_gate(
                frame.loc[train],
                prediction_columns=CONTRIBUTORS,
                regularization=regularization,
            )
            prediction = predict_soft_gate(gate, frame.loc[validation])
            predictions[regularization] = prediction
            rows.append(
                {
                    "outer_block": outer,
                    "candidate": f"lambda_{regularization:g}",
                    "regularization": regularization,
                    "rmse": _rmse(truth, prediction),
                    "delta_vs_no_op": _rmse(truth, prediction) - noop_rmse,
                    "selected": regularization == selected_by_outer[outer],
                }
            )
        selected = selected_by_outer[outer]
        reconstructed[validation] = predictions[selected]
        current = [row for row in rows if row["outer_block"] == outer]
        posthoc_best = min(current, key=lambda row: row["rmse"])
        selected_row = next(row for row in current if row["selected"])
        selections.append(
            {
                "outer_block": outer,
                "inner_no_op_rmse": inner_noop,
                "inner_selected_gate_rmse": None,
                "selected_regularization": selected,
                "outer_no_op_rmse": noop_rmse,
                "outer_selected_rmse": selected_row["rmse"],
                "outer_selected_delta": selected_row["delta_vs_no_op"],
                "posthoc_best_candidate": posthoc_best["candidate"],
                "posthoc_best_rmse": posthoc_best["rmse"],
                "outer_selection_regret": selected_row["rmse"] - posthoc_best["rmse"],
            }
        )
    if not np.allclose(reconstructed, stored["prediction"].to_numpy(float), atol=1e-10):
        raise AssertionError("selected outer gate predictions were not reproduced")
    return rows, selections, reconstructed


def _gate_dynamics(
    frame: pd.DataFrame, selected_by_outer: dict[str, float]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for outer in sorted(frame["block"].unique()):
        validation = frame["block"].to_numpy() == outer
        gate = fit_soft_gate(
            frame.loc[~validation],
            prediction_columns=CONTRIBUTORS,
            regularization=selected_by_outer[outer],
        )
        weights = soft_gate_weights(gate, frame.loc[validation])
        predictions = frame.loc[validation, CONTRIBUTORS].to_numpy(float)
        truth = frame.loc[validation, "truth"].to_numpy(float)
        layers = frame.loc[validation, "layer"].to_numpy(int)
        for layer in (2, 3, 4):
            keep = layers == layer
            prior = gate.layers[layer].prior
            current = weights[keep]
            floored = np.flatnonzero(prior < 1e-6)
            best = np.argmin((predictions[keep] - truth[keep, None]) ** 2, axis=1)
            floored_best = np.isin(best, floored) if len(floored) else np.zeros(len(best), bool)
            row: dict[str, object] = {
                "outer_block": outer,
                "layer": layer,
                "rows": int(keep.sum()),
                "regularization": selected_by_outer[outer],
                "mean_l1_weight_shift": float(np.mean(np.abs(current - prior).sum(axis=1))),
                "p95_l1_weight_shift": float(
                    np.quantile(np.abs(current - prior).sum(axis=1), 0.95)
                ),
                "max_weight_above_0_8_share": float(np.mean(current.max(axis=1) > 0.8)),
                "floored_contributors": [CONTRIBUTORS[index] for index in floored],
                "floored_contributor_is_row_oracle_share": float(np.mean(floored_best)),
            }
            for index, name in enumerate(CONTRIBUTORS):
                row[f"prior_{name}"] = float(prior[index])
                row[f"mean_{name}"] = float(current[:, index].mean())
            rows.append(row)
    return rows


def _feature_diagnostics(frame: pd.DataFrame) -> tuple[list[dict], dict[str, object]]:
    shift_rows: list[dict[str, object]] = []
    for outer in sorted(frame["block"].unique()):
        outer_rows = frame["block"].eq(outer)
        for feature in STATE_FEATURES:
            train = frame.loc[~outer_rows, feature].to_numpy(float)
            test = frame.loc[outer_rows, feature].to_numpy(float)
            train_finite = train[np.isfinite(train)]
            test_finite = test[np.isfinite(test)]
            scale = (
                float(np.subtract(*np.quantile(train_finite, [0.75, 0.25])))
                if len(train_finite)
                else 1.0
            )
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = float(np.std(train_finite)) if len(train_finite) else 1.0
            distance = (
                float(wasserstein_distance(train_finite, test_finite) / max(scale, 1e-12))
                if len(train_finite) and len(test_finite)
                else None
            )
            shift_rows.append(
                {
                    "outer_block": outer,
                    "feature": feature,
                    "normalized_wasserstein": distance,
                    "train_missing_rate": float(np.mean(~np.isfinite(train))),
                    "outer_missing_rate": float(np.mean(~np.isfinite(test))),
                    "missing_rate_delta": float(
                        np.mean(~np.isfinite(test)) - np.mean(~np.isfinite(train))
                    ),
                }
            )

    unique = frame.loc[:, ["time", *STATE_FEATURES]].drop_duplicates("time")
    values = unique.loc[:, STATE_FEATURES]
    pairs: list[dict[str, object]] = []
    for left_number, left in enumerate(STATE_FEATURES):
        for right in STATE_FEATURES[left_number + 1 :]:
            keep = values[left].notna() & values[right].notna()
            correlation = (
                float(spearmanr(values.loc[keep, left], values.loc[keep, right]).statistic)
                if int(keep.sum()) >= 10
                else None
            )
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "spearman": correlation,
                    "abs_spearman": abs(correlation) if correlation is not None else None,
                }
            )
    pairs.sort(key=lambda row: row["abs_spearman"] or -1, reverse=True)
    transform = RobustStateTransform.fit(unique, STATE_FEATURES)
    design = transform.transform(unique)
    singular = np.linalg.svd(design, compute_uv=False)
    effective_rank = int(np.sum(singular > singular.max() * 1e-10))
    missing = ~np.isfinite(values.to_numpy(float))
    redundancy = {
        "unique_timestamps": int(len(unique)),
        "design_columns": int(design.shape[1]),
        "effective_rank": effective_rank,
        "condition_number_nonzero": float(singular[0] / singular[effective_rank - 1]),
        "unique_missing_patterns": int(len(np.unique(missing, axis=0))),
        "top_abs_spearman_pairs": pairs[:10],
        "feature_missing_rates": {
            feature: float(values[feature].isna().mean()) for feature in STATE_FEATURES
        },
    }
    return shift_rows, redundancy


def _mapping_stability(frame: pd.DataFrame) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for (block, layer, state), group in frame.groupby(["block", "layer", "state"], sort=True):
        truth = group["truth"].to_numpy(float)
        scores = {name: _rmse(truth, group[name].to_numpy(float)) for name in CONTRIBUTORS}
        best = min(scores, key=scores.get)
        rows.append(
            {
                "block": block,
                "layer": int(layer),
                "state": state,
                "rows": int(len(group)),
                "best_contributor": best,
                "best_rmse": scores[best],
                "worst_rmse": max(scores.values()),
                "spread_rmse": max(scores.values()) - min(scores.values()),
            }
        )
    winner_frame = pd.DataFrame(rows)
    consistency: list[dict[str, object]] = []
    for (layer, state), group in winner_frame.groupby(["layer", "state"], sort=True):
        consistency.append(
            {
                "layer": int(layer),
                "state": state,
                "blocks": int(group["block"].nunique()),
                "unique_winners": int(group["best_contributor"].nunique()),
                "winners": sorted(group["best_contributor"].unique()),
                "minimum_rows": int(group["rows"].min()),
            }
        )
    stable = [row for row in consistency if row["blocks"] == 3]
    return {
        "cell_winners": rows,
        "winner_consistency": consistency,
        "three_block_cells": len(stable),
        "three_block_same_winner_cells": sum(row["unique_winners"] == 1 for row in stable),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-dir", type=Path, default=Path("artifacts/p2_public_state_soft_gate_v1")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/p2_soft_gate_failure_diagnostic/result.json")
    )
    args = parser.parse_args()
    source_result = json.loads((args.experiment_dir / "result.json").read_text(encoding="utf-8"))
    stored = pd.read_parquet(args.experiment_dir / "oof.parquet")
    stored["time"] = pd.to_datetime(stored["time"], utc=True)
    stack = _load_oof_stack()
    data = load_p2_data(args.data_dir)
    public = build_public_state_features(data.observations, stack[["time", "layer"]])
    stack = pd.concat(
        [stack.reset_index(drop=True), public.loc[:, STATE_FEATURES].reset_index(drop=True)], axis=1
    )
    if not stack[["time", "layer", "block"]].equals(stored[["time", "layer", "block"]]):
        raise ValueError("diagnostic OOF keys differ from experiment OOF")
    baseline = stored["baseline_prediction"].to_numpy(float)
    candidate = stored["prediction"].to_numpy(float)
    labels, thresholds = _state_labels(stack)
    stack["state"] = labels
    local_time = pd.to_datetime(stack["time"], utc=True).dt.tz_convert("Asia/Seoul")
    stack["kst_week"] = local_time.dt.tz_localize(None).dt.to_period("W-SUN").astype(str)

    grid = [
        float(value)
        for value in json.loads(
            Path("configs/experiments/p2_public_state_soft_gate_v1.json").read_text(
                encoding="utf-8"
            )
        )["gate"]["regularization_grid"]
    ]
    selected = {
        key: float(value)
        for key, value in source_result["selected_regularization_by_outer"].items()
    }
    regularization_rows, selections, reconstructed = _regularization_diagnostics(
        stack, stored, selected, grid
    )
    for row in selections:
        row["inner_selected_gate_rmse"] = next(
            item["rmse"]
            for item in source_result["inner_scores"][row["outer_block"]]
            if float(item["regularization"]) == row["selected_regularization"]
        )
        row["inner_would_choose_no_op"] = row["inner_no_op_rmse"] <= row["inner_selected_gate_rmse"]
    if not np.allclose(reconstructed, candidate, atol=1e-10):
        raise AssertionError("candidate reproduction failed")

    block_rows = _decomposition_rows(stack, baseline, candidate, ["block"])
    layer_rows = _decomposition_rows(stack, baseline, candidate, ["layer"])
    state_rows = _decomposition_rows(stack, baseline, candidate, ["block", "layer", "state"])
    weekly_rows = _decomposition_rows(stack, baseline, candidate, ["block", "layer", "kst_week"])
    weekly_rows.sort(key=lambda row: row["delta_sse"], reverse=True)
    feature_shift, redundancy = _feature_diagnostics(stack)
    dynamics = _gate_dynamics(stack, selected)
    mapping = _mapping_stability(stack)

    result: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "research-only posthoc failure diagnostic; no candidate retuning",
        "uploaded": False,
        "external_values_used": False,
        "target_layer_hidden_values_used": False,
        "metric_reconciliation": {
            "rows": int(len(stack)),
            "baseline_rmse": _rmse(stack["truth"], baseline),
            "candidate_rmse": _rmse(stack["truth"], candidate),
            "delta_rmse": _rmse(stack["truth"], candidate) - _rmse(stack["truth"], baseline),
            "candidate_reproduction_max_abs_error": float(
                np.max(np.abs(reconstructed - candidate))
            ),
        },
        "state_thresholds_from_2024_same_season": thresholds,
        "error_decomposition_by_block": block_rows,
        "error_decomposition_by_layer": layer_rows,
        "state_cell_decomposition": state_rows,
        "top_weekly_harm_cells": weekly_rows[:15],
        "regularization_outer_diagnostic": regularization_rows,
        "selection_diagnostic": selections,
        "gate_dynamics": dynamics,
        "feature_shift": feature_shift,
        "feature_redundancy": redundancy,
        "state_to_winner_mapping": mapping,
        "verified_findings": {
            "no_op_absent_from_original_grid": True,
            "inner_no_op_would_win_outer_count": int(
                sum(row["inner_would_choose_no_op"] for row in selections)
            ),
            "selected_at_max_regularization_outer_count": int(
                sum(row["selected_regularization"] == max(grid) for row in selections)
            ),
            "floored_expert_outer_layer_cells": int(
                sum(bool(row["floored_contributors"]) for row in dynamics)
            ),
            "winner_mapping_stable_share": (
                mapping["three_block_same_winner_cells"] / mapping["three_block_cells"]
                if mapping["three_block_cells"]
                else None
            ),
        },
        "interpretation": {
            "verified": (
                "The executed model tested forced, prior-anchored linear gating, not the broader "
                "hypothesis that public physical state can condition contributor skill."
            ),
            "likely": (
                "Failure is driven by a combination of wrong-direction adjustments in the 2024 "
                "same-season block, unstable state-to-expert mapping across only three trajectories, "
                "and structural exclusion of contributors whose simplex prior is near zero."
            ),
            "unresolved": (
                "Whether a cross-fitted residual-correction gate with an exact no-op arm and positive "
                "weight floors generalizes remains untested."
            ),
        },
        "source_hashes": {
            "experiment_result": _sha256(args.experiment_dir / "result.json"),
            "experiment_oof": _sha256(args.experiment_dir / "oof.parquet"),
            "observations.csv": _sha256(args.data_dir / "observations.csv"),
        },
    }
    _write_json(args.output, result)
    print(
        json.dumps(
            {"status": "passed", "output": args.output.as_posix(), **result["verified_findings"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
