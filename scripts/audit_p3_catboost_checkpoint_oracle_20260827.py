"""Read-only P3 CatBoost checkpoint oracle audit.

This diagnostic replays earlier tree prefixes from the already saved CatBoost models
on the historical local validation windows.  It never opens P3 test, sample, or
submission files.  The validation targets are used only to quantify oracle headroom;
therefore the output is diagnostic evidence, not promotion evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from p3_wave.meaningful_learning_curve import HYPOTHESES
from p3_wave.persistence_shrink import (
    LongLeadPersistenceShrink,
    apply_long_lead_persistence_shrink,
)
from p3_wave.validation import expand_leads, rmse


PREFIXES = (0.40, 0.55, 0.70, 0.85, 1.00)
SEEDS = (20260816, 20260817, 20260818)
# Lexical order matches the canonical KEYS sort used by the sealed OOF artifact.
FOLDS = ("2024_h2_storm", "2025_h1", "winter_transition")
TREE_FRACTIONS = tuple(round(value / 20.0, 2) for value in range(1, 21))
KEYS = ["fold", "anchor_id", "station", "lead_h"]


def _load_base(root: Path):
    path = root / "scripts/run_p3_corrected_repeated_forward_catboost_v1.py"
    spec = importlib.util.spec_from_file_location("p3_checkpoint_audit_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_model(path: Path) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def _tree_count(total: int, fraction: float) -> int:
    return max(1, min(total, int(round(total * fraction))))


def _fold_predictions(
    *,
    base: Any,
    root: Path,
    artifact: Path,
    prefix_tag: str,
    seed: int,
    fold: str,
    validation_ids: np.ndarray,
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[float, np.ndarray], dict[float, np.ndarray], dict[str, int]]:
    x_valid, _, valid_meta = expand_leads(
        features, anchors, validation_ids, feature_columns
    )
    feature_lookup = features.set_index("anchor_id")
    multi_x_valid = feature_lookup.loc[
        validation_ids, ["station", *feature_columns]
    ].reset_index(drop=True)
    multi_x_valid["station"] = multi_x_valid["station"].astype(str)

    model_dir = artifact / "models" / prefix_tag / f"seed_{seed}" / "folds" / fold
    single = _load_model(model_dir / "single.cbm")
    multi = _load_model(model_dir / "multi.cbm")
    single_total = int(single.tree_count_)
    multi_total = int(multi.tree_count_)

    anchor_lookup = anchors.set_index("anchor_id")
    current_case = anchor_lookup.loc[validation_ids, "current_hs"].to_numpy(float)
    meta = valid_meta.copy()
    meta["fold"] = fold
    meta = meta.sort_values(KEYS).reset_index(drop=True)

    single_predictions: dict[float, np.ndarray] = {}
    multi_predictions: dict[float, np.ndarray] = {}
    for fraction in TREE_FRACTIONS:
        single_trees = _tree_count(single_total, fraction)
        multi_trees = _tree_count(multi_total, fraction)
        single_absolute = np.clip(
            valid_meta["current_hs"].to_numpy(float)
            + np.asarray(
                single.predict(base._cat_frame(x_valid), ntree_end=single_trees),
                dtype=float,
            ),
            0.0,
            30.0,
        )
        single_frame = valid_meta[KEYS[1:]].copy()
        single_frame["fold"] = fold
        single_frame["prediction"] = single_absolute
        single_frame = single_frame.sort_values(KEYS).reset_index(drop=True)
        single_predictions[fraction] = single_frame["prediction"].to_numpy(float)

        multi_delta = np.asarray(
            multi.predict(multi_x_valid, ntree_end=multi_trees), dtype=float
        )
        multi_absolute = np.clip(current_case[:, None] + multi_delta, 0.0, 30.0)
        multi_frame = pd.DataFrame(
            {
                "anchor_id": np.repeat(validation_ids, 6),
                "station": np.repeat(
                    anchor_lookup.loc[validation_ids, "station"].astype(str).to_numpy(),
                    6,
                ),
                "lead_h": np.tile(np.asarray((3, 6, 9, 12, 18, 24), int), len(validation_ids)),
                "fold": fold,
                "prediction": multi_absolute.reshape(-1),
            }
        ).sort_values(KEYS).reset_index(drop=True)
        if not multi_frame[KEYS].equals(meta[KEYS]):
            raise AssertionError(f"multi prediction key mismatch: {prefix_tag}/{seed}/{fold}")
        multi_predictions[fraction] = multi_frame["prediction"].to_numpy(float)

    return meta, single_predictions, multi_predictions, {
        "single": single_total,
        "multi": multi_total,
    }


def _candidate_prediction(
    *,
    name: str,
    lead_h: np.ndarray,
    single_prediction: np.ndarray,
    multi_prediction: np.ndarray,
    persistence: np.ndarray,
) -> np.ndarray:
    if name == "single_horizon_residual_head":
        raw = single_prediction
    elif name == "multi_trajectory_residual_head":
        raw = multi_prediction
    elif name == "fixed_horizon_splice":
        raw = np.where(np.isin(lead_h, [3, 6, 9, 12]), multi_prediction, single_prediction)
    else:
        raise KeyError(name)
    return apply_long_lead_persistence_shrink(
        raw,
        persistence,
        lead_h,
        config=LongLeadPersistenceShrink(weight=0.2, active_leads=(12, 18, 24)),
    )


def _rmse_by_fold(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    scored = frame[KEYS + ["target_hs"]].copy()
    scored["prediction"] = prediction
    return {
        str(fold): float(rmse(group["target_hs"], group["prediction"]))
        for fold, group in scored.groupby("fold", sort=True, observed=True)
    }


def run(root: Path, output: Path) -> dict[str, Any]:
    artifact = root / "artifacts/p3_meaningful_learning_curve_20260823_v1"
    features = pd.read_parquet(root / "artifacts/p3/features_all20_v1/train_features.parquet")
    anchors = pd.read_parquet(root / "artifacts/p3/features_all20_v1/train_anchors.parquet")
    validation_keys = pd.read_parquet(artifact / "validation_keys.parquet")
    original_oof = pd.read_parquet(artifact / "oof/learning_curve_oof.parquet")
    feature_columns = json.loads((artifact / "feature_columns.json").read_text(encoding="utf-8"))[
        "columns"
    ]
    base = _load_base(root)

    fold_ids = {
        fold: validation_keys.loc[
            validation_keys["fold"].astype(str).eq(fold), "anchor_id"
        ].to_numpy(dtype=np.int64)
        for fold in FOLDS
    }
    result: dict[str, Any] = {
        "schema_version": "p3_catboost_checkpoint_oracle.v1",
        "scope": "historical_local_validation_only",
        "promotion_evidence": False,
        "official_test_sample_submission_reads": 0,
        "tree_fraction_grid": list(TREE_FRACTIONS),
        "prefixes": {},
    }

    for prefix in PREFIXES:
        prefix_tag = f"p{int(round(prefix * 100)):03d}"
        reference = original_oof.loc[
            np.isclose(original_oof["prefix_fraction"].to_numpy(float), prefix)
        ].sort_values(KEYS).reset_index(drop=True)
        if len(reference) != 1_086:
            raise ValueError(f"unexpected reference rows for {prefix}: {len(reference)}")

        seed_single: dict[int, dict[float, list[np.ndarray]]] = {
            seed: {fraction: [] for fraction in TREE_FRACTIONS} for seed in SEEDS
        }
        seed_multi: dict[int, dict[float, list[np.ndarray]]] = {
            seed: {fraction: [] for fraction in TREE_FRACTIONS} for seed in SEEDS
        }
        tree_counts: dict[str, int] | None = None
        for seed in SEEDS:
            seed_keys: list[pd.DataFrame] = []
            for fold in FOLDS:
                meta, single_map, multi_map, counts = _fold_predictions(
                    base=base,
                    root=root,
                    artifact=artifact,
                    prefix_tag=prefix_tag,
                    seed=seed,
                    fold=fold,
                    validation_ids=fold_ids[fold],
                    features=features,
                    anchors=anchors,
                    feature_columns=feature_columns,
                )
                tree_counts = counts if tree_counts is None else tree_counts
                seed_keys.append(meta[KEYS])
                for fraction in TREE_FRACTIONS:
                    seed_single[seed][fraction].append(single_map[fraction])
                    seed_multi[seed][fraction].append(multi_map[fraction])
            keys = pd.concat(seed_keys, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
            if not keys.equals(reference[KEYS]):
                raise AssertionError(f"reference key mismatch: {prefix_tag}/{seed}")
            for fraction in TREE_FRACTIONS:
                seed_single[seed][fraction] = np.concatenate(seed_single[seed][fraction])
                seed_multi[seed][fraction] = np.concatenate(seed_multi[seed][fraction])

        lead_h = reference["lead_h"].to_numpy(int)
        persistence = reference["persistence"].to_numpy(float)
        truth = reference["target_hs"].to_numpy(float)
        incumbent = reference["incumbent_prediction"].to_numpy(float)
        incumbent_rmse = float(rmse(truth, incumbent))

        single_mean = {
            fraction: np.mean(
                np.column_stack([seed_single[seed][fraction] for seed in SEEDS]), axis=1
            )
            for fraction in TREE_FRACTIONS
        }
        multi_mean = {
            fraction: np.mean(
                np.column_stack([seed_multi[seed][fraction] for seed in SEEDS]), axis=1
            )
            for fraction in TREE_FRACTIONS
        }

        curves: dict[str, list[dict[str, Any]]] = {
            HYPOTHESES[0]: [],
            HYPOTHESES[1]: [],
        }
        for name in HYPOTHESES[:2]:
            for fraction in TREE_FRACTIONS:
                prediction = _candidate_prediction(
                    name=name,
                    lead_h=lead_h,
                    single_prediction=single_mean[fraction],
                    multi_prediction=multi_mean[fraction],
                    persistence=persistence,
                )
                curves[name].append(
                    {
                        "tree_fraction": fraction,
                        "single_trees": _tree_count(tree_counts["single"], fraction),
                        "multi_trees": _tree_count(tree_counts["multi"], fraction),
                        "rmse_m": float(rmse(truth, prediction)),
                        "delta_vs_fixed_final_incumbent_m": float(
                            rmse(truth, prediction) - incumbent_rmse
                        ),
                        "fold_rmse_m": _rmse_by_fold(reference, prediction),
                    }
                )

        splice_grid: list[dict[str, Any]] = []
        for single_fraction in TREE_FRACTIONS:
            for multi_fraction in TREE_FRACTIONS:
                prediction = _candidate_prediction(
                    name=HYPOTHESES[2],
                    lead_h=lead_h,
                    single_prediction=single_mean[single_fraction],
                    multi_prediction=multi_mean[multi_fraction],
                    persistence=persistence,
                )
                splice_grid.append(
                    {
                        "single_tree_fraction": single_fraction,
                        "multi_tree_fraction": multi_fraction,
                        "single_trees": _tree_count(tree_counts["single"], single_fraction),
                        "multi_trees": _tree_count(tree_counts["multi"], multi_fraction),
                        "rmse_m": float(rmse(truth, prediction)),
                        "delta_vs_fixed_final_incumbent_m": float(
                            rmse(truth, prediction) - incumbent_rmse
                        ),
                        "fold_rmse_m": _rmse_by_fold(reference, prediction),
                    }
                )

        summaries: dict[str, Any] = {}
        for name in HYPOTHESES[:2]:
            best = min(curves[name], key=lambda row: row["rmse_m"])
            final = next(row for row in curves[name] if row["tree_fraction"] == 1.0)
            summaries[name] = {
                "best": best,
                "final": final,
                "oracle_gain_vs_final_candidate_m": float(final["rmse_m"] - best["rmse_m"]),
                "beats_fixed_final_incumbent": bool(best["rmse_m"] < incumbent_rmse),
            }
        splice_best = min(splice_grid, key=lambda row: row["rmse_m"])
        splice_final = next(
            row
            for row in splice_grid
            if row["single_tree_fraction"] == 1.0 and row["multi_tree_fraction"] == 1.0
        )
        summaries[HYPOTHESES[2]] = {
            "best": splice_best,
            "final": splice_final,
            "oracle_gain_vs_final_candidate_m": float(
                splice_final["rmse_m"] - splice_best["rmse_m"]
            ),
            "beats_fixed_final_incumbent": bool(splice_best["rmse_m"] < incumbent_rmse),
        }

        reconstruction_errors = {
            HYPOTHESES[0]: abs(
                summaries[HYPOTHESES[0]]["final"]["rmse_m"]
                - float(rmse(truth, reference[HYPOTHESES[0]].to_numpy(float)))
            ),
            HYPOTHESES[1]: abs(
                summaries[HYPOTHESES[1]]["final"]["rmse_m"]
                - float(rmse(truth, reference[HYPOTHESES[1]].to_numpy(float)))
            ),
            HYPOTHESES[2]: abs(
                summaries[HYPOTHESES[2]]["final"]["rmse_m"]
                - float(rmse(truth, reference[HYPOTHESES[2]].to_numpy(float)))
            ),
        }
        if max(reconstruction_errors.values()) > 1e-12:
            raise AssertionError(
                f"saved-model final reconstruction mismatch for {prefix_tag}: {reconstruction_errors}"
            )
        result["prefixes"][f"{prefix:.2f}"] = {
            "rows": int(len(reference)),
            "cases": int(reference["anchor_id"].nunique()),
            "incumbent_final_rmse_m": incumbent_rmse,
            "tree_counts": tree_counts,
            "reconstruction_rmse_absolute_errors": reconstruction_errors,
            "summaries": summaries,
            "single_curve": curves[HYPOTHESES[0]],
            "multi_curve": curves[HYPOTHESES[1]],
            "splice_grid_rows": len(splice_grid),
            "splice_grid": splice_grid,
        }

    common: dict[str, Any] = {}
    for name, curve_key in (
        (HYPOTHESES[0], "single_curve"),
        (HYPOTHESES[1], "multi_curve"),
    ):
        candidates: list[dict[str, Any]] = []
        for fraction in TREE_FRACTIONS:
            rows = [
                next(
                    row
                    for row in prefix_row[curve_key]
                    if row["tree_fraction"] == fraction
                )
                for prefix_row in result["prefixes"].values()
            ]
            deltas = [row["delta_vs_fixed_final_incumbent_m"] for row in rows]
            candidates.append(
                {
                    "tree_fraction": fraction,
                    "worst_delta_vs_incumbent_m": float(max(deltas)),
                    "mean_delta_vs_incumbent_m": float(np.mean(deltas)),
                    "improved_prefixes": int(sum(value < 0.0 for value in deltas)),
                    "deltas_by_prefix_m": {
                        prefix: float(row["delta_vs_fixed_final_incumbent_m"])
                        for prefix, row in zip(result["prefixes"], rows, strict=True)
                    },
                }
            )
        common[name] = min(
            candidates,
            key=lambda row: (
                row["worst_delta_vs_incumbent_m"],
                row["mean_delta_vs_incumbent_m"],
                row["tree_fraction"],
            ),
        )

    splice_candidates: list[dict[str, Any]] = []
    for single_fraction in TREE_FRACTIONS:
        for multi_fraction in TREE_FRACTIONS:
            rows = [
                next(
                    row
                    for row in prefix_row["splice_grid"]
                    if row["single_tree_fraction"] == single_fraction
                    and row["multi_tree_fraction"] == multi_fraction
                )
                for prefix_row in result["prefixes"].values()
            ]
            deltas = [row["delta_vs_fixed_final_incumbent_m"] for row in rows]
            late_deltas = deltas[2:]
            splice_candidates.append(
                {
                    "single_tree_fraction": single_fraction,
                    "multi_tree_fraction": multi_fraction,
                    "single_trees": rows[0]["single_trees"],
                    "multi_trees": rows[0]["multi_trees"],
                    "worst_delta_vs_incumbent_m": float(max(deltas)),
                    "mean_delta_vs_incumbent_m": float(np.mean(deltas)),
                    "late_worst_delta_vs_incumbent_m": float(max(late_deltas)),
                    "late_mean_delta_vs_incumbent_m": float(np.mean(late_deltas)),
                    "improved_prefixes": int(sum(value < 0.0 for value in deltas)),
                    "improved_late_prefixes": int(sum(value < 0.0 for value in late_deltas)),
                    "deltas_by_prefix_m": {
                        prefix: float(row["delta_vs_fixed_final_incumbent_m"])
                        for prefix, row in zip(result["prefixes"], rows, strict=True)
                    },
                }
            )
    common[HYPOTHESES[2]] = min(
        splice_candidates,
        key=lambda row: (
            row["worst_delta_vs_incumbent_m"],
            row["mean_delta_vs_incumbent_m"],
            row["single_tree_fraction"],
            row["multi_tree_fraction"],
        ),
    )
    result["common_checkpoint_oracle"] = common
    result["interpretation_guard"] = {
        "same_historical_validation_truth_used_for_selection": True,
        "diagnostic_only": True,
        "original_full_prefix_gate_delta_m_at_most": -0.03,
        "checkpoint_only_can_satisfy_original_gate": bool(
            result["prefixes"]["1.00"]["summaries"][HYPOTHESES[2]]["best"][
                "delta_vs_fixed_final_incumbent_m"
            ]
            <= -0.03
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/checkpoint_retroaudit_20260827_v1/p3_catboost_checkpoint_oracle.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    result = run(root, output)
    compact = {
        prefix: {
            name: {
                "best": values["best"],
                "oracle_gain_vs_final_candidate_m": values[
                    "oracle_gain_vs_final_candidate_m"
                ],
                "beats_fixed_final_incumbent": values["beats_fixed_final_incumbent"],
            }
            for name, values in row["summaries"].items()
        }
        for prefix, row in result["prefixes"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
