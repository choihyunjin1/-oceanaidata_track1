"""Leakage-reduced LOFO audit for the saved P3 CatBoost checkpoint grid.

The input oracle artifact contains historical train-only validation RMSE values for
every single/multi tree-prefix pair.  For each held fold this script selects one pair
using only the other folds, evaluates it once on the held fold, and then pools the
held-fold squared errors.  It never opens official P3 test, sample, or submission
files and it does not train or mutate a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


KEYS = ("fold", "anchor_id", "station", "lead_h")
FINAL_PAIR = (1.0, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pooled_rmse(rows: list[tuple[int, float]]) -> float:
    count = sum(n for n, _ in rows)
    if count <= 0:
        raise ValueError("cannot pool an empty RMSE surface")
    return math.sqrt(sum(n * value * value for n, value in rows) / count)


def _select_pair(
    grid: list[dict[str, Any]],
    *,
    training_folds: tuple[str, ...],
    fold_counts: dict[str, int],
) -> dict[str, Any]:
    candidates: list[tuple[float, float, float, dict[str, Any]]] = []
    for row in grid:
        pooled = _pooled_rmse(
            [(fold_counts[fold], float(row["fold_rmse_m"][fold])) for fold in training_folds]
        )
        candidates.append(
            (
                pooled,
                float(row["single_tree_fraction"]),
                float(row["multi_tree_fraction"]),
                row,
            )
        )
    # Exact minimum with deterministic earlier-tree tie breaking.
    pooled, _, _, selected = min(candidates, key=lambda item: item[:3])
    return {"selection_rmse_m": pooled, "grid_row": selected}


def run(*, root: Path, oracle_path: Path, output_path: Path) -> dict[str, Any]:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    if oracle.get("schema_version") != "p3_catboost_checkpoint_oracle.v1":
        raise ValueError("unexpected checkpoint oracle schema")
    if oracle.get("official_test_sample_submission_reads") != 0:
        raise PermissionError("oracle input does not declare a train-only scope")

    oof_path = root / "artifacts/p3_meaningful_learning_curve_20260823_v1/oof/learning_curve_oof.parquet"
    oof = pd.read_parquet(oof_path, columns=list(KEYS) + ["target_hs", "incumbent_prediction", "prefix_fraction"])
    if oof.duplicated(["prefix_fraction", *KEYS]).any():
        raise ValueError("sealed OOF keys are duplicated")

    output: dict[str, Any] = {
        "schema_version": "p3_catboost_checkpoint_lofo.v1",
        "scope": "historical_train_only_validation",
        "selection": "held-fold excluded pooled SSE; exact minimum; earlier-tree tie break",
        "promotion_evidence": False,
        "official_test_sample_submission_reads": 0,
        "source_sha256": {
            "oracle": _sha256(oracle_path),
            "sealed_oof": _sha256(oof_path),
        },
        "prefixes": {},
    }

    for prefix_text, prefix_row in oracle["prefixes"].items():
        prefix = float(prefix_text)
        reference = oof.loc[oof["prefix_fraction"].eq(prefix)].copy()
        folds = tuple(sorted(reference["fold"].astype(str).unique()))
        if len(folds) != 3:
            raise ValueError(f"expected three folds at prefix {prefix_text}")
        fold_counts = {
            fold: int(reference.loc[reference["fold"].astype(str).eq(fold)].shape[0])
            for fold in folds
        }
        grid = prefix_row["splice_grid"]
        if len(grid) != 400:
            raise ValueError(f"expected a 20x20 splice grid at prefix {prefix_text}")
        final = next(
            row
            for row in grid
            if (float(row["single_tree_fraction"]), float(row["multi_tree_fraction"]))
            == FINAL_PAIR
        )

        held_rows: list[dict[str, Any]] = []
        selected_pool: list[tuple[int, float]] = []
        final_pool: list[tuple[int, float]] = []
        incumbent_pool: list[tuple[int, float]] = []
        for held_fold in folds:
            training_folds = tuple(fold for fold in folds if fold != held_fold)
            selected = _select_pair(
                grid,
                training_folds=training_folds,
                fold_counts=fold_counts,
            )
            row = selected["grid_row"]
            selected_rmse = float(row["fold_rmse_m"][held_fold])
            final_rmse = float(final["fold_rmse_m"][held_fold])
            held_reference = reference.loc[reference["fold"].astype(str).eq(held_fold)]
            incumbent_rmse = math.sqrt(
                float(
                    ((held_reference["target_hs"] - held_reference["incumbent_prediction"]) ** 2).mean()
                )
            )
            n = fold_counts[held_fold]
            selected_pool.append((n, selected_rmse))
            final_pool.append((n, final_rmse))
            incumbent_pool.append((n, incumbent_rmse))
            held_rows.append(
                {
                    "held_fold": held_fold,
                    "selection_folds": list(training_folds),
                    "selection_rmse_m": float(selected["selection_rmse_m"]),
                    "selected_single_tree_fraction": float(row["single_tree_fraction"]),
                    "selected_multi_tree_fraction": float(row["multi_tree_fraction"]),
                    "selected_single_trees": int(row["single_trees"]),
                    "selected_multi_trees": int(row["multi_trees"]),
                    "held_rows": n,
                    "held_selected_rmse_m": selected_rmse,
                    "held_final_rmse_m": final_rmse,
                    "held_incumbent_rmse_m": incumbent_rmse,
                    "held_delta_selected_minus_final_m": selected_rmse - final_rmse,
                    "held_delta_selected_minus_incumbent_m": selected_rmse - incumbent_rmse,
                }
            )

        selected_rmse = _pooled_rmse(selected_pool)
        final_rmse = _pooled_rmse(final_pool)
        incumbent_rmse = _pooled_rmse(incumbent_pool)
        output["prefixes"][prefix_text] = {
            "rows": int(sum(fold_counts.values())),
            "fold_counts": fold_counts,
            "held_fold_results": held_rows,
            "lofo_selected_rmse_m": selected_rmse,
            "final_splice_rmse_m": final_rmse,
            "incumbent_rmse_m": incumbent_rmse,
            "delta_selected_minus_final_m": selected_rmse - final_rmse,
            "delta_selected_minus_incumbent_m": selected_rmse - incumbent_rmse,
            "improved_held_folds_vs_final": int(
                sum(row["held_delta_selected_minus_final_m"] < 0.0 for row in held_rows)
            ),
            "improved_held_folds_vs_incumbent": int(
                sum(row["held_delta_selected_minus_incumbent_m"] < 0.0 for row in held_rows)
            ),
        }

    deltas = [
        row["delta_selected_minus_incumbent_m"] for row in output["prefixes"].values()
    ]
    output["summary"] = {
        "improved_prefixes_vs_incumbent": int(sum(value < 0.0 for value in deltas)),
        "all_prefixes_improve_vs_incumbent": bool(all(value < 0.0 for value in deltas)),
        "mean_delta_selected_minus_incumbent_m": float(sum(deltas) / len(deltas)),
        "interpretation": "LOFO checkpoint transport diagnostic; not a fresh official or promotion result",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--oracle",
        type=Path,
        default=Path("artifacts/checkpoint_retroaudit_20260827_v1/p3_catboost_checkpoint_oracle.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/p3_catboost_checkpoint_lofo_20260827_v1/result.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    oracle = args.oracle if args.oracle.is_absolute() else root / args.oracle
    output = args.output if args.output.is_absolute() else root / args.output
    result = run(root=root, oracle_path=oracle, output_path=output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
