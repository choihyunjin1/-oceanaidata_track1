"""Independently validate the P2 top-three tuning artifacts and submissions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.gbm_tournament import align_with_deep_stack, evaluate_deep_pair
from p2_restore.gbm_tuning import TUNING_FAMILIES
from p2_restore.submission import validate_submission


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=tolerance):
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(truth)) ** 2)))


def validate_family(
    root: Path, family: str, deep: pd.DataFrame, test_index: pd.DataFrame
) -> dict[str, object]:
    family_root = root / family
    result_path = family_root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("family") != family or result.get("uploaded") is not False:
        raise ValueError(f"invalid family result contract: {family}")
    for name, value in result["artifacts"].items():
        path = Path(value["path"])
        if not path.is_file() or _sha256(path) != value["sha256"]:
            raise ValueError(f"family artifact hash mismatch: {family}/{name}")
    oof = pd.read_parquet(result["artifacts"]["oof"]["path"])
    if len(oof) != 69_850 or oof.duplicated(["time", "layer", "block"]).any():
        raise ValueError(f"family OOF grain changed: {family}")
    aligned = align_with_deep_stack(deep, oof)
    truth = aligned["truth"].to_numpy(float)
    _close(
        _rmse(truth, aligned["gbm_prediction"].to_numpy(float)),
        result["tuning"]["outer_rmse"],
        f"{family} outer RMSE",
    )
    for layer in (2, 3, 4):
        selected = aligned["layer"].to_numpy(int) == layer
        _close(
            _rmse(truth[selected], aligned.loc[selected, "gbm_prediction"].to_numpy(float)),
            result["tuning"]["outer_by_layer_rmse"][str(layer)],
            f"{family} layer {layer} RMSE",
        )
    pair = evaluate_deep_pair(aligned)
    for key in (
        "deep_rmse",
        "deep_lobo_rmse",
        "fitted_blend_rmse",
        "fitted_delta_vs_deep",
        "lobo_blend_rmse",
        "lobo_delta_vs_deep_lobo",
    ):
        _close(pair[key], result["deep_pair"][key], f"{family} {key}")
    for layer in (2, 3, 4):
        _close(
            pair["fitted_weights_by_layer"][str(layer)],
            result["deep_pair"]["fitted_weights_by_layer"][str(layer)],
            f"{family} fitted layer {layer} weight",
        )
    submission_validation = validate_submission(
        result["artifacts"]["submission"]["path"], test_index
    )
    return {
        "family": family,
        "rows": len(oof),
        "unique_keys": True,
        "outer_rmse": result["tuning"]["outer_rmse"],
        "lobo_pair_rmse": result["deep_pair"]["lobo_blend_rmse"],
        "submission": submission_validation,
        "result_sha256": _sha256(result_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--root", type=Path, default=Path("artifacts/p2_top3_parallel_tuning_v1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/p2_top3_parallel_tuning_v1/independent_validation.json"),
    )
    args = parser.parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    data = load_p2_data(data_dir)
    deep_path = Path("artifacts/p2_deep_finalists_v1/stacked_oof.parquet")
    deep = pd.read_parquet(deep_path)
    families = [
        validate_family(args.root, family, deep, data.test_index) for family in TUNING_FAMILIES
    ]
    master_path = args.root / "result.json"
    master = json.loads(master_path.read_text(encoding="utf-8"))
    ranked = sorted(
        families, key=lambda value: (value["lobo_pair_rmse"], value["outer_rmse"], value["family"])
    )
    if [value["family"] for value in ranked] != [value["family"] for value in master["ranking"]]:
        raise ValueError("master tuning ranking is not reproducible")
    research = master["research_pair_submission"]
    research_validation = validate_submission(research["path"], data.test_index)
    if _sha256(Path(research["path"])) != research["sha256"]:
        raise ValueError("research pair submission hash mismatch")
    winner = json.loads(
        (args.root / ranked[0]["family"] / "result.json").read_text(encoding="utf-8")
    )
    standalone = pd.read_csv(winner["artifacts"]["submission"]["path"])
    deep_submission = pd.read_csv("submissions/p2/P2_DEEP_STACK_V1.csv")
    research_frame = pd.read_csv(research["path"])
    reconstructed = np.empty(len(research_frame), dtype=float)
    for layer in (2, 3, 4):
        selected = research_frame["layer"].to_numpy(int) == layer
        weight = float(research["weights_by_layer"][str(layer)])
        reconstructed[selected] = (1.0 - weight) * deep_submission.loc[selected, "temp"].to_numpy(
            float
        ) + weight * standalone.loc[selected, "temp"].to_numpy(float)
    max_abs = float(np.max(np.abs(reconstructed - research_frame["temp"].to_numpy(float))))
    if max_abs > 1e-12:
        raise ValueError(f"research pair submission is not reproducible: {max_abs}")
    output = {
        "status": "passed",
        "master_result_sha256": _sha256(master_path),
        "deep_oof_sha256": _sha256(deep_path),
        "families": families,
        "ranking": [value["family"] for value in ranked],
        "research_pair": {
            "family": research["family"],
            "rows": research_validation["rows"],
            "sha256": research["sha256"],
            "reconstruction_max_abs_error": max_abs,
        },
        "raw_rows_written": 0,
        "uploaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar.write_text(f"{_sha256(args.output)}  {args.output.name}\n", encoding="ascii")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
