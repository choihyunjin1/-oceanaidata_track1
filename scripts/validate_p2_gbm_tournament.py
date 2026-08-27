"""Independently recount the P2 GBM tournament and submission contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.gbm_tournament import GBM_ARM_SPECS
from p2_restore.submission import validate_submission


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(truth)) ** 2)))


def _assert_close(actual: float, expected: float, label: str, *, tolerance: float = 1e-12) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def _aligned(frame: pd.DataFrame) -> pd.DataFrame:
    copy = frame.copy()
    copy["time"] = pd.to_datetime(copy["time"], utc=True)
    return copy.sort_values(["time", "layer", "block"]).reset_index(drop=True)


def validate_tournament(
    result_path: Path,
    deep_oof_path: Path,
    output_root: Path,
    test_index_path: Path,
) -> dict[str, object]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    deep = _aligned(pd.read_parquet(deep_oof_path))
    keys = ["time", "layer", "block"]
    required_deep = {*keys, "truth", "prediction", "lobo_prediction"}
    if missing := required_deep.difference(deep.columns):
        raise ValueError(f"deep OOF is missing columns: {sorted(missing)}")
    if deep[keys].duplicated().any() or len(deep) != 69_850:
        raise ValueError("deep OOF grain or row count is invalid")
    truth = deep["truth"].to_numpy(float)
    _assert_close(
        _rmse(truth, deep["prediction"].to_numpy(float)),
        float(result["oof_contract"]["deep_stack_rmse"]),
        "deep-stack RMSE",
    )
    _assert_close(
        _rmse(truth, deep["lobo_prediction"].to_numpy(float)),
        float(result["oof_contract"]["deep_stack_lobo_rmse"]),
        "deep-stack LOBO RMSE",
    )
    test_index = pd.read_csv(test_index_path, dtype={"station": "string", "time": "string"})
    arms: dict[str, object] = {}
    for spec in GBM_ARM_SPECS:
        root = output_root / spec.name
        oof_path = root / "oof.parquet"
        paired_path = root / "paired_oof.parquet"
        oof = _aligned(pd.read_parquet(oof_path))
        paired = _aligned(pd.read_parquet(paired_path))
        if not deep[[*keys, "truth"]].equals(oof[[*keys, "truth"]]):
            raise ValueError(f"OOF grain or truth differs: {spec.name}")
        if not deep[[*keys, "truth"]].equals(paired[[*keys, "truth"]]):
            raise ValueError(f"paired OOF grain or truth differs: {spec.name}")
        saved = result["arms"][spec.name]
        standalone = _rmse(truth, oof["prediction"].to_numpy(float))
        fitted = _rmse(truth, paired["fitted_pair_prediction"].to_numpy(float))
        lobo = _rmse(truth, paired["lobo_pair_prediction"].to_numpy(float))
        _assert_close(standalone, float(saved["rmse"]), f"{spec.name} standalone")
        _assert_close(
            fitted,
            float(saved["pair_with_deep"]["fitted_blend_rmse"]),
            f"{spec.name} fitted pair",
        )
        _assert_close(
            lobo,
            float(saved["pair_with_deep"]["lobo_blend_rmse"]),
            f"{spec.name} LOBO pair",
        )
        reconstructed = np.full(len(paired), np.nan)
        for layer in (2, 3, 4):
            selected = paired["layer"].eq(layer).to_numpy()
            weight = float(saved["pair_with_deep"]["fitted_weights_by_layer"][str(layer)])
            reconstructed[selected] = (1.0 - weight) * paired.loc[
                selected, "deep_prediction"
            ].to_numpy(float) + weight * paired.loc[selected, "gbm_prediction"].to_numpy(float)
        pair_error = float(
            np.max(np.abs(reconstructed - paired["fitted_pair_prediction"].to_numpy(float)))
        )
        if pair_error > 1e-12:
            raise ValueError(f"{spec.name} fitted layer blend did not reproduce")
        full = result["arm_full_outputs"][spec.name]
        submission_path = Path(full["submission"]["path"])
        if _sha256(submission_path) != full["submission"]["sha256"]:
            raise ValueError(f"{spec.name} submission hash mismatch")
        submission_validation = validate_submission(submission_path, test_index)
        arms[spec.name] = {
            "rows": len(oof),
            "standalone_rmse": standalone,
            "fitted_pair_rmse": fitted,
            "lobo_pair_rmse": lobo,
            "pair_reproduction_max_abs_error": pair_error,
            "oof_sha256": _sha256(oof_path),
            "paired_oof_sha256": _sha256(paired_path),
            "submission_sha256": _sha256(submission_path),
            "submission_rows": submission_validation["rows"],
        }
    hybrid = result["hybrid_research_candidate"]
    hybrid_path = Path(hybrid["path"])
    if _sha256(hybrid_path) != hybrid["sha256"]:
        raise ValueError("hybrid research submission hash mismatch")
    hybrid_validation = validate_submission(hybrid_path, test_index)
    ranking = sorted(
        (
            {
                "arm": name,
                "standalone_rmse": values["standalone_rmse"],
                "fitted_pair_rmse": values["fitted_pair_rmse"],
                "lobo_pair_rmse": values["lobo_pair_rmse"],
            }
            for name, values in arms.items()
        ),
        key=lambda row: (row["lobo_pair_rmse"], row["fitted_pair_rmse"], row["standalone_rmse"]),
    )
    if ranking[0]["arm"] != result["selected_for_parameter_search"]:
        raise ValueError("independent GBM ranking differs from saved selection")
    return {
        "validated_at": datetime.now().astimezone().isoformat(),
        "status": "passed",
        "raw_rows_written": 0,
        "result_sha256": _sha256(result_path),
        "deep_oof_sha256": _sha256(deep_oof_path),
        "oof_rows": len(deep),
        "unique_oof_keys": int(len(deep.drop_duplicates(keys))),
        "arms": arms,
        "ranking": ranking,
        "selected_for_parameter_search": ranking[0]["arm"],
        "hybrid_submission": {
            "rows": hybrid_validation["rows"],
            "sha256": _sha256(hybrid_path),
            "minimum": hybrid_validation["minimum"],
            "maximum": hybrid_validation["maximum"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", type=Path, default=Path("artifacts/p2_gbm_family_tournament_v1/result.json")
    )
    parser.add_argument(
        "--deep-oof",
        type=Path,
        default=Path("artifacts/p2_deep_finalists_v1/stacked_oof.parquet"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/p2_gbm_family_tournament_v1")
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/p2_gbm_family_tournament_v1/independent_validation.json"),
    )
    args = parser.parse_args()
    report = validate_tournament(
        args.result,
        args.deep_oof,
        args.output_root,
        args.data_dir / "test_index.csv",
    )
    _write = args.output.with_suffix(args.output.suffix + ".tmp")
    _write.parent.mkdir(parents=True, exist_ok=True)
    _write.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write.replace(args.output)
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar.write_text(_sha256(args.output) + "\n", encoding="ascii")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
