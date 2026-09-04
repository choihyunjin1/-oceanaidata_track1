"""Model-driven inference for the highest clean-lineage P3 candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

HERE = Path(__file__).resolve().parent
KEYS = ["case_id", "station", "lead_h"]
LEADS = (3, 6, 9, 12, 18, 24)


def _activate_source(package: Path) -> None:
    source = package / "07_source" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _cat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["station"] = result["station"].astype(str)
    if "lead_h" in result:
        result["lead_h"] = result["lead_h"].astype(str)
    return result


def _axis_prediction(
    features: pd.DataFrame, test_index: pd.DataFrame, model_dir: Path
) -> np.ndarray:
    from p3_wave.corrected_fixed_long_shrink import FixedLongLeadShrinkCalibrator
    from p3_wave.loss_router import (
        OBSERVED_FEATURES,
        build_inference_router_features,
        expand_case_router_features,
        route_row_predictions,
    )

    payload = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
    columns = payload["columns"] if isinstance(payload, dict) else payload
    case_order = test_index[["case_id", "station"]].drop_duplicates().reset_index(drop=True)
    cases = case_order.merge(
        features, on=["case_id", "station"], how="left", validate="one_to_one"
    )
    source = cases.set_index(["case_id", "station"])
    row_keys = pd.MultiIndex.from_frame(test_index[["case_id", "station"]])
    single_x = source.loc[row_keys, columns].reset_index(drop=True)
    single_x.insert(0, "lead_h", test_index["lead_h"].to_numpy())
    single_x.insert(0, "station", test_index["station"].astype(str).to_numpy())
    current_rows = source.loc[row_keys, "hs_current"].to_numpy(float)
    single_x.insert(2, "current_hs_for_residual", current_rows)
    single = CatBoostRegressor()
    single.load_model(model_dir / "single.cbm")
    single_prediction = np.clip(
        current_rows + single.predict(_cat_frame(single_x)), 0.0, 30.0
    )
    multi_x = cases[["station", *columns]].copy()
    multi_x["station"] = multi_x["station"].astype(str)
    multi = CatBoostRegressor()
    multi.load_model(model_dir / "multi.cbm")
    multi_delta = np.asarray(multi.predict(multi_x), dtype=np.float64)
    current_case = cases["hs_current"].to_numpy(float)
    multi_prediction = np.clip(current_case[:, None] + multi_delta, 0.0, 30.0)
    components = np.stack(
        [
            single_prediction.reshape(len(cases), len(LEADS)),
            multi_prediction,
            np.repeat(current_case[:, None], len(LEADS), axis=1),
        ],
        axis=2,
    )
    case_x = build_inference_router_features(
        cases.loc[:, OBSERVED_FEATURES],
        cases["station"].to_numpy(str),
        current_case,
        components,
    )
    meta = pd.DataFrame(
        {
            "fold": "anonymous_test",
            "anchor_id": np.arange(len(cases), dtype=np.int64),
            "station": cases["station"].astype(str),
            "anchor_time": pd.NaT,
        }
    )
    row_x, row_meta, row_components = expand_case_router_features(case_x, meta, components)
    router = joblib.load(model_dir / "router.joblib")
    weights = router.predict_weights(row_x)
    inactive = ~row_meta["lead_h"].isin([12, 18, 24]).to_numpy()
    weights[inactive] = np.array([0.5, 0.5, 0.0])
    routed = route_row_predictions(row_components, weights)
    calibrator: FixedLongLeadShrinkCalibrator = joblib.load(model_dir / "calibrator.joblib")
    return calibrator.predict(routed, current_rows, test_index["lead_h"].to_numpy(int))


def predict(data_dir: str | Path, package_dir: str | Path, output_path: str | Path) -> dict:
    package = Path(package_dir).resolve()
    data_root = Path(data_dir).resolve()
    _activate_source(package)
    from p3_wave.data import load_p3_data
    from p3_wave.features import build_test_features
    from p3_wave.final_inference import (
        apply_saved_router,
        csv_float_roundtrip,
        predict_catboost_components,
    )
    from p3_wave.persistence_shrink import apply_long_lead_persistence_shrink

    contract = json.loads((package / "contract.json").read_text(encoding="utf-8"))
    data = load_p3_data(data_root)
    features = build_test_features(data).features
    test_index = data.test_index[KEYS].copy()
    persistence_frame = pd.read_csv(data_root / "baseline_persistence.csv")
    if not persistence_frame[KEYS].equals(test_index):
        raise RuntimeError("P3 persistence/test key order differs")
    persistence = persistence_frame["hs_pred"].to_numpy(float)

    original_dir = package / "03_model" / "weights" / "original"
    single, multi, current = predict_catboost_components(
        features,
        test_index,
        model_path=original_dir / "model.cbm",
        multi_model_path=original_dir / "model_multi.cbm",
        feature_columns_path=original_dir / "feature_columns.json",
    )
    original_routed = apply_saved_router(
        features,
        test_index,
        single,
        multi,
        current,
        joblib.load(original_dir / "router.joblib"),
    )
    original = csv_float_roundtrip(
        apply_long_lead_persistence_shrink(
            csv_float_roundtrip(original_routed),
            persistence,
            test_index["lead_h"].to_numpy(int),
        )
    )
    axis = csv_float_roundtrip(
        _axis_prediction(features, test_index, package / "03_model" / "weights" / "axis")
    )
    values = original.copy()
    active = test_index["lead_h"].isin(contract["active_leads"]).to_numpy()
    values[active] += float(contract["alpha"]) * (axis[active] - original[active])
    if not np.array_equal(values[~active], original[~active]):
        raise RuntimeError("P3 short-lead no-op drift")
    if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 30.0:
        raise RuntimeError("P3 final physical/domain guard failed")
    candidate = test_index.copy()
    candidate["hs_pred"] = values
    target = Path(output_path)
    if not target.is_absolute():
        target = package / target
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(target, index=False, encoding="utf-8", lineterminator="\n")
    from common import sha256_file

    actual = sha256_file(target)
    if actual != contract["candidate_sha256"]:
        raise RuntimeError(f"P3 model-driven output SHA drift: {actual}")
    return {
        "status": "READY_MODEL_INFERENCE_EXACT_NOT_UPLOADED",
        "candidate_id": contract["candidate_id"],
        "rows": len(candidate),
        "columns": list(candidate.columns),
        "changed_rows": int(active.sum()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "key_order_exact": True,
        "sha256": actual,
        "candidate_hash_exact": True,
        "package_atomic": True,
        "checkpoint_files_loaded": 9,
        "prediction_source": "two_saved_catboost_router_chains_then_frozen_affine_combination",
        "lineage": "organizer_distributed_data_only_scratch_models",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=HERE.parents[1])
    parser.add_argument("--output", type=Path, default=Path("05_answer/P3_submission.csv"))
    args = parser.parse_args()
    print(json.dumps(predict(args.data_dir, args.package_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
