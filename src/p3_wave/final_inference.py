"""Pure saved-weight inference for the frozen P3 long-lead candidate."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import LEADS, load_p3_data, resolve_p3_data_dir
from .features import build_test_features
from .loss_router import (
    OBSERVED_FEATURES,
    build_inference_router_features,
    expand_case_router_features,
    route_row_predictions,
)
from .persistence_shrink import apply_long_lead_persistence_shrink
from .submission import build_submission, validate_submission, write_submission

KEYS = ["case_id", "station", "lead_h"]
ACTIVE_ROUTER_LEADS = (12, 18, 24)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["station"] = result["station"].astype(str)
    if "lead_h" in result:
        result["lead_h"] = result["lead_h"].astype(str)
    return result


def csv_float_roundtrip(values: np.ndarray) -> np.ndarray:
    """Mirror the historical component-CSV boundary without reading a saved CSV."""

    buffer = io.StringIO()
    pd.DataFrame({"value": np.asarray(values, dtype=np.float64)}).to_csv(
        buffer, index=False, lineterminator="\n"
    )
    buffer.seek(0)
    return pd.read_csv(buffer)["value"].to_numpy(dtype=np.float64)


def _single_matrix(features: pd.DataFrame, index: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    source = features.set_index(["case_id", "station"])
    keys = pd.MultiIndex.from_frame(index[["case_id", "station"]])
    selected = source.loc[keys, columns].reset_index(drop=True)
    selected.insert(0, "lead_h", index["lead_h"].to_numpy())
    selected.insert(0, "station", index["station"].astype(str).to_numpy())
    selected.insert(2, "current_hs_for_residual", source.loc[keys, "hs_current"].to_numpy(float))
    return _cat_frame(selected)


def predict_catboost_components(
    features: pd.DataFrame,
    test_index: pd.DataFrame,
    *,
    model_path: str | Path,
    multi_model_path: str | Path,
    feature_columns_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from catboost import CatBoostRegressor

    columns = json.loads(Path(feature_columns_path).read_text(encoding="utf-8"))
    source = features.set_index(["case_id", "station"])
    keys = pd.MultiIndex.from_frame(test_index[["case_id", "station"]])
    current = source.loc[keys, "hs_current"].to_numpy(float)

    single = CatBoostRegressor()
    single.load_model(model_path)
    single_prediction = np.clip(
        current + single.predict(_single_matrix(features, test_index, columns)), 0.0, 30.0
    )

    multi = CatBoostRegressor()
    multi.load_model(multi_model_path)
    multi_matrix = _cat_frame(features[["station", *columns]])
    delta = np.asarray(multi.predict(multi_matrix), dtype=np.float64)
    long = pd.DataFrame(
        {
            "case_id": np.repeat(features["case_id"].to_numpy(), len(LEADS)),
            "station": np.repeat(features["station"].to_numpy(), len(LEADS)),
            "lead_h": np.tile(np.asarray(LEADS), len(features)),
            "prediction": np.clip(
                np.repeat(features["hs_current"].to_numpy(float), len(LEADS)) + delta.reshape(-1),
                0.0,
                30.0,
            ),
        }
    )
    multi_prediction = test_index.merge(long, on=KEYS, how="left", validate="one_to_one")[
        "prediction"
    ].to_numpy(float)
    if not np.isfinite(single_prediction).all() or not np.isfinite(multi_prediction).all():
        raise ValueError("P3 CatBoost components contain non-finite values")
    # The frozen router was fitted/deployed after the two components had been
    # written and reread as CSV.  Recreate that deterministic numeric boundary
    # in memory so saved-weight inference is byte-identical without depending
    # on either intermediate submission file.
    return (
        csv_float_roundtrip(single_prediction),
        csv_float_roundtrip(multi_prediction),
        current,
    )


def apply_saved_router(
    features: pd.DataFrame,
    test_index: pd.DataFrame,
    single: np.ndarray,
    multi: np.ndarray,
    current: np.ndarray,
    router: object,
) -> np.ndarray:
    case_order = test_index[["case_id", "station"]].drop_duplicates().reset_index(drop=True)
    joined = test_index[KEYS].copy()
    joined["single"] = single
    joined["multi"] = multi
    joined["persistence"] = current
    components: list[np.ndarray] = []
    case_current: list[float] = []
    for row in case_order.itertuples(index=False):
        block = joined.loc[
            joined["case_id"].eq(row.case_id) & joined["station"].eq(row.station)
        ].sort_values("lead_h")
        if tuple(block["lead_h"].astype(int)) != tuple(LEADS):
            raise ValueError("P3 test case is missing an official lead")
        components.append(block[["single", "multi", "persistence"]].to_numpy(float))
        case_current.append(float(block["persistence"].iloc[0]))
    component_array = np.stack(components)

    observed = case_order.merge(
        features[["case_id", "station", *OBSERVED_FEATURES]],
        on=["case_id", "station"],
        how="left",
        validate="one_to_one",
    )
    case_x = build_inference_router_features(
        observed.loc[:, OBSERVED_FEATURES],
        case_order["station"].to_numpy(str),
        np.asarray(case_current),
        component_array,
    )
    meta = case_order.copy()
    meta["fold"] = "hidden_test"
    meta["anchor_id"] = np.arange(len(meta), dtype=np.int64)
    meta["anchor_time"] = pd.NaT
    row_x, row_meta, row_components = expand_case_router_features(
        case_x,
        meta[["fold", "anchor_id", "station", "anchor_time"]],
        component_array,
    )
    weights = router.predict_weights(row_x)
    inactive = ~row_meta["lead_h"].isin(ACTIVE_ROUTER_LEADS).to_numpy()
    weights[inactive] = np.array([0.5, 0.5, 0.0])
    routed = pd.DataFrame(
        {
            "case_id": np.repeat(case_order["case_id"].to_numpy(str), len(LEADS)),
            "station": np.repeat(case_order["station"].to_numpy(str), len(LEADS)),
            "lead_h": np.tile(np.asarray(LEADS), len(case_order)),
            "prediction": route_row_predictions(row_components, weights),
        }
    )
    return test_index.merge(routed, on=KEYS, how="left", validate="one_to_one")[
        "prediction"
    ].to_numpy(float)


def reproduce_final_submission(
    *,
    data_dir: str | Path,
    model_path: str | Path,
    multi_model_path: str | Path,
    feature_columns_path: str | Path,
    router_path: str | Path,
    output_path: str | Path,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    root = resolve_p3_data_dir(data_dir)
    data = load_p3_data(root)
    features = build_test_features(data).features
    test_index = data.test_index[KEYS].copy()
    single, multi, current = predict_catboost_components(
        features,
        test_index,
        model_path=model_path,
        multi_model_path=multi_model_path,
        feature_columns_path=feature_columns_path,
    )
    router = joblib.load(router_path)
    routed = apply_saved_router(features, test_index, single, multi, current, router)
    persistence = pd.read_csv(root / "baseline_persistence.csv")
    if not persistence[KEYS].equals(test_index):
        raise ValueError("P3 persistence keys/order differ from test_index")
    final = apply_long_lead_persistence_shrink(
        csv_float_roundtrip(routed),
        persistence["hs_pred"].to_numpy(float),
        test_index["lead_h"].to_numpy(int),
    )
    target = write_submission(build_submission(test_index, final), test_index, output_path)
    reread = pd.read_csv(target)
    validate_submission(reread, test_index)
    actual = sha256_file(target)
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(f"P3 reproduced SHA differs: {actual} != {expected_sha256}")
    return {
        "rows": len(reread),
        "cases": int(reread["case_id"].nunique()),
        "minimum": float(reread["hs_pred"].min()),
        "maximum": float(reread["hs_pred"].max()),
        "sha256": actual,
    }
