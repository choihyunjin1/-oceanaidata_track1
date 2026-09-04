"""Reproduce the frozen P3 submission from saved CatBoost weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from train_predict_p3_final import _test_matrix

from p3_wave.data import LEADS, load_p3_data, resolve_p3_data_dir
from p3_wave.submission import build_submission, write_submission


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--cache-dir", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--model-dir", default="submissions/p3_frozen_catboost")
    parser.add_argument(
        "--output", default="submissions/p3_frozen_catboost/reproduced_submission.csv"
    )
    parser.add_argument("--reference", default="submissions/p3_frozen_catboost/submission.csv")
    args = parser.parse_args()
    root = resolve_p3_data_dir(args.data_dir)
    data = load_p3_data(root)
    test_index = data.test_index[["case_id", "station", "lead_h"]].copy()
    features = pd.read_parquet(Path(args.cache_dir) / "test_features.parquet")
    model_dir = Path(args.model_dir)
    columns = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))

    single = CatBoostRegressor()
    single.load_model(model_dir / "model.cbm")
    matrix = _test_matrix(features, test_index, columns)
    current_lookup = features.set_index(["case_id", "station"])
    keys = pd.MultiIndex.from_frame(test_index[["case_id", "station"]])
    current = current_lookup.loc[keys, "hs_current"].to_numpy(dtype=float)
    single_prediction = np.clip(current + single.predict(matrix), 0.0, 30.0)

    multi = CatBoostRegressor()
    multi.load_model(model_dir / "model_multi.cbm")
    multi_matrix = features[["station", *columns]].copy()
    multi_matrix["station"] = multi_matrix["station"].astype(str)
    delta = np.asarray(multi.predict(multi_matrix), dtype=float)
    multi_long = pd.DataFrame(
        {
            "case_id": np.repeat(features["case_id"].to_numpy(), len(LEADS)),
            "station": np.repeat(features["station"].to_numpy(), len(LEADS)),
            "lead_h": np.tile(np.asarray(LEADS), len(features)),
            "prediction": np.clip(
                np.repeat(features["hs_current"].to_numpy(dtype=float), len(LEADS))
                + delta.reshape(-1),
                0.0,
                30.0,
            ),
        }
    )
    multi_prediction = test_index.merge(
        multi_long,
        on=["case_id", "station", "lead_h"],
        how="left",
        validate="one_to_one",
    )["prediction"].to_numpy(dtype=float)
    prediction = 0.5 * (single_prediction + multi_prediction)
    output = write_submission(build_submission(test_index, prediction), test_index, args.output)
    reference = pd.read_csv(args.reference)
    difference = float(np.max(np.abs(reference["hs_pred"].to_numpy(dtype=float) - prediction)))
    if difference > 1e-12:
        raise ValueError(f"saved-model reproduction differs by {difference}")
    print(
        json.dumps(
            {
                "status": "passed_saved_weight_reproduction",
                "rows": len(test_index),
                "max_abs_difference": difference,
                "output": output.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
