"""Rebuild P3 features and scratch-train the two CatBoost base-model branches.

The learned loss-router sources are preserved beside this script because the
official highest clean candidate uses two historical, independently trained
router chains.  This command retrains their single/multi base estimators into a
new output directory and never overwrites the certified weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

HERE = Path(__file__).resolve().parent
LEADS = (3, 6, 9, 12, 18, 24)


def train(data_dir: str | Path, package_dir: str | Path, output_dir: str | Path) -> dict:
    started = time.perf_counter()
    package = Path(package_dir).resolve()
    source = package / "07_source" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from p3_wave.data import load_p3_data
    from p3_wave.features import build_test_features, build_training_features
    from p3_wave.models import compact_feature_columns, threshold_case_weights
    from p3_wave.validation import expand_leads

    target = Path(output_dir)
    if not target.is_absolute():
        target = package / target
    if target.exists():
        raise FileExistsError(f"P3 retrain output already exists: {target}")
    target.mkdir(parents=True)
    data = load_p3_data(Path(data_dir).resolve())
    train_set = build_training_features(data, dense_spacing_minutes=20)
    test_set = build_test_features(data)
    feature_columns = compact_feature_columns(list(train_set.feature_columns))
    anchors = train_set.anchors
    features = train_set.features
    train_ids = anchors["anchor_id"].to_numpy(np.int64)
    x_train, y_train, meta = expand_leads(features, anchors, train_ids, feature_columns)
    x_train = x_train.copy()
    x_train["station"] = x_train["station"].astype(str)
    x_train["lead_h"] = x_train["lead_h"].astype(str)
    single = CatBoostRegressor(
        loss_function="RMSE",
        iterations=700,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=8.0,
        random_strength=0.2,
        random_seed=20260817,
        thread_count=8,
        verbose=False,
        allow_writing_files=False,
    )
    single.fit(
        x_train,
        y_train,
        sample_weight=threshold_case_weights(meta["current_hs"].to_numpy()),
        cat_features=[0, 1],
        verbose=False,
    )
    single.save_model(target / "single.cbm")
    lookup = features.set_index("anchor_id")
    multi_x = lookup.loc[train_ids, ["station", *feature_columns]].reset_index(drop=True)
    multi_x["station"] = multi_x["station"].astype(str)
    anchor_lookup = anchors.set_index("anchor_id")
    multi_y = np.column_stack(
        [
            anchor_lookup.loc[train_ids, f"target_{lead}"].to_numpy(float)
            - anchor_lookup.loc[train_ids, "current_hs"].to_numpy(float)
            for lead in LEADS
        ]
    )
    multi = CatBoostRegressor(
        loss_function="MultiRMSE",
        iterations=1200,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=10.0,
        random_strength=0.15,
        random_seed=20260817,
        task_type="GPU",
        devices="0",
        boosting_type="Plain",
        verbose=False,
        allow_writing_files=False,
    )
    multi.fit(
        multi_x,
        multi_y,
        sample_weight=threshold_case_weights(
            anchor_lookup.loc[train_ids, "current_hs"].to_numpy(float)
        ),
        cat_features=[0],
        verbose=False,
    )
    multi.save_model(target / "multi.cbm")
    (target / "feature_columns.json").write_text(
        json.dumps(feature_columns, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    test_set.features.to_parquet(target / "test_features.parquet", index=False)
    result = {
        "status": "SCRATCH_BASE_MODELS_TRAINED_ROUTER_RETRAIN_SEPARATE",
        "training_anchor_count": len(anchors),
        "feature_count": len(feature_columns),
        "single_iterations": 700,
        "multi_iterations": 1200,
        "pretrained_weights_loaded": 0,
        "external_data_rows": 0,
        "runtime_seconds": time.perf_counter() - started,
        "router_training_sources": [
            "07_source/scripts/run_p3_component_loss_router.py",
            "07_source/scripts/run_p3_corrected_repeated_forward_catboost_v2.py",
        ],
    }
    (target / "TRAINING_RECEIPT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=HERE.parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("03_model/retrained_base"))
    args = parser.parse_args()
    print(json.dumps(train(args.data_dir, args.package_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
