"""Isolate the fixed 59-feature event-phase increment with matched GPU CatBoost arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from p3_wave.models import compact_feature_columns, threshold_case_weights
from p3_wave.validation import build_forecast_folds, expand_leads, metric_slices


def _model(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=700,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=8.0,
        random_strength=0.2,
        random_seed=seed,
        task_type="GPU",
        devices="0",
        verbose=False,
        allow_writing_files=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--event-cache", default="artifacts/p3/features_event_v1")
    parser.add_argument("--output-dir", default="artifacts/p3/event_phase_probe")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    anchors = pd.read_parquet(Path(args.base_cache) / "train_anchors.parquet")
    feature_frames = {
        "base_gpu": pd.read_parquet(Path(args.base_cache) / "train_features.parquet"),
        "event_gpu": pd.read_parquet(Path(args.event_cache) / "train_features.parquet"),
    }
    rows: list[pd.DataFrame] = []
    feature_counts: dict[str, int] = {}
    for arm, features in feature_frames.items():
        columns = compact_feature_columns(
            [column for column in features if column not in {"anchor_id", "station"}]
        )
        feature_counts[arm] = len(columns)
        for fold_number, fold in enumerate(build_forecast_folds(anchors)):
            x_train, y_train, train_meta = expand_leads(features, anchors, fold.train_ids, columns)
            x_valid, _, valid_meta = expand_leads(features, anchors, fold.validation_ids, columns)
            for frame in (x_train, x_valid):
                frame["station"] = frame["station"].astype(str)
                frame["lead_h"] = frame["lead_h"].astype(str)
            model = _model(20260817 + fold_number)
            model.fit(
                x_train,
                y_train,
                sample_weight=threshold_case_weights(train_meta["current_hs"].to_numpy()),
                cat_features=[0, 1],
                verbose=False,
            )
            prediction = np.clip(
                valid_meta["current_hs"].to_numpy() + model.predict(x_valid), 0.0, 30.0
            )
            frame = valid_meta.copy()
            frame["fold"] = fold.name
            frame["arm"] = arm
            frame["prediction"] = prediction
            rows.append(frame)
    oof = pd.concat(rows, ignore_index=True)
    metrics = {
        arm: metric_slices(group, group["prediction"].to_numpy())
        for arm, group in oof.groupby("arm", observed=True)
    }
    base = oof.loc[oof["arm"].eq("base_gpu")]
    metrics["persistence"] = metric_slices(base, base["current_hs"].to_numpy())
    oof_path = output / "oof.parquet"
    oof.to_parquet(oof_path, index=False, compression="zstd")
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "research_only_matched_increment",
        "gpu_nondeterministic": True,
        "feature_counts": feature_counts,
        "metrics": metrics,
        "oof_sha256": hashlib.sha256(oof_path.read_bytes()).hexdigest(),
    }
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
