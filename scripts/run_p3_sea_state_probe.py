"""Matched GPU CatBoost ablation for eight fixed P3 wind-sea state features."""

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
from p3_wave.sea_state import SEA_STATE_FEATURES, append_sea_state_features
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
        boosting_type="Plain",
        verbose=False,
        allow_writing_files=False,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--output-dir", default="artifacts/p3/sea_state_probe")
    args = parser.parse_args()
    cache = Path(args.cache_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    base = pd.read_parquet(cache / "train_features.parquet")
    candidate = append_sea_state_features(base)
    frames = {"base_gpu": base, "sea_state_gpu": candidate}

    rows: list[pd.DataFrame] = []
    feature_counts: dict[str, int] = {}
    for arm, features in frames.items():
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
                valid_meta["current_hs"].to_numpy(float) + model.predict(x_valid), 0.0, 30.0
            )
            part = valid_meta.copy()
            part["fold"] = fold.name
            part["arm"] = arm
            part["prediction"] = prediction
            rows.append(part)

    oof = pd.concat(rows, ignore_index=True)
    metrics = {
        arm: metric_slices(group, group["prediction"].to_numpy(float))
        for arm, group in oof.groupby("arm", observed=True)
    }
    base_rows = oof.loc[oof["arm"].eq("base_gpu")]
    metrics["persistence"] = metric_slices(base_rows, base_rows["current_hs"].to_numpy(float))
    metrics["delta_candidate_minus_base"] = {
        "rmse": float(metrics["sea_state_gpu"]["rmse"] - metrics["base_gpu"]["rmse"]),
        "by_lead": {
            lead: float(
                metrics["sea_state_gpu"]["by_lead"][lead] - metrics["base_gpu"]["by_lead"][lead]
            )
            for lead in metrics["base_gpu"]["by_lead"]
        },
        "by_station": {
            station: float(
                metrics["sea_state_gpu"]["by_station"][station]
                - metrics["base_gpu"]["by_station"][station]
            )
            for station in metrics["base_gpu"]["by_station"]
        },
    }
    oof_path = output / "oof.parquet"
    oof.to_parquet(oof_path, index=False, compression="zstd")
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "research_only_matched_fixed_physical_feature_ablation",
        "gpu_nondeterministic": True,
        "feature_family": list(SEA_STATE_FEATURES),
        "feature_counts": feature_counts,
        "tp_caveat": "tp is mean period, not peak period; inverse wave age is an explicit proxy",
        "metrics": metrics,
        "provenance": {
            "train_features_sha256": _sha256(cache / "train_features.parquet"),
            "train_anchors_sha256": _sha256(cache / "train_anchors.parquet"),
            "oof_sha256": _sha256(oof_path),
            "external_observations_used": 0,
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
