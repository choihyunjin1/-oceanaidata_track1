"""Run a fixed P3 structural model tournament on chronological independent cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from p3_wave.models import ResidualRegressor, compact_feature_columns, threshold_case_weights
from p3_wave.validation import build_forecast_folds, expand_leads, metric_slices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="artifacts/p3/features_v1")
    parser.add_argument("--output-dir", default="artifacts/p3/initial_tournament")
    parser.add_argument(
        "--backends", nargs="+", default=["ridge", "lightgbm", "xgboost", "catboost"]
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache = Path(args.cache_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    all_features = [c for c in features.columns if c not in {"anchor_id", "station"}]
    compact = compact_feature_columns(all_features)
    folds = build_forecast_folds(anchors)
    configurations = [(backend, compact) for backend in args.backends]
    rows: list[pd.DataFrame] = []
    fold_reports: dict[str, object] = {}
    for backend, selected in configurations:
        backend_rows: list[pd.DataFrame] = []
        for number, fold in enumerate(folds):
            x_train, y_train, train_meta = expand_leads(features, anchors, fold.train_ids, selected)
            x_valid, _, valid_meta = expand_leads(features, anchors, fold.validation_ids, selected)
            weights = threshold_case_weights(train_meta["current_hs"].to_numpy())
            model = ResidualRegressor(backend, seed=20260816 + number)
            model.fit(x_train, y_train, sample_weight=weights)
            delta = model.predict_delta(x_valid)
            prediction = np.clip(valid_meta["current_hs"].to_numpy() + delta, 0.0, 30.0)
            frame = valid_meta.copy()
            frame["fold"] = fold.name
            frame["backend"] = backend
            frame["prediction"] = prediction
            frame["persistence"] = frame["current_hs"]
            backend_rows.append(frame)
        combined = pd.concat(backend_rows, ignore_index=True)
        rows.append(combined)
        fold_reports[backend] = {
            "candidate": metric_slices(combined, combined["prediction"].to_numpy()),
            "persistence": metric_slices(combined, combined["persistence"].to_numpy()),
            "folds": {
                name: metric_slices(group, group["prediction"].to_numpy())
                for name, group in combined.groupby("fold", observed=True)
            },
        }
    oof = pd.concat(rows, ignore_index=True)
    oof_path = output / "oof.parquet"
    oof.to_parquet(oof_path, index=False, compression="zstd")
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "research_only_initial_structure_tournament",
        "feature_count": len(compact),
        "feature_surface": "compact_fixed_v1",
        "dense_anchor_spacing_minutes": 60,
        "selection": "first eligible per station then 78h separation inside validation windows",
        "target": "target_hs_minus_current_hs",
        "sample_weight": "exp(-0.45*max(current_hs-1.5,0))",
        "metrics": fold_reports,
        "oof_sha256": hashlib.sha256(oof_path.read_bytes()).hexdigest(),
    }
    result_path = output / "metrics.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
