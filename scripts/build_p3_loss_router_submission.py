"""Fit the validated P3 loss router and build a non-uploaded submission candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from p3_wave.data import LEADS, load_p3_data, resolve_p3_data_dir
from p3_wave.loss_router import (
    OBSERVED_FEATURES,
    ComponentLossRouter,
    RouterConfig,
    build_case_router_data,
    build_inference_router_features,
    expand_case_router_features,
    expand_case_router_rows,
    route_row_predictions,
)
from p3_wave.submission import build_submission, write_submission

ACTIVE_LEADS = (12, 18, 24)
KEYS = ["case_id", "station", "lead_h"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _aligned_component_cases(
    test_index: pd.DataFrame,
    raw: pd.DataFrame,
    multi: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    for name, frame in (("single", raw), ("multi", multi)):
        if not frame[KEYS].equals(test_index[KEYS]):
            raise ValueError(f"{name} component keys do not exactly match test_index")
    joined = test_index[KEYS].copy()
    joined["single"] = raw["hs_pred"].to_numpy(float)
    joined["multi"] = multi["hs_pred"].to_numpy(float)
    current_lookup = test_features.set_index(["case_id", "station"])["hs_current"]
    key_index = pd.MultiIndex.from_frame(joined[["case_id", "station"]])
    joined["persistence"] = current_lookup.loc[key_index].to_numpy(float)
    case_order = test_index[["case_id", "station"]].drop_duplicates().reset_index(drop=True)
    blocks: list[np.ndarray] = []
    current: list[float] = []
    for row in case_order.itertuples(index=False):
        block = joined.loc[
            joined["case_id"].eq(row.case_id) & joined["station"].eq(row.station)
        ].sort_values("lead_h")
        if tuple(block["lead_h"].astype(int)) != tuple(LEADS):
            raise ValueError("test component case is missing an official lead")
        blocks.append(block[["single", "multi", "persistence"]].to_numpy(float))
        current.append(float(block["persistence"].iloc[0]))
    return case_order, np.stack(blocks), np.asarray(current, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--cache-dir", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--router-dir", default="artifacts/p3/lead_long_loss_router")
    parser.add_argument("--frozen-dir", default="submissions/p3_frozen_catboost")
    parser.add_argument("--output-dir", default="submissions/p3_lead_long_loss_router")
    args = parser.parse_args()
    cache = Path(args.cache_dir)
    router_dir = Path(args.router_dir)
    frozen_dir = Path(args.frozen_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_root = resolve_p3_data_dir(args.data_dir)
    data = load_p3_data(data_root)
    test_index = data.test_index[KEYS].copy()

    frozen_manifest = json.loads((frozen_dir / "manifest.json").read_text(encoding="utf-8"))
    component_paths = {
        "single": frozen_dir / "submission_raw.csv",
        "multi": frozen_dir / "submission_multi.csv",
    }
    for name, path in component_paths.items():
        manifest_name = "submission_raw.csv" if name == "single" else "submission_multi.csv"
        if _sha256(path) != frozen_manifest["artifact_sha256"][manifest_name]:
            raise ValueError(f"saved {name} component hash does not match frozen manifest")
    raw = pd.read_csv(component_paths["single"])
    multi = pd.read_csv(component_paths["multi"])
    test_features = pd.read_parquet(cache / "test_features.parquet")
    case_order, test_components, test_current = _aligned_component_cases(
        test_index, raw, multi, test_features
    )

    oof = pd.read_parquet("artifacts/p3/final_ensemble_validation/oof.parquet")
    train_features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    case_x, case_meta, case_components, _ = build_case_router_data(oof, train_features, anchors)
    ordered = oof.sort_values(["fold", "anchor_id", "lead_h"])
    truth = np.stack(
        [
            group.sort_values("lead_h")["target_hs"].to_numpy(float)
            for _, group in ordered.groupby(["fold", "anchor_id"], sort=False)
        ]
    )
    row_x, _, _, row_losses = expand_case_router_rows(case_x, case_meta, case_components, truth)
    router_metrics_path = router_dir / "metrics.json"
    router_metrics = json.loads(router_metrics_path.read_text(encoding="utf-8"))
    last = next(row for row in router_metrics["selections"] if row["fold"] == "2025_h1")
    selected = RouterConfig(
        alpha=float(last["config"]["alpha"]),
        temperature_multiplier=float(last["config"]["temperature_multiplier"]),
        strength=float(last["config"]["strength"]),
        name=str(last["selected"]),
    )
    router = ComponentLossRouter(selected).fit(row_x, row_losses)

    observed = case_order.merge(
        test_features[["case_id", "station", *OBSERVED_FEATURES]],
        on=["case_id", "station"],
        how="left",
        validate="one_to_one",
    )
    if observed[list(OBSERVED_FEATURES)].isna().all(axis=1).any():
        raise ValueError("test router observed features failed to align")
    test_case_x = build_inference_router_features(
        observed.loc[:, OBSERVED_FEATURES],
        case_order["station"].to_numpy(str),
        test_current,
        test_components,
    )
    test_meta = case_order.copy()
    test_meta["fold"] = "hidden_test"
    test_meta["anchor_id"] = np.arange(len(test_meta), dtype=np.int64)
    test_meta["anchor_time"] = pd.NaT
    test_row_x, test_row_meta, test_row_components = expand_case_router_features(
        test_case_x,
        test_meta[["fold", "anchor_id", "station", "anchor_time"]],
        test_components,
    )
    weights = router.predict_weights(test_row_x)
    inactive = ~test_row_meta["lead_h"].isin(ACTIVE_LEADS).to_numpy()
    weights[inactive] = np.array([0.5, 0.5, 0.0])
    routed_prediction = route_row_predictions(test_row_components, weights)
    routed = pd.DataFrame(
        {
            "case_id": np.repeat(case_order["case_id"].to_numpy(str), len(LEADS)),
            "station": np.repeat(case_order["station"].to_numpy(str), len(LEADS)),
            "lead_h": np.tile(np.asarray(LEADS, dtype=int), len(case_order)),
            "hs_pred": routed_prediction,
        }
    )
    prediction = test_index.merge(routed, on=KEYS, how="left", validate="one_to_one")[
        "hs_pred"
    ].to_numpy(float)
    submission_path = write_submission(
        build_submission(test_index, prediction), test_index, output / "submission.csv"
    )
    frozen_submission = pd.read_csv(frozen_dir / "submission.csv")
    if not frozen_submission[KEYS].equals(test_index):
        raise ValueError("frozen submission keys changed")
    short = test_index["lead_h"].isin([3, 6, 9]).to_numpy()
    short_error = float(
        np.max(np.abs(prediction[short] - frozen_submission.loc[short, "hs_pred"].to_numpy(float)))
    )
    if short_error > 1e-12:
        raise ValueError("short-lead submission rows must reproduce the frozen candidate")
    model_path = output / "router.joblib"
    joblib.dump(router, model_path)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "local_candidate_not_uploaded",
        "problem": "P3_wave_forecast",
        "model": "long-lead chronological component-loss soft router over frozen CatBoost components",
        "active_leads": list(ACTIVE_LEADS),
        "selected_router_config": last,
        "fit_cases": int(len(case_meta)),
        "fit_rows": int(len(row_x)),
        "test_cases": int(len(case_order)),
        "test_rows": int(len(test_index)),
        "weight_summary": {
            name: {
                "mean": float(weights[:, index].mean()),
                "p10": float(np.quantile(weights[:, index], 0.10)),
                "p90": float(np.quantile(weights[:, index], 0.90)),
            }
            for index, name in enumerate(("single", "multi", "persistence"))
        },
        "prediction": {
            "minimum": float(np.min(prediction)),
            "median": float(np.median(prediction)),
            "maximum": float(np.max(prediction)),
            "short_lead_frozen_max_abs_difference": short_error,
        },
        "git_sha": _git_sha(),
        "source_sha256": {
            "router_metrics": _sha256(router_metrics_path),
            "router_oof": _sha256(router_dir / "oof.parquet"),
            "frozen_single_submission": _sha256(component_paths["single"]),
            "frozen_multi_submission": _sha256(component_paths["multi"]),
            "test_features": _sha256(cache / "test_features.parquet"),
        },
        "artifact_sha256": {
            "submission.csv": _sha256(submission_path),
            "router.joblib": _sha256(model_path),
        },
        "external_observations_used": 0,
        "hidden_test_labels_used": 0,
        "uploaded": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
