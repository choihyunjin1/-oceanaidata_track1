"""Conditional P3-B winner full training and local-only fresh process replay.

No official input, CSV, upload, or old saved model/router is permitted here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402
from run_p3_episode_weight_20260905_v2 import (  # noqa: E402
    helpers,
    prepare,
    save,
    sha,
    stamp,
    training_weights,
)

from p3_wave.loss_router import (  # noqa: E402
    OBSERVED_FEATURES,
    ComponentLossRouter,
    build_case_router_data,
    build_inference_router_features,
    expand_case_router_features,
    expand_case_router_rows,
    route_row_predictions,
)
from p3_wave.persistence_shrink import (  # noqa: E402
    LongLeadPersistenceShrink,
    apply_long_lead_persistence_shrink,
)
from p3_wave.validation import expand_leads  # noqa: E402

NAME = "p3_episode_policy_fulltrain_20260905_v2"
CONFIG = ROOT / "configs/experiments" / f"{NAME}.json"
OUT = ROOT / "artifacts" / NAME
REPORT = ROOT / "reports" / NAME
LOCK = ROOT / "artifacts" / f"{NAME}.ATTEMPT_LOCK.json"
B_OUT = ROOT / "artifacts/p3_episode_weight_20260905_v2"
B_REPORT = ROOT / "reports/p3_episode_weight_20260905_v2"
LEADS = (3, 6, 9, 12, 18, 24)


def guard(source, b_config):
    permitted = {(ROOT / path).resolve() for path in b_config["inputs"]}
    permitted |= {B_OUT / "oof_control.parquet", B_OUT / "oof_episode_weight.parquet", B_OUT / "validation_keys.parquet"}
    allowed_source = {source / path for path in b_config["source_files"]}

    def audit(event, args):
        if event == "socket.connect":
            raise PermissionError("network denied")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if "external_data" in path.parts or "hidden" in path.name.lower():
            raise PermissionError("external/hidden file denied")
        if source in path.parents and (path not in allowed_source or (isinstance(args[1], str) and any(c in args[1] for c in "wax+"))):
            raise PermissionError("official source input not authorized")
        if path.suffix.lower() == ".csv" and path not in allowed_source:
            raise PermissionError("CSV denied")
        if path.suffix.lower() in {".parquet", ".cbm", ".joblib", ".pt", ".ckpt"} and path not in permitted and OUT not in path.parents:
            raise PermissionError("old/unapproved artifact denied")

    sys.addaudithook(audit)


def eligibility(config):
    b_config_path = ROOT / config["b_config"]
    if sha(b_config_path) != config["b_config_sha256"] or sha(ROOT / "scripts/run_p3_episode_weight_20260905_v2.py") != config["b_runner_sha256"]:
        raise ValueError("B frozen implementation/config changed")
    b_result = json.loads((B_REPORT / "result.json").read_text(encoding="utf-8"))
    qa = json.loads((B_REPORT / "independent-qa.json").read_text(encoding="utf-8"))
    if qa["status"] != "PASS" or qa["result_sha256"] != sha(B_REPORT / "result.json"):
        raise ValueError("completed matching B independent QA required")
    winner = b_result["winner"]
    if b_result["status"] != "COMPLETE" or winner not in ("control", "episode_weight") or not b_result["metric_rmse_m"][winner] < b_result["metric_rmse_m"]["legacy_control"]:
        raise ValueError("no independently verified improvement; no full fitting")
    if any(b_result["official_access"].values()) or b_result["public_score_fitting"]:
        raise ValueError("B data contract mismatch")
    path = B_OUT / f"oof_{winner}.parquet"
    if sha(path) != b_result["artifact_sha256"][path.name]:
        raise ValueError("selected OOF hash changed")
    return json.loads(b_config_path.read_text(encoding="utf-8")), b_result, winner


def predict_cases(cases, columns, models, router):
    """Label-free inference surface reusable for an authorized later deploy stage."""
    if cases.anchor_id.duplicated().any():
        raise ValueError("duplicate case ids")
    n = len(cases)
    current = cases.hs_current.to_numpy(float)
    matrix = cases[columns].iloc[np.repeat(np.arange(n), 6)].reset_index(drop=True)
    matrix.insert(0, "station", np.repeat(cases.station.astype(str).to_numpy(), 6))
    matrix.insert(1, "lead_h", np.tile(LEADS, n).astype(str))
    matrix.insert(2, "current_hs_for_residual", np.repeat(current, 6))
    multi_x = cases[["station", *columns]].copy()
    multi_x.station = multi_x.station.astype(str)
    singles, multis = [], []
    for single, multi in models:
        singles.append(np.clip(np.repeat(current, 6)+single.predict(matrix, thread_count=2, task_type="CPU"), 0, 30).reshape(n, 6))
        multis.append(np.clip(current[:, None]+multi.predict(multi_x, thread_count=2, task_type="CPU"), 0, 30))
    component = np.stack([np.mean(singles, axis=0), np.mean(multis, axis=0), np.repeat(current[:, None], 6, axis=1)], axis=2)
    case_x = build_inference_router_features(cases.loc[:, OBSERVED_FEATURES], cases.station.to_numpy(str), current, component)
    metadata = pd.DataFrame({"fold": "train_context_replay_not_validation", "anchor_id": cases.anchor_id.to_numpy(), "station": cases.station.to_numpy(str), "anchor_time": pd.NaT})
    row_x, row_meta, row_components = expand_case_router_features(case_x, metadata, component)
    weights = router.predict_weights(row_x)
    active = row_meta.lead_h.isin([12, 18, 24]).to_numpy()
    weights[~active] = [0.5, 0.5, 0.0]
    routed = route_row_predictions(row_components, weights)
    prediction = apply_long_lead_persistence_shrink(routed, component[:, :, 2].reshape(-1), row_meta.lead_h.to_numpy(), config=LongLeadPersistenceShrink(weight=0.2, active_leads=(12, 18, 24)))
    output = row_meta[["anchor_id", "station", "lead_h"]].copy()
    output[["single_prediction", "multi_prediction", "persistence"]] = component.reshape(-1, 3)
    output["final_prediction"] = prediction
    if not np.isfinite(output.final_prediction).all() or not output.final_prediction.between(0, 30).all():
        raise ValueError("nonfinite/out-of-range replay")
    return output


def train(config, source):
    started = time.perf_counter()
    if LOCK.exists() or OUT.exists() or (REPORT / "result.json").exists():
        raise RuntimeError("exactly-once fullfit attempt already exists")
    b_config, b_result, winner = eligibility(config)
    guard(source, b_config)
    recipe, features, anchors, columns, _, selected, _, integrity = prepare(b_config, source)
    base = helpers(b_config)
    aligned_ids = np.sort(anchors.anchor_id.to_numpy())
    if not np.array_equal(features.set_index("anchor_id").loc[aligned_ids, "hs_current"].to_numpy(), anchors.set_index("anchor_id").loc[aligned_ids, "current_hs"].to_numpy()):
        raise ValueError("training/inference current-hs contract mismatch")
    if config["full_seeds"] != [20260817, 20260917]:
        raise ValueError("full seed schedule changed")
    manifest = {"created_utc": stamp(), "pid": os.getpid(), "runner_sha256": sha(Path(__file__)), "config_sha256": sha(CONFIG), "b_result_sha256": sha(B_REPORT / "result.json"), "b_qa_sha256": sha(B_REPORT / "independent-qa.json"), "winner": winner, "verified_inputs": integrity["verified_inputs"]}
    save(LOCK, manifest)
    OUT.mkdir(parents=True)
    receipts = []
    try:
        ids = np.sort(anchors.anchor_id.to_numpy())
        weights = training_weights(anchors, ids, winner)
        x_train, y_train, meta = expand_leads(features, anchors, ids, columns)
        x_train = base._cat_frame(x_train)
        single_weights = pd.Series(weights, index=ids).loc[meta.anchor_id].to_numpy()
        multi_x = features.set_index("anchor_id").loc[ids, ["station", *columns]].reset_index(drop=True)
        multi_x.station = multi_x.station.astype(str)
        models = []
        for seed in config["full_seeds"]:
            destination = OUT / "models" / str(seed)
            destination.mkdir(parents=True)
            pair = []
            for kind in ("single", "multi"):
                fit_start = time.perf_counter()
                model = base._single_model(recipe, seed) if kind == "single" else base._multi_model(recipe, seed)
                if kind == "single":
                    model.fit(x_train, y_train, sample_weight=single_weights, cat_features=[0, 1], verbose=False)
                else:
                    model.fit(multi_x, base._multi_target(anchors, ids), sample_weight=weights, cat_features=[0], verbose=False)
                path = destination / f"{kind}.cbm"
                model.save_model(path)
                pair.append(model)
                receipts.append({"seed": seed, "component": kind, "model_sha256": sha(path), "model_path": path.relative_to(OUT).as_posix(), "elapsed_seconds": time.perf_counter()-fit_start, "train_anchors": len(ids), "train_rows": len(x_train) if kind == "single" else len(ids)})
                progress = {"stage": "full_backbone", "completed_fits": len(receipts), "maximum_fits": 4, "elapsed_seconds": time.perf_counter()-started, "pid": os.getpid()}
                save(OUT / "progress.json", progress, progress=True)
                print(json.dumps(progress), flush=True)
            models.append(tuple(pair))
        oof = pd.read_parquet(B_OUT / f"oof_{winner}.parquet")
        case_x, case_meta, components, _ = build_case_router_data(oof, features, anchors)
        lookup = anchors.set_index("anchor_id")
        truth = np.column_stack([lookup.loc[case_meta.anchor_id, f"target_{lead}"].to_numpy(float) for lead in LEADS])
        row_x, _, _, row_losses = expand_case_router_rows(case_x, case_meta, components, truth)
        router = ComponentLossRouter(base._router_config(recipe)).fit(row_x, row_losses)
        joblib.dump(router, OUT / "router.joblib")
        replay_cases = features.set_index("anchor_id").loc[np.sort(selected.anchor_id.to_numpy())].reset_index()
        replay = predict_cases(replay_cases, columns, models, router)
        replay_cases.to_parquet(OUT / "replay_cases.parquet", index=False)
        replay.to_parquet(OUT / "replay_predictions.parquet", index=False)
        save(OUT / "feature_columns.json", {"columns": columns})
        result = {"experiment_id": NAME, "status": "FULL_TRAIN_COMPLETE_PENDING_FRESH_PROCESS_REPLAY", "winner": winner, "fit_count": {"full_backbone": 4, "full_router": 1, "historical": 0}, "router_train_cases": len(case_meta), "router_train_rows": len(row_x), "router_old_coefficients_copied": False, "component_mean_before_router": True, "historical_selection_rmse_m": b_result["metric_rmse_m"][winner], "replay_is_validation": False, "training_replay_metric_not_computed": True, "elapsed_seconds": time.perf_counter()-started, "fit_receipts": receipts, "manifest": manifest, "official_access": {"test": 0, "sample": 0, "hidden": 0, "submission_csv": 0, "uploads": 0}, "public_score_fitting": False, "artifact_sha256": {path.name: sha(path) for path in [OUT / "router.joblib", OUT / "replay_cases.parquet", OUT / "replay_predictions.parquet", OUT / "feature_columns.json"]}}
        save(REPORT / "result.json", result)
        save(OUT / "terminal_result.json", {"status": result["status"], "result_sha256": sha(REPORT / "result.json"), "fit_count": result["fit_count"]})
        return {"status": result["status"], "winner": winner, "fit_count": result["fit_count"], "elapsed_seconds": result["elapsed_seconds"]}
    except Exception as error:
        save(OUT / "terminal_result.json", {"status": "TERMINAL_TECHNICAL_FAILURE", "exception_type": type(error).__name__, "message": str(error), "full_backbone_fits_completed": len(receipts), "automatic_retry": False})
        raise


def replay(config, source):
    b_config, _, _ = eligibility(config)
    guard(source, b_config)
    result = json.loads((REPORT / "result.json").read_text(encoding="utf-8"))
    if result["manifest"]["runner_sha256"] != sha(Path(__file__)) or result["manifest"]["config_sha256"] != sha(CONFIG):
        raise ValueError("frozen full implementation changed")
    if os.getpid() == result["manifest"]["pid"]:
        raise ValueError("replay must be a fresh process")
    if not all(sha(OUT / name) == value for name, value in result["artifact_sha256"].items()):
        raise ValueError("full artifact hash mismatch")
    if not all(sha(OUT / row["model_path"]) == row["model_sha256"] for row in result["fit_receipts"]):
        raise ValueError("full model hash mismatch")
    cases = pd.read_parquet(OUT / "replay_cases.parquet")
    columns = json.loads((OUT / "feature_columns.json").read_text(encoding="utf-8"))["columns"]
    models = [(CatBoostRegressor().load_model(OUT / "models" / str(seed) / "single.cbm"), CatBoostRegressor().load_model(OUT / "models" / str(seed) / "multi.cbm")) for seed in config["full_seeds"]]
    router = joblib.load(OUT / "router.joblib")
    fresh = predict_cases(cases, columns, models, router)
    stored = pd.read_parquet(OUT / "replay_predictions.parquet")
    keys = ["anchor_id", "station", "lead_h"]
    if not fresh[keys].equals(stored[keys]):
        raise ValueError("fresh-process replay key mismatch")
    error = float(np.max(np.abs(fresh.final_prediction.to_numpy()-stored.final_prediction.to_numpy())))
    if error != 0.0:
        raise ValueError("fresh-process replay differs")
    proof = {"status": "PASS", "created_utc": stamp(), "process_id": os.getpid(), "training_process_id": result["manifest"]["pid"], "models_reloaded": 4, "router_reloaded": 1, "cases": len(cases), "rows": len(fresh), "maximum_absolute_difference_m": error, "new_fits": 0, "official_inputs": 0, "prediction_device": "CPU", "result_sha256": sha(REPORT / "result.json"), "training_replay_not_validation": True}
    save(REPORT / "fresh-process-replay.json", proof)
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("train", "replay"))
    parser.add_argument("--gpu-approved", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    if args.mode == "train" and not args.gpu_approved:
        raise PermissionError("root GPU approval required")
    print(json.dumps(train(config, source) if args.mode == "train" else replay(config, source)), flush=True)
