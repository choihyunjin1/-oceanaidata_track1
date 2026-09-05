"""Frozen 18-fit P3 episode-weight comparison. Research-only, no official input.

The old control seed is reused as train-derived OOF, not a saved router or answer.
New two-seed components are averaged before separate past-only router refits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402
from run_p3_direct_sse_meta_20260905_v2 import bootstrap, slices  # noqa: E402

from p3_wave.corrected_repeated_forward import build_corrected_repeated_forward_folds  # noqa: E402
from p3_wave.models import threshold_case_weights  # noqa: E402
from p3_wave.revin_patch import (  # noqa: E402
    assign_storm_episodes_from_wave,
    event_balanced_weights,
)
from p3_wave.validation import expand_leads  # noqa: E402

NAME = "p3_episode_weight_20260905_v2"
CONFIG = ROOT / "configs/experiments" / f"{NAME}.json"
OUT = ROOT / "artifacts" / NAME
SMOKE = ROOT / "artifacts" / f"{NAME}_synthetic_smoke"
REPORT = ROOT / "reports" / NAME
LOCK = ROOT / "artifacts" / f"{NAME}.ATTEMPT_LOCK.json"
KEYS = ["anchor_id", "station", "lead_h", "fold"]
LEADS = (3, 6, 9, 12, 18, 24)


def stamp():
    return datetime.now(UTC).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj, *, progress=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if progress else "x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False, default=str)


def helpers(config):
    spec = importlib.util.spec_from_file_location("p3_episode_weight_base", ROOT / config["base_runner"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_weights(anchors, ids, arm):
    ids = np.asarray(ids, dtype=np.int64)
    selected = anchors.set_index("anchor_id").loc[ids]
    if selected.current_hs.lt(1.5).any():
        raise ValueError("background anchor not eligible for this storm-only experiment")
    weights = threshold_case_weights(selected.current_hs.to_numpy(float))
    if arm == "episode_weight":
        weights *= event_balanced_weights(anchors, ids)
        weights /= weights.mean()
    elif arm != "control":
        raise ValueError("unexpected weight arm")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("weights invalid")
    return weights


def average_components(first, second):
    """Key-aligned mean of two complete component predictions; truth must agree."""
    columns = [*KEYS, "current_hs", "target_hs", "single_prediction", "multi_prediction", "persistence"]
    left = first[columns].sort_values(KEYS).reset_index(drop=True)
    right = second[columns].sort_values(KEYS).reset_index(drop=True)
    if left.duplicated(KEYS).any() or right.duplicated(KEYS).any() or not left[KEYS].equals(right[KEYS]):
        raise ValueError("component ensemble key mismatch")
    for column in ("current_hs", "target_hs", "persistence"):
        if not np.array_equal(left[column], right[column]):
            raise ValueError("component ensemble target/context mismatch")
    for column in ("single_prediction", "multi_prediction"):
        left[column] = (left[column].to_numpy() + right[column].to_numpy()) / 2
    left["equal_prediction"] = (left.single_prediction + left.multi_prediction) / 2
    return left


def install_guard(source, config):
    permitted = {(ROOT / name).resolve() for name in config["inputs"]}
    allowed_source = {source / name for name in config["source_files"]}

    def guard(event, args):
        if event == "socket.connect":
            raise PermissionError("network not allowed")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        owned = OUT in path.parents or SMOKE in path.parents
        if "external_data" in path.parts or "hidden" in path.name.lower():
            raise PermissionError("external/hidden path denied")
        if source in path.parents and (path not in allowed_source or (isinstance(args[1], str) and any(c in args[1] for c in "wax+"))):
            raise PermissionError("source input not authorized")
        if path.suffix.lower() == ".csv" and path not in allowed_source:
            raise PermissionError("CSV denied")
        if path.suffix.lower() in {".cbm", ".ckpt", ".pt", ".joblib", ".tabpfn_fit", ".parquet"} and path not in permitted and not owned:
            raise PermissionError("unapproved artifact/checkpoint denied")

    sys.addaudithook(guard)


def prepare(config, source):
    verified = {}
    for name, expected in config["inputs"].items():
        verified[name] = sha(ROOT / name)
        if verified[name] != expected:
            raise ValueError(f"pinned input hash differs: {name}")
    for name, expected in config["source_files"].items():
        verified[f"source/{name}"] = sha(source / name)
        if verified[f"source/{name}"] != expected:
            raise ValueError("source training hash differs")
    recipe = json.loads((ROOT / config["reference_config"]).read_text(encoding="utf-8"))
    for part in ("single", "multi"):
        recipe["model"][part]["thread_count"] = 2
    cache = ROOT / config["cache"]
    features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    columns = json.loads((cache / "feature_columns.json").read_text(encoding="utf-8"))["columns"]
    wave = pd.read_csv(source / "train_wave.csv")
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    folds, selected, split = build_corrected_repeated_forward_folds(anchors, windows=recipe["validation"]["windows"], gap_hours=78, footprint_hours=72)
    keys = pd.read_parquet(cache / "validation_keys.parquet")
    sort = ["anchor_id", "station", "fold"]
    actual = selected.sort_values(sort).reset_index(drop=True)
    expected = keys.sort_values(sort).reset_index(drop=True)
    if not actual[sort].equals(expected[sort]) or not np.array_equal(actual.episode_id, expected.episode_id):
        raise ValueError("recomputed validation selection differs")
    if len(columns) != 591 or len(anchors) != 24360 or len(selected) != 181:
        raise ValueError("feature/case/anchor contract mismatch")
    base_oof = pd.read_parquet(cache / "oof.parquet")
    summaries = []
    for fold in folds:
        w = training_weights(anchors, fold.train_ids, "episode_weight")
        subset = anchors.set_index("anchor_id").loc[fold.train_ids]
        episode_sizes = subset.groupby(["station", "episode_id"]).size()
        summaries.append({"fold": fold.name, "train_anchors": len(fold.train_ids), "cases": len(fold.validation_ids), "train_episodes": len(episode_sizes), "largest_episode_train_anchor_count": int(episode_sizes.max()), "weight_min": float(w.min()), "weight_max": float(w.max()), "weight_mean": float(w.mean()), "weights_only_outer_train": True})
    integrity = {"verified_inputs": verified, "split_audit": split, "weight_audit": summaries, "feature_count": len(columns), "train_anchor_count": len(anchors), "cases": len(selected), "reuse_control_seed_one_oof_only": True, "official_input_rows": 0}
    return recipe, features, anchors, columns, folds, selected, base_oof, integrity


def synthetic_smoke(config):
    if SMOKE.exists() or (REPORT / "synthetic-smoke.json").exists():
        raise RuntimeError("synthetic smoke already recorded")
    recipe = json.loads((ROOT / config["reference_config"]).read_text(encoding="utf-8"))
    rng = np.random.default_rng(20260905)
    matrix = pd.DataFrame({"station": np.repeat(["A", "B"], 24), "x": rng.normal(size=48), "z": rng.normal(size=48)})
    weights = np.linspace(0.1, 2, 48)
    SMOKE.mkdir(parents=True)
    rows = []
    for kind in ("single", "multi"):
        params = dict(recipe["model"][kind])
        params.update(iterations=3, thread_count=2)
        y = rng.normal(size=48) if kind == "single" else rng.normal(size=(48, 6))
        model = CatBoostRegressor(**params, random_seed=20260905, verbose=False, allow_writing_files=False)
        model.fit(matrix, y, sample_weight=weights, cat_features=[0])
        before = model.predict(matrix, thread_count=2)
        path = SMOKE / f"{kind}.cbm"
        model.save_model(path)
        reloaded = CatBoostRegressor().load_model(path)
        error = float(np.max(np.abs(before - reloaded.predict(matrix, thread_count=2))))
        if error != 0.0:
            raise ValueError("synthetic reload parity failed")
        rows.append({"kind": kind, "synthetic_rows": 48, "synthetic_fit_count": 1, "reload_max_abs": error, "task_type": params.get("task_type", "CPU"), "weights_supported": True})
    result = {"status": "PASS", "historical_fits": 0, "synthetic_fits": 2, "created_utc": stamp(), "runner_sha256": sha(Path(__file__)), "config_sha256": sha(CONFIG), "checks": rows}
    save(REPORT / "synthetic-smoke.json", result)
    return result


def fit_pair(base, recipe, features, anchors, columns, fold, seed, arm, seed_set, record_fit):
    started = time.perf_counter()
    x_train, y_train, train_meta = expand_leads(features, anchors, fold.train_ids, columns)
    x_valid, _, valid_meta = expand_leads(features, anchors, fold.validation_ids, columns)
    per_anchor = training_weights(anchors, fold.train_ids, arm)
    weight_lookup = pd.Series(per_anchor, index=fold.train_ids)
    single_w = weight_lookup.loc[train_meta.anchor_id].to_numpy()
    if not np.array_equal(single_w, np.tile(per_anchor, 6)):
        raise ValueError("lead expansion weights differ from original anchor mass")
    destination = OUT / "models" / arm / f"seed_set_{seed_set}" / fold.name
    destination.mkdir(parents=True)
    single = base._single_model(recipe, seed)
    single_started = time.perf_counter()
    single.fit(base._cat_frame(x_train), y_train, sample_weight=single_w, cat_features=[0, 1], verbose=False)
    single_raw = single.predict(base._cat_frame(x_valid), thread_count=2)
    single_path = destination / "single.cbm"
    single.save_model(single_path)
    loaded_single = CatBoostRegressor().load_model(single_path)
    single_error = float(np.max(np.abs(single_raw - loaded_single.predict(base._cat_frame(x_valid), thread_count=2))))
    if single_error != 0.0:
        raise ValueError("single saved model reload differs")
    record_fit({"arm": arm, "seed_set": seed_set, "seed": seed, "fold": fold.name, "component": "single", "elapsed_seconds": time.perf_counter() - single_started, "train_rows": len(x_train), "reload_max_abs": single_error, "model_sha256": sha(single_path)})
    feature_lookup = features.set_index("anchor_id")
    multi_train = feature_lookup.loc[fold.train_ids, ["station", *columns]].reset_index(drop=True)
    multi_valid = feature_lookup.loc[fold.validation_ids, ["station", *columns]].reset_index(drop=True)
    multi_train.station = multi_train.station.astype(str)
    multi_valid.station = multi_valid.station.astype(str)
    multi_started = time.perf_counter()
    multi = base._multi_model(recipe, seed)
    multi.fit(multi_train, base._multi_target(anchors, fold.train_ids), sample_weight=per_anchor, cat_features=[0], verbose=False)
    multi_raw = multi.predict(multi_valid, thread_count=2)
    multi_path = destination / "multi.cbm"
    multi.save_model(multi_path)
    loaded_multi = CatBoostRegressor().load_model(multi_path)
    multi_error = float(np.max(np.abs(multi_raw - loaded_multi.predict(multi_valid, thread_count=2))))
    if multi_error != 0.0:
        raise ValueError("multi saved model reload differs")
    record_fit({"arm": arm, "seed_set": seed_set, "seed": seed, "fold": fold.name, "component": "multi", "elapsed_seconds": time.perf_counter() - multi_started, "train_rows": len(multi_train), "reload_max_abs": multi_error, "model_sha256": sha(multi_path)})
    output = valid_meta.copy()
    output["fold"] = fold.name
    output["single_prediction"] = np.clip(valid_meta.current_hs.to_numpy(float) + single_raw, 0, 30)
    output = output.merge(base._multi_validation_frame(anchors, fold.validation_ids, multi_raw), on=["anchor_id", "station", "lead_h"], validate="one_to_one")
    output["persistence"] = output.current_hs
    output["equal_prediction"] = (output.single_prediction + output.multi_prediction) / 2
    return output, {"arm": arm, "seed_set": seed_set, "fold": fold.name, "seed": seed, "pair_seconds": time.perf_counter() - started, "weight_mean": float(per_anchor.mean()), "weight_sha256": hashlib.sha256(per_anchor.tobytes()).hexdigest()}


def execute(config, source):
    started = time.perf_counter()
    if LOCK.exists() or OUT.exists() or (REPORT / "result.json").exists():
        raise RuntimeError("exactly-once attempt/output exists; automatic retry forbidden")
    smoke = json.loads((REPORT / "synthetic-smoke.json").read_text(encoding="utf-8"))
    if smoke["status"] != "PASS" or smoke["config_sha256"] != sha(CONFIG) or smoke["runner_sha256"] != sha(Path(__file__)):
        raise ValueError("matching synthetic compatibility proof is required")
    recipe, features, anchors, columns, folds, selected, baseline, integrity = prepare(config, source)
    base = helpers(config)
    manifest = {"experiment_id": NAME, "created_utc": stamp(), "pid": os.getpid(), "runner_sha256": sha(Path(__file__)), "config_sha256": sha(CONFIG), "verified_inputs": integrity["verified_inputs"], "fit_budget": config["fit_budget"], "gpu_exclusive_approval": "root_release_after_P2_A_terminal", "cpu_threads": 2}
    save(LOCK, manifest)
    OUT.mkdir(parents=True)
    fit_receipts, pair_receipts = [], []

    def record_fit(receipt):
        fit_receipts.append(receipt)
        progress = {"created_utc": stamp(), "stage": "historical_backbone", "completed_fits": len(fit_receipts), "maximum_fits": 18, "arm": receipt["arm"], "fold": receipt["fold"], "component": receipt["component"], "elapsed_seconds": time.perf_counter() - started, "pid": os.getpid()}
        save(OUT / "fit-receipts.json", {"fits": fit_receipts, "pairs": pair_receipts}, progress=True)
        save(OUT / "progress.json", progress, progress=True)
        print(json.dumps(progress), flush=True)

    try:
        parts = {}
        for arm, seed_set in (("control", 1), ("episode_weight", 0), ("episode_weight", 1)):
            blocks = []
            for i, fold in enumerate(folds):
                print(json.dumps({"stage": "starting_pair", "arm": arm, "seed_set": seed_set, "fold": fold.name, "completed_fits": len(fit_receipts)}), flush=True)
                block, pair = fit_pair(base, recipe, features, anchors, columns, fold, config["fold_seed_sets"][seed_set][i], arm, seed_set, record_fit)
                blocks.append(block)
                pair_receipts.append(pair)
            parts[(arm, seed_set)] = pd.concat(blocks, ignore_index=True)
            parts[(arm, seed_set)].to_parquet(OUT / f"components_{arm}_{seed_set}.parquet", index=False)
        if len(fit_receipts) != 18:
            raise ValueError("historical fit budget mismatch")
        rebuilt = {}
        details = {}
        aligned_baseline = baseline.sort_values(KEYS).reset_index(drop=True)
        metadata = selected[["anchor_id", "station", "fold", "anchor_time", "episode_id"]]
        for arm in config["arms"]:
            first = baseline if arm == "control" else parts[(arm, 0)]
            component_oof = average_components(first, parts[(arm, 1)])
            oof, detail, _ = base._evaluate_fixed_structure(component_oof=component_oof, train_features=features, anchors=anchors, fold_order=tuple(f.name for f in folds), config=recipe, split_audit=integrity["split_audit"], expected_validation_ids=selected.anchor_id.to_numpy())
            oof = oof.sort_values(KEYS).reset_index(drop=True)
            if not oof[KEYS].equals(aligned_baseline[KEYS]):
                raise ValueError("complete-policy evaluation alignment failed")
            if [r["past_fit_cases"] for r in detail["router_receipts"]] != [0, 49, 128]:
                raise ValueError("arm-specific router chronology failed")
            oof = oof.merge(metadata, on=["anchor_id", "station", "fold"], validate="many_to_one")
            evaluation = oof.copy()
            evaluation["final_prediction"] = aligned_baseline.final_prediction.to_numpy()
            predicted = oof.final_prediction.to_numpy()
            details[arm] = {"vs_legacy_control": slices(evaluation, predicted), "paired_bootstrap_vs_legacy": bootstrap(evaluation, predicted, config["bootstrap"]), "fixed_structure_diagnostics": detail}
            rebuilt[arm] = oof
            oof.to_parquet(OUT / f"oof_{arm}.parquet", index=False)
        effect = rebuilt["episode_weight"].copy()
        effect["final_prediction"] = rebuilt["control"].final_prediction.to_numpy()
        weighted = rebuilt["episode_weight"].final_prediction.to_numpy()
        episode_effect = {"paired_effect": slices(effect, weighted), "paired_bootstrap": bootstrap(effect, weighted, config["bootstrap"])}
        values = {"legacy_control": float(np.sqrt(np.mean(np.square(baseline.target_hs-baseline.final_prediction)))), **{arm: details[arm]["vs_legacy_control"]["rmse_m"] for arm in config["arms"]}}
        winner = min(values, key=lambda arm: (values[arm], ["legacy_control", "control", "episode_weight"].index(arm)))
        selected.to_parquet(OUT / "validation_keys.parquet", index=False)
        result = {"experiment_id": NAME, "status": "COMPLETE", "decision": "INTERNAL_CANDIDATE_AVAILABLE" if winner != "legacy_control" else "NO_INTERNAL_GAIN", "winner": winner, "metric_rmse_m": values, "arms": details, "episode_weight_effect_vs_seed_matched_control": episode_effect, "integrity": integrity, "fit_count": {"historical_backbone": 18, "historical_router": 4, "full_backbone": 0, "full_router": 0, "synthetic_smoke": 2}, "fit_receipts": fit_receipts, "pair_receipts": pair_receipts, "elapsed_seconds": time.perf_counter()-started, "manifest": manifest, "official_access": {"test": 0, "sample": 0, "hidden": 0, "submission_csv": 0, "uploads": 0}, "public_score_fitting": False, "a_b_combined": False, "surface": "reused_historical_development_not_fresh_confirmation", "expected_official_score": None, "gpu_released_at_terminal": True, "artifact_sha256": {path.name: sha(path) for path in OUT.glob("*.parquet")}}
        save(REPORT / "result.json", result)
        save(OUT / "terminal_result.json", {"status": "COMPLETE", "decision": result["decision"], "winner": winner, "result_sha256": sha(REPORT / "result.json"), "fit_count": result["fit_count"]})
        save(OUT / "progress.json", {"stage": "COMPLETE", "completed_fits": 18, "pid": os.getpid()}, progress=True)
        return {"status": "COMPLETE", "winner": winner, "rmse_m": values, "fit_count": result["fit_count"], "elapsed_seconds": result["elapsed_seconds"]}
    except Exception as error:
        save(OUT / "terminal_result.json", {"status": "TERMINAL_TECHNICAL_FAILURE", "exception_type": type(error).__name__, "message": str(error), "historical_fits_completed": len(fit_receipts), "automatic_retry": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "synthetic-smoke", "execute"))
    parser.add_argument("--gpu-approved", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    if args.mode != "preflight" and not args.gpu_approved:
        raise PermissionError("root GPU resource release must be recorded before training")
    install_guard(source, config)
    if args.mode == "preflight":
        *_, proof = prepare(config, source)
        save(REPORT / "preflight.json", proof)
        print(json.dumps({"status": "PREFLIGHT_PASS", "features": proof["feature_count"], "cases": proof["cases"], "train_anchors": proof["train_anchor_count"], "fits": 0}))
    elif args.mode == "synthetic-smoke":
        print(json.dumps(synthetic_smoke(config)))
    else:
        print(json.dumps(execute(config, source)), flush=True)
