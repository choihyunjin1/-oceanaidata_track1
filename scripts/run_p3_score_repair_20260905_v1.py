"""Source-only P3 weather block-missingness ablation; never reads official inputs.

--audit: provenance and identical historical OOF comparison, no model fits.
--execute: exactly-once 3 folds x 2 arms x 2 seeds, saved local models/OOF.
All paths are repository-relative or P3_DATA_DIR. No deployment action is implemented.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil

from p3_wave.corrected_repeated_forward import build_corrected_repeated_forward_folds
from p3_wave.data import LEADS
from p3_wave.models import compact_feature_columns, threshold_case_weights
from p3_wave.revin_patch import assign_storm_episodes_from_wave
from p3_wave.validation import expand_leads, metric_slices

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "p3_score_repair_20260905_v1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT}.json"
OUT = ROOT / "artifacts" / EXPERIMENT
REPORT = ROOT / "reports" / EXPERIMENT
KEYS = ["fold", "anchor_id", "station", "lead_h"]
WEATHER_PREFIXES = (
    "wspd_",
    "gust_",
    "airt_",
    "relh_",
    "caph_",
    "wdir_sin_",
    "wdir_cos_",
    "wind_wave_alignment_",
    "wind_input_proxy_",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temp.replace(path)


def now() -> str:
    return datetime.now(UTC).isoformat()


def official_access() -> dict:
    return dict.fromkeys(
        [
            "test_context_rows",
            "test_index_rows",
            "sample_rows",
            "hidden_rows",
            "submission_rows",
            "uploads",
        ],
        0,
    )


def install_read_boundary(source: Path) -> None:
    allowed_csv = {(source / name).resolve() for name in ("train_wave.csv", "train_atmos.csv")}

    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        name = path.name.lower()
        if name.endswith(".csv") and path not in allowed_csv:
            raise PermissionError("CSV outside immutable training allowlist")
        if name in {
            "test_context.parquet",
            "test_features.parquet",
            "test_index.csv",
            "sample_submission.csv",
        }:
            raise PermissionError("official input access forbidden")
        if "external_data" in path.parts or "hidden" in name:
            raise PermissionError("external or hidden input forbidden")

    sys.addaudithook(audit)


def weather_columns(columns) -> list[str]:
    return [c for c in columns if c.startswith(WEATHER_PREFIXES)]


def weather_observed(frame: pd.DataFrame) -> np.ndarray:
    valid = [c for c in weather_columns(frame.columns) if "_valid_" in c]
    if not valid:
        raise ValueError("weather support columns missing")
    return frame[valid].gt(0).any(axis=1).to_numpy()


def mask_weather(frame: pd.DataFrame) -> pd.DataFrame:
    masked = frame.copy()
    columns = weather_columns(frame.columns)
    valid = [c for c in columns if "_valid_" in c]
    values = [c for c in columns if c not in valid]
    masked.loc[:, values] = np.nan
    masked.loc[:, valid] = 0.0
    return masked


def augment_weather(frame, target, weight):
    observed = weather_observed(frame)
    original_weight = np.asarray(weight, dtype=float).copy()
    original_weight[observed] *= 0.5
    result = pd.concat([frame, mask_weather(frame.loc[observed])], ignore_index=True)
    targets = np.concatenate([target, np.asarray(target)[observed]])
    weights = np.concatenate([original_weight, np.asarray(weight)[observed] * 0.5])
    source_rows = np.concatenate([np.arange(len(frame)), np.flatnonzero(observed)])
    sums = np.bincount(source_rows, weights=weights, minlength=len(frame))
    if not np.allclose(sums, weight, rtol=0, atol=1e-12):
        raise AssertionError("augmentation changed original case weight")
    return (
        result,
        targets,
        weights,
        {"observed_rows": int(observed.sum()), "expanded_rows": len(result)},
    )


def aligned(reference: pd.DataFrame, other: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if reference.duplicated(KEYS).any() or other.duplicated(KEYS).any():
        raise ValueError("duplicate OOF keys")
    left = reference[KEYS + ["target_hs"]].copy()
    left["_ref_order"] = np.arange(len(left))
    joined = left.merge(
        other[KEYS + ["target_hs"] + columns],
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_other"),
        sort=False,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("OOF population differs")
    joined = joined.sort_values("_ref_order").reset_index(drop=True).drop(columns="_ref_order")
    if not np.allclose(joined.target_hs, joined.target_hs_other, rtol=0, atol=1e-12):
        raise ValueError("OOF truth differs")
    if not np.isfinite(joined[columns].to_numpy()).all():
        raise ValueError("nonfinite OOF predictions")
    return joined


def sliced(frame: pd.DataFrame, prediction) -> dict:
    result = metric_slices(frame, np.asarray(prediction))
    result["by_fold"] = {
        str(f): float(
            np.sqrt(np.mean((np.asarray(prediction)[idx] - frame.loc[idx, "target_hs"]) ** 2))
        )
        for f, idx in frame.groupby("fold", observed=True).groups.items()
    }
    result["mean_signed_error_m"] = float(
        np.mean(np.asarray(prediction) - frame.target_hs.to_numpy())
    )
    return result


def preflight(config, source):
    verified = {}
    for name, expected in config["source_files"].items():
        path = source / name
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"immutable source hash differs: {name}")
        verified[f"source/{name}"] = actual
    for relative, expected in config["pinned_files"].items():
        actual = digest(ROOT / relative)
        if actual != expected:
            raise ValueError(f"pinned clean dependency hash differs: {relative}")
        verified[relative] = actual
    features = pd.read_parquet(ROOT / "artifacts/p3/features_all20_v1/train_features.parquet")
    anchors = pd.read_parquet(ROOT / "artifacts/p3/features_all20_v1/train_anchors.parquet")
    if len(features) != 24360 or not features[["anchor_id", "station"]].equals(
        anchors[["anchor_id", "station"]]
    ):
        raise ValueError("train cache alignment changed")
    columns = compact_feature_columns([c for c in features if c not in ("anchor_id", "station")])
    if len(columns) != config["features"]["count"]:
        raise ValueError("feature count changed")
    wave = pd.read_csv(source / "train_wave.csv")
    wave["time"] = pd.to_datetime(wave.time, utc=True)
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    folds, selected, split = build_corrected_repeated_forward_folds(
        anchors,
        windows=config["windows"],
        gap_hours=78,
        footprint_hours=72,
    )
    reference = pd.read_parquet(
        ROOT / "artifacts/p3_corrected_repeated_forward_catboost_v2/oof.parquet"
    )
    keys = pd.read_parquet(
        ROOT / "artifacts/p3_corrected_repeated_forward_catboost_v2/validation_keys.parquet"
    )
    key_columns = ["fold", "anchor_id", "station", "episode_id"]
    if (
        not selected[key_columns]
        .sort_values(key_columns)
        .reset_index(drop=True)
        .equals(keys[key_columns].sort_values(key_columns).reset_index(drop=True))
    ):
        raise ValueError("selected validation cases differ from historical reference")
    expected = []
    for fold in folds:
        _, _, meta = expand_leads(features, anchors, fold.validation_ids, [])
        meta["fold"] = fold.name
        expected.append(meta)
    reference = (
        aligned(
            pd.concat(expected, ignore_index=True),
            reference,
            [
                "single_prediction",
                "multi_prediction",
                "equal_prediction",
                "routed_prediction",
                "final_prediction",
                "persistence",
            ],
        )
        .drop(columns=["_merge", "target_hs_other"])
        .reset_index(drop=True)
    )
    if len(reference) != 1086 or split["unique_station_episode_count"] != 181:
        raise ValueError("historical surface contract changed")
    result = {
        "experiment_id": EXPERIMENT,
        "created_utc": now(),
        "status": "PREFLIGHT_PASS",
        "fit_count": 0,
        "verified_sha256": verified,
        "split_audit": split,
        "config_sha256": digest(CONFIG),
        "runner_sha256": digest(Path(__file__)),
        "reference_provenance": "pinned distributed train cache; pinned source-only single/multi residual OOF, prequential router and fixed 0.2 persistence shrink; no Public-fitted coefficient",
        "raw_oof_limit": "historical component arrays already clipped; raw preclip values unavailable, not reconstructed",
        "official_access": official_access(),
        "previously_seen_exploratory_surface": True,
        "zero_fit_metrics": {},
        "zero_fit_complementarity": {},
    }
    for name in [
        "single_prediction",
        "multi_prediction",
        "equal_prediction",
        "routed_prediction",
        "final_prediction",
        "persistence",
    ]:
        result["zero_fit_metrics"][name] = sliced(reference, reference[name])
    optional = config["optional_tabpfn_oof"]
    path = ROOT / optional["path"]
    if path.exists() and digest(path) == optional["sha256"]:
        tab = aligned(
            reference,
            pd.read_parquet(path),
            ["tabpfn_prediction", "candidate_prediction", "clean_fallback_prediction"],
        )
        if not np.allclose(
            tab.clean_fallback_prediction, reference.final_prediction, rtol=0, atol=1e-12
        ):
            raise ValueError("TabPFN fallback is not the exact clean reference")
        result["verified_sha256"][optional["path"]] = optional["sha256"]
        for name in ("tabpfn_prediction", "candidate_prediction"):
            result["zero_fit_metrics"][name] = sliced(reference, tab[name])
        six = reference.final_prediction.to_numpy().copy()
        use = reference.lead_h.eq(6).to_numpy()
        six[use] = 0.75 * six[use] + 0.25 * tab.tabpfn_prediction.to_numpy()[use]
        reference["tabpfn25_6h_only"] = six
        result["zero_fit_complementarity"]["tabpfn25_6h_only"] = sliced(reference, six)
    else:
        result["tabpfn_status"] = "MISSING_OR_HASH_UNVERIFIED_NOT_USED"
    for label, col in [
        ("single25_all_leads", "single_prediction"),
        ("multi25_all_leads", "multi_prediction"),
    ]:
        prediction = 0.75 * reference.final_prediction + 0.25 * reference[col]
        reference[label] = prediction
        result["zero_fit_complementarity"][label] = sliced(reference, prediction)
    result["weather_feature_columns"] = weather_columns(columns)
    result["training_weather_observed_anchor_count"] = int(
        weather_observed(features[columns]).sum()
    )
    save_json(REPORT / "zero-fit-audit.json", result)
    return features, anchors, columns, folds, reference, result


def run(config, source):
    started = time.perf_counter()
    features, anchors, columns, folds, reference, audit = preflight(config, source)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "ATTEMPT_LOCK.json"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "pid": os.getpid(),
                "created_utc": now(),
                "config_sha256": digest(CONFIG),
                "runner_sha256": digest(Path(__file__)),
            },
            stream,
        )
    fits = []
    oof_blocks = []
    process = psutil.Process()
    peak_rss = 0.0
    wall_cap = config["resources"]["wall_cap_seconds"]

    def resource_guard(_env=None):
        nonlocal peak_rss
        rss = process.memory_info().rss / 2**30
        peak_rss = max(peak_rss, rss)
        if rss > config["resources"]["rss_cap_gib"]:
            raise MemoryError("P3 RSS cap exceeded")
        if time.perf_counter() - started > wall_cap:
            raise TimeoutError("P3 screening wall cap exceeded; do not retry")

    try:
        for fold in folds:
            train, target, train_meta = expand_leads(features, anchors, fold.train_ids, columns)
            valid, _, valid_meta = expand_leads(features, anchors, fold.validation_ids, columns)
            base_weight = threshold_case_weights(train_meta.current_hs.to_numpy())
            valid_meta["fold"] = fold.name
            valid_meta["weather_observed"] = weather_observed(valid)
            for seed in config["seeds"]:
                for arm in config["arms"]:
                    resource_guard()
                    if arm == "weather_blockmask":
                        matrix, y, weight, aug = augment_weather(train, target, base_weight)
                    else:
                        matrix, y, weight = train.copy(), target, base_weight
                        aug = {
                            "observed_rows": int(weather_observed(train).sum()),
                            "expanded_rows": len(train),
                        }
                    for name in ("station", "lead_h"):
                        categories = (
                            ["G-ORS", "I-ORS", "S-ORS"] if name == "station" else list(LEADS)
                        )
                        matrix[name] = pd.Categorical(matrix[name], categories=categories)
                        valid[name] = pd.Categorical(valid[name], categories=categories)
                    progress = {
                        "status": "RUNNING",
                        "pid": os.getpid(),
                        "fold": fold.name,
                        "arm": arm,
                        "seed": seed,
                        "completed_fits": len(fits),
                        "maximum_fits": 12,
                        "elapsed_seconds": time.perf_counter() - started,
                        "threads": 2,
                    }
                    save_json(OUT / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                    t0 = time.perf_counter()
                    model = lgb.LGBMRegressor(**config["model"], random_state=seed)
                    model.fit(
                        matrix,
                        y,
                        sample_weight=weight,
                        categorical_feature=["station", "lead_h"],
                        callbacks=[resource_guard],
                    )
                    raw = valid_meta.current_hs.to_numpy() + model.predict(valid)
                    if not np.isfinite(raw).all():
                        raise ValueError("nonfinite model prediction")
                    prediction = np.clip(raw, 0, 30)
                    block = valid_meta.copy()
                    block["arm"], block["seed"] = arm, seed
                    block["raw_prediction"], block["prediction"] = raw, prediction
                    oof_blocks.append(block)
                    model_path = OUT / "models" / f"{fold.name}_{arm}_{seed}.txt"
                    model_path.parent.mkdir(exist_ok=True)
                    model.booster_.save_model(str(model_path))
                    restored = lgb.Booster(model_file=str(model_path))
                    if not np.allclose(
                        model.predict(valid),
                        restored.predict(valid, num_threads=2),
                        rtol=0,
                        atol=1e-12,
                    ):
                        raise ValueError("saved model separate load prediction differs")
                    receipt = {
                        **progress,
                        "status": "FIT_COMPLETE",
                        "fit_number": len(fits) + 1,
                        "fit_seconds": time.perf_counter() - t0,
                        "train_anchor_count": len(fold.train_ids),
                        "train_rows": len(matrix),
                        "validation_cases": len(fold.validation_ids),
                        "augmentation": aug,
                        "model_sha256": digest(model_path),
                        "model_path": model_path.relative_to(ROOT).as_posix(),
                        "reload_prediction_pass": True,
                        "rss_gib": process.memory_info().rss / 2**30,
                    }
                    fits.append(receipt)
                    save_json(OUT / "fit-receipts.json", {"fits": fits})
                    print(json.dumps(receipt), flush=True)
                    del model, restored, matrix
                    gc.collect()
            del train, target, valid, train_meta
            gc.collect()
        oof = pd.concat(oof_blocks, ignore_index=True)
        oof.to_parquet(OUT / "oof.parquet", index=False)
        result = {
            "experiment_id": EXPERIMENT,
            "status": "COMPLETE",
            "terminal": True,
            "fit_count": len(fits),
            "runtime_seconds": time.perf_counter() - started,
            "peak_rss_gib": peak_rss,
            "config_sha256": digest(CONFIG),
            "runner_sha256": digest(Path(__file__)),
            "oof_sha256": digest(OUT / "oof.parquet"),
            "official_access": official_access(),
            "metrics": {},
            "seed_metrics": {},
            "split_audit": audit["split_audit"],
            "fresh_independent_confirmation": False,
            "submission_ready": False,
            "deployment_authorized": False,
        }
        for arm in config["arms"]:
            for seed in config["seeds"]:
                one = oof.loc[oof.arm.eq(arm) & oof.seed.eq(seed)].reset_index(drop=True)
                result["seed_metrics"][f"{arm}_{seed}"] = sliced(one, one.prediction)
            means = (
                oof.loc[oof.arm.eq(arm)]
                .groupby(KEYS + ["target_hs"], as_index=False, sort=False)[
                    ["prediction", "raw_prediction"]
                ]
                .mean()
            )
            match = aligned(reference, means, ["prediction", "raw_prediction"])
            result["metrics"][f"{arm}_raw"] = sliced(reference, match.raw_prediction)
            result["metrics"][f"{arm}_standalone"] = sliced(reference, match.prediction)
            blend = 0.75 * reference.final_prediction + 0.25 * match.prediction
            result["metrics"][f"{arm}_fallback25"] = sliced(reference, blend)
        base = audit["zero_fit_metrics"]["final_prediction"]["rmse"]
        for value in result["metrics"].values():
            value["delta_vs_clean_reference_m"] = value["rmse"] - base
            value["conditional_public_point_scale_not_prediction"] = (base - value["rmse"]) * 15.871
        result["weather_ablation_delta_m"] = (
            result["metrics"]["weather_blockmask_standalone"]["rmse"]
            - result["metrics"]["control_standalone"]["rmse"]
        )
        all_candidates = {
            **audit["zero_fit_complementarity"],
            **{k: v for k, v in result["metrics"].items() if not k.endswith("_raw")},
        }
        best = min(all_candidates, key=lambda name: all_candidates[name]["rmse"])
        result["next_decision"] = {
            "lowest_historical_candidate": best,
            "rmse_m": all_candidates[best]["rmse"],
            "clean_reference_rmse_m": base,
            "interpretation": "Previously seen surface; requires clean full-train/save/separate-predict deployability before official materialization",
        }
        if best in reference:
            chosen = reference[best].to_numpy()
        else:
            chosen_arm = "weather_blockmask" if best.startswith("weather_blockmask") else "control"
            means = (
                oof.loc[oof.arm.eq(chosen_arm)]
                .groupby(KEYS + ["target_hs"], as_index=False, sort=False)[["prediction"]]
                .mean()
            )
            chosen = aligned(reference, means, ["prediction"]).prediction.to_numpy()
            if best.endswith("fallback25"):
                chosen = 0.75 * reference.final_prediction.to_numpy() + 0.25 * chosen
        episode_lookup = anchors.set_index("anchor_id").episode_id
        np.savez_compressed(
            OUT / "qa_oof.npz",
            key=reference[KEYS].astype(str).agg("|".join, axis=1).to_numpy(dtype=str),
            fold=reference.fold.to_numpy(dtype=str),
            station=reference.station.to_numpy(dtype=str),
            lead_h=reference.lead_h.to_numpy(dtype=int),
            episode_id=episode_lookup.loc[reference.anchor_id].to_numpy(dtype=int),
            truth=reference.target_hs.to_numpy(dtype=float),
            reference=reference.final_prediction.to_numpy(dtype=float),
            prediction=chosen,
        )
        result["qa_oof_sha256"] = digest(OUT / "qa_oof.npz")
        save_json(REPORT / "result.json", result)
        save_json(OUT / "terminal_result.json", result)
        save_json(
            OUT / "progress.json",
            {
                "status": "COMPLETE",
                "completed_fits": len(fits),
                "elapsed_seconds": result["runtime_seconds"],
            },
        )
        print(json.dumps(result), flush=True)
        return result
    except Exception as error:
        if oof_blocks:
            pd.concat(oof_blocks, ignore_index=True).to_parquet(
                OUT / "partial-oof.parquet", index=False
            )
        result = {
            "experiment_id": EXPERIMENT,
            "status": "INCOMPLETE_BUDGET"
            if isinstance(error, (TimeoutError, MemoryError))
            else "TECHNICAL_FAILURE",
            "terminal": True,
            "successful_fit_count": len(fits),
            "error_type": type(error).__name__,
            "error": str(error),
            "runtime_seconds": time.perf_counter() - started,
            "peak_rss_gib": peak_rss,
            "official_access": official_access(),
            "automatic_retry": False,
        }
        save_json(OUT / "terminal_result.json", result)
        save_json(REPORT / "result.json", result)
        raise


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audit", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    install_read_boundary(source)
    if args.audit:
        *_, result = preflight(config, source)
        print(
            json.dumps({"status": result["status"], "fit_count": 0, "cases": 181, "rows": 1086}),
            flush=True,
        )
    else:
        run(config, source)


if __name__ == "__main__":
    main()
