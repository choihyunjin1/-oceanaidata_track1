"""Independent arithmetic/hash/weight QA of completed P3-B; never refits models."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p3_wave.validation import expand_leads  # noqa: E402

NAME = "p3_episode_weight_20260905_v2"
OUT = ROOT / "artifacts" / NAME
REPORT = ROOT / "reports" / NAME / "result.json"
KEYS = ["anchor_id", "station", "lead_h", "fold"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(frame, column="final_prediction"):
    return float(np.sqrt(np.square(frame.target_hs.to_numpy()-frame[column].to_numpy()).mean()))


def independent_episode_weight_hashes(anchors, selected, source, config):
    """Rebuild episode groups with a different grouping expression and no runner import."""
    wave = pd.read_csv(source / "train_wave.csv", usecols=["station", "time", "hs"])
    wave["time"] = pd.to_datetime(wave.time, utc=True)
    pieces = []
    for _, group in wave.groupby("station", sort=True):
        group = group.sort_values("time").copy()
        high = group.hs.ge(1.5) & group.hs.notna()
        # Break at every background/missing timestamp or a gap. High runs stay intact
        # even when target-eligible anchors are absent inside the event.
        group["independent_group"] = ((~high) | group.time.diff().ne(pd.Timedelta(minutes=20))).cumsum()
        pieces.append(group.loc[high, ["station", "time", "independent_group"]])
    mapping = pd.concat(pieces).rename(columns={"time": "anchor_time"})
    joined = anchors.copy()
    joined["anchor_time"] = pd.to_datetime(joined.anchor_time, utc=True)
    joined = joined.merge(mapping, on=["station", "anchor_time"], validate="one_to_one")
    if len(joined) != len(anchors):
        raise ValueError("independent episode mapping lost anchors")
    table = joined.set_index("anchor_id")
    recipe = json.loads((ROOT / config["reference_config"]).read_text(encoding="utf-8"))
    hashes = {}
    for fold, start, _ in recipe["validation"]["windows"]:
        valid = table.loc[selected.loc[selected.fold.eq(fold), "anchor_id"]]
        held = set(zip(valid.station, valid.independent_group, strict=True))
        train = joined.loc[joined.anchor_time.lt(pd.Timestamp(start, tz="UTC")-pd.Timedelta(hours=78))].copy()
        train = train.loc[[(s, g) not in held for s, g in zip(train.station, train.independent_group, strict=True)]].sort_values("anchor_id")
        threshold = np.exp(-0.45*np.maximum(train.current_hs.to_numpy(float)-1.5, 0))
        threshold /= threshold.mean()
        sizes = train.groupby(["station", "independent_group"]).independent_group.transform("size").to_numpy(float)
        event = 1/np.sqrt(sizes)
        event /= event.mean()
        combined = threshold*event
        combined /= combined.mean()
        hashes[(fold, "control")] = hashlib.sha256(threshold.tobytes()).hexdigest()
        hashes[(fold, "episode_weight")] = hashlib.sha256(combined.tobytes()).hexdigest()
    return hashes


def fresh_process_model_replay(anchors, selected, cache):
    features = pd.read_parquet(cache / "train_features.parquet")
    columns = json.loads((cache / "feature_columns.json").read_text(encoding="utf-8"))["columns"]
    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    maximum = 0.0
    rows = 0
    models = 0
    for arm, seed_set in (("control", 1), ("episode_weight", 0), ("episode_weight", 1)):
        stored = pd.read_parquet(OUT / f"components_{arm}_{seed_set}.parquet").sort_values(KEYS).reset_index(drop=True)
        for fold in selected.fold.unique():
            ids = np.sort(selected.loc[selected.fold.eq(fold), "anchor_id"].to_numpy())
            x_valid, _, metadata = expand_leads(features, anchors, ids, columns)
            x_valid.station = x_valid.station.astype(str)
            x_valid.lead_h = x_valid.lead_h.astype(str)
            directory = OUT / "models" / arm / f"seed_set_{seed_set}" / fold
            single = CatBoostRegressor().load_model(directory / "single.cbm")
            single_delta = single.predict(x_valid, thread_count=2, task_type="CPU")
            replay = metadata[["anchor_id", "station", "lead_h"]].copy()
            replay["fold"] = fold
            replay["single_prediction"] = np.clip(metadata.current_hs.to_numpy()+single_delta, 0, 30)
            multi_x = feature_lookup.loc[ids, ["station", *columns]].reset_index(drop=True)
            multi_x.station = multi_x.station.astype(str)
            multi = CatBoostRegressor().load_model(directory / "multi.cbm")
            multi_delta = multi.predict(multi_x, thread_count=2, task_type="CPU")
            absolute = np.clip(anchor_lookup.loc[ids, "current_hs"].to_numpy()[:, None]+multi_delta, 0, 30)
            multi_frame = pd.DataFrame({"anchor_id": np.repeat(ids, 6), "lead_h": np.tile([3, 6, 9, 12, 18, 24], len(ids)), "multi_prediction": absolute.reshape(-1)})
            replay = replay.merge(multi_frame, on=["anchor_id", "lead_h"], validate="one_to_one").sort_values(KEYS).reset_index(drop=True)
            expected = stored.loc[stored.fold.eq(fold)].sort_values(KEYS).reset_index(drop=True)
            if not replay[KEYS].equals(expected[KEYS]):
                raise ValueError("fresh-process model replay key mismatch")
            maximum = max(maximum, float(np.max(np.abs(replay[["single_prediction", "multi_prediction"]].to_numpy()-expected[["single_prediction", "multi_prediction"]].to_numpy()))))
            rows += len(replay)
            models += 2
    return {"process_id": os.getpid(), "models_reloaded": models, "six_lead_rows": rows, "scalar_component_predictions": 2*rows, "maximum_absolute_difference_m": maximum, "new_fits": 0, "prediction_device": "CPU", "cpu_threads": 2, "official_inputs": 0}


def run():
    result = json.loads(REPORT.read_text(encoding="utf-8"))
    config_path = ROOT / "configs/experiments" / f"{NAME}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cache = ROOT / config["cache"]
    baseline = pd.read_parquet(cache / "oof.parquet").sort_values(KEYS).reset_index(drop=True)
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    selected = pd.read_parquet(OUT / "validation_keys.parquet")
    checks = {"terminal_complete": result["status"] == "COMPLETE"}
    checks["source_and_cache_hashes_after_run"] = all(sha(ROOT / path) == value for path, value in config["inputs"].items())
    source = Path(os.environ["P3_DATA_DIR"])
    checks["distributed_source_hashes_after_run"] = all(sha(source / path) == value for path, value in config["source_files"].items())
    weight_hashes = independent_episode_weight_hashes(anchors, selected, source, config)
    checks["episode_weights_independently_recomputed"] = all(row["weight_sha256"] == weight_hashes[(row["fold"], row["arm"])] for row in result["pair_receipts"])
    checks["frozen_runner_and_config_unchanged"] = sha(config_path) == result["manifest"]["config_sha256"] and sha(ROOT / "scripts" / f"run_{NAME}.py") == result["manifest"]["runner_sha256"]
    checks["output_hashes_match"] = all(sha(OUT / path) == value for path, value in result["artifact_sha256"].items())
    receipts = result["fit_receipts"]
    checks["eighteen_backbone_four_router_fits"] = result["fit_count"] == {"historical_backbone": 18, "historical_router": 4, "full_backbone": 0, "full_router": 0, "synthetic_smoke": 2} and len(receipts) == 18
    checks["control_six_candidate_twelve_fits"] = sum(r["arm"] == "control" for r in receipts) == 6 and sum(r["arm"] == "episode_weight" for r in receipts) == 12
    folds = ["2024_h2_storm", "winter_transition", "2025_h1"]
    checks["exact_seed_schedule"] = all(r["seed"] == config["fold_seed_sets"][r["seed_set"]][folds.index(r["fold"])] for r in receipts)
    checks["model_hashes_and_reload_zero"] = all(sha(OUT / "models" / row["arm"] / f"seed_set_{row['seed_set']}" / row["fold"] / f"{row['component']}.cbm") == row["model_sha256"] and row["reload_max_abs"] == 0.0 for row in receipts)
    replay = fresh_process_model_replay(anchors, selected, cache)
    checks["fresh_process_model_replay"] = replay["models_reloaded"] == 18 and replay["maximum_absolute_difference_m"] == 0.0 and os.getpid() != result["manifest"]["pid"]
    checks["unique_181_cases_all_six_leads"] = len(selected) == 181 and not selected.anchor_id.duplicated().any()
    expected_keys = pd.read_parquet(cache / "validation_keys.parquet").sort_values("anchor_id").reset_index(drop=True)
    checks["validation_keys_unchanged"] = selected.sort_values("anchor_id").reset_index(drop=True).equals(expected_keys)
    recipe = json.loads((ROOT / config["reference_config"]).read_text(encoding="utf-8"))
    metadata = selected.copy()
    metadata["anchor_time"] = pd.to_datetime(metadata.anchor_time, utc=True)
    chronology = []
    previous = []
    for fold, start, _ in recipe["validation"]["windows"]:
        current = metadata.loc[metadata.fold.eq(fold)]
        past = metadata.loc[metadata.fold.isin(previous)]
        row = {"fold": fold, "cases": len(current), "past_cases": len(past)}
        if previous:
            ready_margin = (pd.Timestamp(start, tz="UTC")-(past.anchor_time.max()+pd.Timedelta(hours=24))).total_seconds()/3600
            gaps = [(current.loc[current.station.eq(station), "anchor_time"].min()-past.loc[past.station.eq(station), "anchor_time"].max()).total_seconds()/3600 for station in current.station.unique() if past.station.eq(station).any()]
            overlap = set(zip(current.station, current.episode_id, strict=True)) & set(zip(past.station, past.episode_id, strict=True))
            row.update(global_past_target_ready_before_fold_start_h=ready_margin, minimum_same_station_anchor_gap_h=min(gaps), same_station_episode_overlap=len(overlap))
            checks[f"{fold}_actual_meta_target_availability"] = ready_margin >= 0 and min(gaps) >= 78 and not overlap
        chronology.append(row)
        previous.append(fold)
    evaluated = {}
    for arm in config["arms"]:
        output = pd.read_parquet(OUT / f"oof_{arm}.parquet").sort_values(KEYS).reset_index(drop=True)
        evaluated[arm] = output
        checks[f"{arm}_exact_keys_and_truth"] = len(output) == 1086 and not output.duplicated(KEYS).any() and output[KEYS].equals(baseline[KEYS]) and np.array_equal(output.target_hs, baseline.target_hs)
        checks[f"{arm}_rmse_recomputed"] = abs(score(output)-result["metric_rmse_m"][arm]) < 1e-14
        first = baseline if arm == "control" else pd.read_parquet(OUT / "components_episode_weight_0.parquet").sort_values(KEYS).reset_index(drop=True)
        second = pd.read_parquet(OUT / f"components_{arm}_1.parquet").sort_values(KEYS).reset_index(drop=True)
        checks[f"{arm}_component_means_before_router"] = all(np.array_equal(output[c], (first[c].to_numpy()+second[c].to_numpy())/2) for c in ("single_prediction", "multi_prediction"))
        values = output[["single_prediction", "multi_prediction", "persistence"]].to_numpy()
        weights = output[["weight_single", "weight_multi", "weight_persistence"]].to_numpy()
        routed = np.sum(values*weights, axis=1)
        prediction = routed.copy()
        active = output.lead_h.isin([12, 18, 24]).to_numpy()
        prediction[active] = 0.8*routed[active]+0.2*output.persistence.to_numpy()[active]
        checks[f"{arm}_router_and_shrink_arithmetic"] = bool(np.max(np.abs(prediction-output.final_prediction.to_numpy())) < 1e-12 and np.all(weights >= 0) and np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-12))
        checks[f"{arm}_short_noop"] = np.array_equal(output.loc[~active, "final_prediction"], output.loc[~active, "equal_prediction"])
        checks[f"{arm}_past_only_router_cases"] = [row["past_fit_cases"] for row in result["arms"][arm]["fixed_structure_diagnostics"]["router_receipts"]] == [0, 49, 128] and all(not row["current_fold_target_used_for_router"] for row in result["arms"][arm]["fixed_structure_diagnostics"]["router_receipts"])
        checks[f"{arm}_all_hard_integrity_contracts"] = all(result["arms"][arm]["fixed_structure_diagnostics"]["gate"]["contract_checks"].values())
        checks[f"{arm}_finite_range"] = bool(np.isfinite(output.final_prediction).all() and output.final_prediction.between(0, 30).all())
    episode_delta = score(evaluated["episode_weight"])-score(evaluated["control"])
    checks["paired_episode_effect_recomputed"] = abs(episode_delta-result["episode_weight_effect_vs_seed_matched_control"]["paired_effect"]["delta_rmse_m"]) < 1e-14
    scores = {"legacy_control": score(baseline), **{arm: score(frame) for arm, frame in evaluated.items()}}
    winner = min(scores, key=lambda arm: (scores[arm], ["legacy_control", "control", "episode_weight"].index(arm)))
    checks["winner_recomputed"] = winner == result["winner"]
    checks["zero_official_public_fit_combination"] = not any(result["official_access"].values()) and result["public_score_fitting"] is False and result["a_b_combined"] is False
    lookup = anchors.set_index("anchor_id")
    checks["distributed_train_truth_alignment"] = all(np.array_equal(baseline.loc[baseline.lead_h.eq(lead), "target_hs"], lookup.loc[baseline.loc[baseline.lead_h.eq(lead), "anchor_id"], f"target_{lead}"]) for lead in [3, 6, 9, 12, 18, 24])
    result_qa = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": {name: bool(value) for name, value in checks.items()}, "checks_passed": int(sum(checks.values())), "checks_total": len(checks), "metric_rmse_m": scores, "paired_episode_delta_rmse_m": episode_delta, "meta_chronology_recalculated_from_actual_validation_metadata": chronology, "fresh_process_replay": replay, "result_sha256": sha(REPORT), "new_fits_during_qa": 0, "official_access_during_qa": 0}
    assert all(checks.values()), [name for name, value in checks.items() if not value]
    return result_qa


if __name__ == "__main__":
    print(json.dumps(run()))
