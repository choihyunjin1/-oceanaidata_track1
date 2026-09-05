"""Exactly nine R fits, frozen three-seed C/R route, internal evidence only."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_p2_missingness_conditional_validation_20260905_v3 as prior  # noqa: E402

previous, base = prior.previous, prior.base
ID = "p2_missingness_conditional_3seed_20260905_v3"
CONFIG = ROOT / "configs/experiments" / f"{ID}.json"
REPORT, OUT = ROOT / "reports" / ID, ROOT / "artifacts" / ID
SEAL = REPORT / "preregistration-seal.json"
FULL_C = ROOT / "artifacts/p2_score_repair_deploy_20260905_v1"
save = prior.save


def read_config():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["experiment_id"] == ID
    assert cfg["seeds"] == [20260901, 20260902, 20260903]
    assert cfg["rule"] == prior.read_config()["rule"]
    assert cfg["epochs"] == 60 and cfg["maximum_new_historical_fits"] == 6
    assert cfg["maximum_new_full_fits"] == 3 and cfg["maximum_new_rule_fits"] == 0
    assert cfg["component_aggregation"] == "arithmetic_3seed_mean_before_conditional_route"
    return cfg


def models_manifest():
    rows = prior.model_manifest(prior.read_config())
    for row in rows:
        row.update(seed=20260901, stage="historical")
    old = json.loads((ROOT / "reports/p2_score_repair_20260905_v1/result.json").read_text(encoding="utf-8"))
    for seed in (20260902, 20260903):
        for fold in previous.load_config()["folds"]:
            fit_id = f"v23_blockmask_{seed}_{fold['id']}"
            receipt = next(row for row in old["fit_receipts"] if row["fit_id"] == fit_id)
            rows.append({"arm": "C", "stage": "historical", "seed": seed, "fold": fold["id"], "fit_id": fit_id, "path": str((previous.OLD / f"{fit_id}.pt").relative_to(ROOT)), "sha256": receipt["model_sha256"]})
    full = json.loads((FULL_C / "train-result.json").read_text(encoding="utf-8"))
    recipe = previous.load_config()
    assert full["arm"] == "v23_blockmask" and full["fullfit_count"] == 3
    assert full["source_sha256"] == recipe["source_sha256"]
    assert full["training_start"] == recipe["train_start"] and full["training_stop"] == recipe["train_stop"]
    assert full["training"]["original_rows"] == 166268
    for receipt in full["fits"]:
        assert receipt["epochs"] == 60
        rows.append({"arm": "C", "stage": "full", "seed": receipt["seed"], "fold": "full", "fit_id": f"C_full_{receipt['seed']}", "path": str((FULL_C / receipt["file"]).relative_to(ROOT)), "sha256": receipt["sha256"]})
    assert len(rows) == 15
    for row in rows:
        assert base.file_hash(ROOT / row["path"]) == row["sha256"]
    return rows


def fingerprints():
    files = ["scripts/run_p2_missingness_conditional_validation_20260905_v3.py", "configs/experiments/p2_missingness_conditional_validation_20260905_v3.json", "reports/p2_missingness_conditional_validation_20260905_v3/result.json", "scripts/run_p2_score_repair_deploy_20260905_v1.py", "configs/experiments/p2_score_repair_deploy_20260905_v1.json", "artifacts/p2_score_repair_deploy_20260905_v1/train-result.json"]
    return {"runner": base.file_hash(Path(__file__)), "config": base.file_hash(CONFIG), "prior": prior.fingerprint(prior.read_config()), "files": {p: base.file_hash(ROOT / p) for p in files}, "models": models_manifest()}


def install_guard(sealed):
    source = Path(os.environ["P2_DATA_DIR"]).resolve() / "observations.csv"
    allowed_models = {(ROOT / item["path"]).resolve() for item in sealed["models"]}
    allowed_npz = (previous.ARTIFACT / "03_training/raw_oof.npz").resolve()

    def guard(event, args):
        if event == "socket.connect":
            raise PermissionError("network forbidden")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if "external_data" in path.parts or "hidden" in path.name.lower():
            raise PermissionError("external/hidden forbidden")
        if path.suffix.lower() == ".csv" and path != source:
            raise PermissionError("only released observations allowed")
        if path == source and isinstance(args[1], str) and any(c in args[1] for c in "wax+"):
            raise PermissionError("source immutable")
        if path.suffix.lower() == ".pt" and path not in allowed_models and OUT not in path.parents:
            raise PermissionError("unapproved checkpoint")
        if path.suffix.lower() == ".npz" and path != allowed_npz and OUT not in path.parents:
            raise PermissionError("unapproved array artifact")
    sys.addaudithook(guard)


def load_model(path):
    model = base.make_model("v23", 11)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def component_mean(predictions):
    if len(predictions) != 3 or not all(np.isfinite(p).all() for p in predictions):
        raise ValueError("three finite component predictions required")
    return np.mean(np.stack(predictions), axis=0)


def metrics_and_episodes(query, truth, fold, models, cfg, old):
    seeds = cfg["seeds"]
    arrays = {"key": (query.time.astype(str) + "|" + query.layer.astype(str)).to_numpy(str), "truth": truth, "fold": fold, "layer": query.layer.to_numpy(), "time": query.time.to_numpy(str), "natural_temp5_finite": np.isfinite(query.temp_5.to_numpy()), "natural_psal5_finite": np.isfinite(query.psal_5.to_numpy())}
    intact = {}
    for arm in ("C", "R"):
        components = []
        for seed in seeds:
            prediction = np.full(len(truth), np.nan)
            for spec in previous.load_config()["folds"]:
                selected = fold == spec["id"]
                prediction[selected] = previous.predict_absolute(models[(arm, seed, spec["id"])], query.loc[selected].reset_index(drop=True))
            if arm == "C" or seed == 20260901:
                assert np.array_equal(prediction, old[f"{arm}_seed{seed}"])
            components.append(prediction)
            arrays[f"intact_{arm}_seed{seed}"] = prediction
        intact[arm] = component_mean(components)
    assert np.array_equal(intact["C"], old["C_mean"])
    trigger = prior.route_trigger(query)
    intact["conditional"] = prior.conditional(intact["C"], intact["R"], trigger)
    arrays["intact_trigger"] = trigger
    arrays.update({f"intact_{arm}": value for arm, value in intact.items()})
    episodes = []
    for episode in prior.read_config()["episodes"]:
        altered, selected = prior.episode_frame(query, episode)
        supported = np.isfinite(altered.baseline.to_numpy()) & (altered.public_temp_count.to_numpy() >= 2)
        current = prior.route_trigger(altered)
        name = episode["id"]
        arrays.update({f"{name}_selected": selected, f"{name}_trigger": current, f"{name}_supported": supported})
        changed_availability = selected & (np.isfinite(query.temp_5.to_numpy()) | np.isfinite(query.psal_5.to_numpy()))
        summary = {**episode, "all_keys": len(truth), "injected_rows": int(selected.sum()), "newly_changed_public_availability_rows": int(changed_availability.sum()), "supported_rows": int(supported.sum()), "unsupported_rows": int((~supported).sum()), "evaluation_rows_deleted": 0}
        if not supported.all():
            summary["status"] = "SUPPORT_BLOCKED"
            episodes.append(summary)
            continue
        selected_fold = fold == episode["fold"]
        predictions = {}
        for arm in ("C", "R"):
            prediction = intact[arm].copy()
            prediction[selected_fold] = component_mean([previous.predict_absolute(models[(arm, seed, episode["fold"])], altered.loc[selected_fold].reset_index(drop=True)) for seed in seeds])
            predictions[arm] = prediction
        predictions["conditional"] = prior.conditional(predictions["C"], predictions["R"], current)
        assert all(np.array_equal(pred[~selected], intact[arm][~selected]) for arm, pred in predictions.items())
        arrays.update({f"{name}_{arm}": pred for arm, pred in predictions.items()})
        summary.update(status="COMPLETE", metrics=prior.panel_metrics(truth, predictions, fold, current, selected))
        episodes.append(summary)
    aggregated = {}
    for group in ("2024_sep_oct", "2025_jul_aug", "2025_nov_dec", "all_new_episodes"):
        selected = [e for e in episodes if not e["development"] and e["status"] == "COMPLETE" and (group == "all_new_episodes" or e["fold"] == group)]
        n = sum(e["metrics"]["injected_episode"]["n"] for e in selected)
        values = {arm: {"n": n, "sse": sum(e["metrics"]["injected_episode"]["metrics"][arm]["sse"] for e in selected)} for arm in intact}
        for value in values.values():
            value["rmse"] = float(np.sqrt(value["sse"] / n)) if n else None
        aggregated[group] = values
    return arrays, {"intact": prior.panel_metrics(truth, intact, fold, trigger), "episodes": episodes, "additional_episode_metrics": aggregated}


def full_predict(model, payload):
    normalized = base.predict_model(model, payload["tokens"], payload["mask"], payload["context"])
    return payload["baseline"] + normalized * payload["scale"]


def execute():
    cfg, recipe = read_config(), previous.load_config()
    sealed = json.loads(SEAL.read_text(encoding="utf-8"))
    assert sealed["hashes"] == fingerprints()
    assert torch.cuda.is_available()
    if OUT.exists() or (REPORT / "result.json").exists():
        raise RuntimeError("exactly once output exists")
    install_guard(sealed["hashes"])
    OUT.mkdir(parents=True)
    save(OUT / "ATTEMPT_LOCK.json", {"pid": os.getpid(), "seal": sealed, "maximum_new_fits": 9})
    started, fits = time.monotonic(), []
    def progress(stage, **extra):
        base.atomic_json(OUT / "progress.json", {"status": "RUNNING", "pid": os.getpid(), "stage": stage, "new_fits": len(fits), "runtime_seconds": time.monotonic() - started, **extra})
    try:
        torch.set_num_threads(1)
        progress("load_released_source")
        frame, labels = previous.load_data(recipe)
        assert len(frame) == 166268
        fold_ids = np.full(len(frame), "", dtype="U20")
        folds = []
        for spec in recipe["folds"]:
            train, valid = base.restoration_masks(frame.time, spec, recipe["purge_days"])
            assert not np.any(train & valid)
            folds.append((spec, train, valid))
            fold_ids[valid] = spec["id"]
        models = {}
        for item in sealed["hashes"]["models"]:
            models[(item["arm"], item["seed"], item["fold"])] = load_model(ROOT / item["path"])
        def train_one(data, receipt, seed, fold):
            if len(fits) >= 9:
                raise RuntimeError("fit cap")
            fit_id = f"R_{seed}_{fold}"
            progress("training", fit_id=fit_id, epoch=0)
            model, fit = previous.fit_model(data, "R", seed, recipe, lambda epoch: progress("training", fit_id=fit_id, epoch=epoch))
            path = OUT / f"{fit_id}.pt"
            torch.save(model.state_dict(), path)
            replay = load_model(path)
            assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), replay.state_dict().values(), strict=True))
            fit.update(fit_id=fit_id, fold=fold, stage="full" if fold == "full" else "historical", path=str(path.relative_to(ROOT)), sha256=base.file_hash(path), training=receipt, state_reload_exact=True)
            fits.append(fit)
            base.atomic_json(OUT / "fit-receipts.json", fits)
            models[("R", seed, fold)] = model
        for spec, train, _ in folds:
            data, receipt = previous.make_training_arrays(frame.loc[train].reset_index(drop=True), labels[train], recipe)
            for seed in cfg["seeds"][1:]:
                train_one(data, receipt, seed, spec["id"])
            del data
        progress("full_training")
        data, receipt = previous.make_training_arrays(frame, labels, recipe)
        for seed in cfg["seeds"]:
            train_one(data, receipt, seed, "full")
        del data
        assert len(fits) == 9
        progress("historical_internal_test")
        selected = fold_ids != ""
        query, truth, fold = frame.loc[selected].reset_index(drop=True), labels[selected], fold_ids[selected]
        old = np.load(previous.ARTIFACT / "03_training/raw_oof.npz", allow_pickle=False)
        assert np.array_equal(old["truth"], truth) and np.array_equal(old["fold"], fold)
        assert np.array_equal(old["key"], (query.time.astype(str) + "|" + query.layer.astype(str)).to_numpy(str))
        arrays, result = metrics_and_episodes(query, truth, fold, models, cfg, old)
        np.savez_compressed(OUT / "validation_predictions.npz", **arrays)
        sample = frame.iloc[np.linspace(0, len(frame)-1, 128, dtype=int)].reset_index(drop=True)
        tokens, mask, context = base.arrays(sample)
        payload = {"tokens": tokens, "mask": mask, "context": context, "baseline": sample.baseline.to_numpy(float), "scale": base.compute_profile_scale(sample), "trigger": prior.route_trigger(sample)}
        for arm in ("C", "R"):
            for seed in cfg["seeds"]:
                payload[f"{arm}_{seed}"] = full_predict(models[(arm, seed, "full")], payload)
        payload["conditional"] = prior.conditional(component_mean([payload[f"C_{s}"] for s in cfg["seeds"]]), component_mean([payload[f"R_{s}"] for s in cfg["seeds"]]), payload["trigger"])
        np.savez_compressed(OUT / "replay_public_inputs.npz", **payload)
        autumn = result["additional_episode_metrics"]["2024_sep_oct"]
        information = result["intact"]["autumn_primary"]["delta_conditional_minus_C_rmse_C"] <= 0 and autumn["conditional"]["rmse"] < autumn["C"]["rmse"]
        result.update(experiment_id=ID, status="INTERNAL_VALIDATION_COMPLETE", decision="INFO_ONLY_CANDIDATE" if information else "NO_INTERNAL_SUPPORT_FOR_CONDITIONAL", information_value_positive=bool(information), new_historical_fits=6, new_full_fits=3, new_rule_fits=0, reused_historical_models=12, reused_full_models=3, official_access_rows=0, csv_written=0, upload=0, evaluation_rows_deleted=0, runner_sha256=base.file_hash(Path(__file__)), config_sha256=base.file_hash(CONFIG), seal_sha256=base.file_hash(SEAL), prediction_sha256=base.file_hash(OUT / "validation_predictions.npz"), replay_input_sha256=base.file_hash(OUT / "replay_public_inputs.npz"), fit_receipts_sha256=base.file_hash(OUT / "fit-receipts.json"), runtime_seconds=time.monotonic()-started, fit_runtime_seconds=sum(f["runtime_seconds"] for f in fits), cpu_threads=1, loader_workers=0, source_sha256=recipe["source_sha256"], historical_rows=len(truth), primary_rows=int((fold == recipe["primary_fold"]).sum()), expected_official_points=None, limitations=["Repeated historical labels; no fresh confirmation", "Three winter interventions are no-op missingness because layer5 already absent", "Autumn3d scenario remains SUPPORT_BLOCKED (four unsupported rows); no deletion or score", "No official inputs or predictions; cannot infer actual official changed rows", "Fresh-process replay is a separate subsequent receipt; INFO_ONLY never promotion or expected point guarantee"])
        save(REPORT / "result.json", result)
        save(OUT / "terminal_result.json", {"status": "COMPLETE", "new_fits": 9, "result_sha256": base.file_hash(REPORT / "result.json")})
        base.atomic_json(OUT / "progress.json", {"status": "COMPLETE", "new_fits": 9, "runtime_seconds": time.monotonic()-started})
        print(json.dumps({"status": "COMPLETE", "new_fits": 9, "decision": result["decision"], "runtime_seconds": result["runtime_seconds"]}), flush=True)
    except Exception as error:
        save(OUT / "terminal_result.json", {"status": "TERMINAL_TECHNICAL_FAILURE", "exception_type": type(error).__name__, "message": str(error), "completed_new_fits": len(fits), "automatic_restart": False})
        raise


def replay():
    cfg = read_config()
    sealed = json.loads(SEAL.read_text(encoding="utf-8"))
    assert sealed["hashes"] == fingerprints()
    install_guard(sealed["hashes"])
    torch.set_num_threads(1)
    result = json.loads((REPORT / "result.json").read_text(encoding="utf-8"))
    assert base.file_hash(OUT / "replay_public_inputs.npz") == result["replay_input_sha256"]
    payload = np.load(OUT / "replay_public_inputs.npz", allow_pickle=False)
    fits = json.loads((OUT / "fit-receipts.json").read_text(encoding="utf-8"))
    manifest = [r for r in sealed["hashes"]["models"] if r["stage"] == "full"] + [r for r in fits if r["stage"] == "full"]
    predictions = {}
    for item in manifest:
        arm, seed = item["arm"], item["seed"]
        assert base.file_hash(ROOT / item["path"]) == item["sha256"]
        values = full_predict(load_model(ROOT / item["path"]), payload)
        assert np.array_equal(values, payload[f"{arm}_{seed}"])
        predictions[f"{arm}_{seed}"] = values
    c = component_mean([predictions[f"C_{s}"] for s in cfg["seeds"]])
    r = component_mean([predictions[f"R_{s}"] for s in cfg["seeds"]])
    assert np.array_equal(prior.conditional(c, r, payload["trigger"]), payload["conditional"])
    save(REPORT / "fresh-process-replay.json", {"status": "PASS", "pid": os.getpid(), "full_models": 6, "replay_rows": 128, "component_predictions_exact": True, "conditional_exact": True, "official_access_rows": 0, "new_fits": 0, "csv_written": 0, "upload": 0, "result_sha256": base.file_hash(REPORT / "result.json"), "replay_input_sha256": result["replay_input_sha256"], "manifest": manifest})
    print(json.dumps({"fresh_process_replay": "PASS", "full_models": 6, "replay_rows": 128}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seal", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.seal:
            read_config()
            save(SEAL, {"experiment_id": ID, "sealed_utc": pd.Timestamp.now(tz="UTC").isoformat(), "hashes": fingerprints(), "fit_cap": 9, "historical_fits": 6, "full_fits": 3})
            print(json.dumps({"status": "SEALED", "runner_sha256": base.file_hash(Path(__file__)), "config_sha256": base.file_hash(CONFIG)}))
        elif args.execute:
            execute()
        else:
            replay()
