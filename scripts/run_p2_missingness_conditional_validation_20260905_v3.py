"""Zero-fit frozen C/R missingness route validation. No official input or output."""

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
import run_p2_objective_alignment_20260905_v2 as previous  # noqa: E402

base = previous.base
ID = "p2_missingness_conditional_validation_20260905_v3"
CONFIG = ROOT / "configs/experiments" / f"{ID}.json"
REPORT = ROOT / "reports" / ID
OUT = ROOT / "artifacts" / ID
SEAL = REPORT / "preregistration-seal.json"


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)


def read_config():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["experiment_id"] == ID and cfg["seed"] == 20260901
    assert cfg["rule"] == "R_if_nonfinite_temp_5_OR_nonfinite_psal_5_else_C"
    assert cfg["maximum_new_backbone_fits"] == cfg["maximum_rule_fits"] == 0
    assert len(cfg["episodes"]) == 10 and sum(not e["development"] for e in cfg["episodes"]) == 9
    return cfg


def model_manifest(cfg):
    old = json.loads((ROOT / "reports/p2_score_repair_20260905_v1/result.json").read_text(encoding="utf-8"))
    other = json.loads((previous.ARTIFACT / "03_training/fit-receipts.json").read_text(encoding="utf-8"))
    rows = []
    for arm, receipts, directory, prefix in (
        ("C", old["fit_receipts"], previous.OLD, "v23_blockmask"),
        ("R", other, previous.ARTIFACT / "04_models", "R"),
    ):
        for fold in previous.load_config()["folds"]:
            fit_id = f"{prefix}_{cfg['seed']}_{fold['id']}"
            receipt = next(row for row in receipts if row["fit_id"] == fit_id)
            path = directory / f"{fit_id}.pt"
            if base.file_hash(path) != receipt["model_sha256"]:
                raise ValueError("old checkpoint hash mismatch")
            rows.append({"arm": arm, "fold": fold["id"], "fit_id": fit_id, "path": str(path.relative_to(ROOT)), "sha256": receipt["model_sha256"]})
    return rows


def fingerprint(cfg):
    files = ["scripts/run_p2_objective_alignment_20260905_v2.py", "configs/experiments/p2_objective_alignment_20260905_v2.json", *previous.DEPENDENCIES,
             "reports/p2_objective_alignment_20260905_v2/result.json", "reports/p2_score_repair_20260905_v1/result.json",
             "artifacts/p2_objective_alignment_20260905_v2/03_training/raw_oof.npz", "artifacts/p2_objective_alignment_20260905_v2/03_training/fit-receipts.json"]
    return {"runner": base.file_hash(Path(__file__)), "config": base.file_hash(CONFIG), "dependencies": {p: base.file_hash(ROOT / p) for p in files}, "models": model_manifest(cfg)}


def route_trigger(frame):
    """Only public layer 5 availability, never own target availability or values."""
    return ~np.isfinite(frame[["temp_5", "psal_5"]].to_numpy(float)).all(axis=1)


def episode_frame(frame, episode):
    times = pd.to_datetime(frame.time, utc=True)
    selected = np.asarray((times >= base.utc(episode["start"])) & (times < base.utc(episode["stop"])))
    changed = frame.copy()
    changed.loc[selected, ["temp_5", "psal_5"]] = np.nan
    changed = base.refresh_public(changed)
    return changed, selected


def conditional(c, r, trigger):
    if c.shape != r.shape or c.shape != trigger.shape:
        raise ValueError("conditional policy keys not aligned")
    return np.where(trigger, r, c)


def panel_metrics(truth, predictions, fold, trigger, selected=None):
    scopes = {"all_intact_keys": np.ones(len(truth), dtype=bool), "autumn_primary": fold == "2024_sep_oct", "trigger": trigger}
    if selected is not None:
        scopes["injected_episode"] = selected
        scopes["outside_injected_episode"] = ~selected
    result = {}
    for name, mask in scopes.items():
        if not mask.any():
            result[name] = {"n": 0, "metrics": None}
            continue
        values = {arm: base.metrics(truth[mask], pred[mask]) for arm, pred in predictions.items()}
        result[name] = {"n": int(mask.sum()), "metrics": values, "delta_conditional_minus_C_rmse_C": values["conditional"]["rmse"] - values["C"]["rmse"]}
    return result


def install_guard(cfg, sealed):
    source = (Path(os.environ["P2_DATA_DIR"]).resolve() / "observations.csv")
    models = {(ROOT / item["path"]).resolve() for item in sealed["models"]}
    old_npz = (previous.ARTIFACT / "03_training/raw_oof.npz").resolve()

    def guard(event, args):
        if event == "socket.connect":
            raise PermissionError("network forbidden")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if "external_data" in path.parts or "hidden" in path.name.lower():
            raise PermissionError("external or hidden path forbidden")
        if path.suffix.lower() == ".csv" and path != source:
            raise PermissionError("only distributed observations source allowed")
        if path == source and isinstance(args[1], str) and any(c in args[1] for c in "wax+"):
            raise PermissionError("source immutable")
        if path.suffix.lower() == ".pt" and path not in models:
            raise PermissionError("unapproved model")
        if path.suffix.lower() == ".npz" and path != old_npz and OUT not in path.parents:
            raise PermissionError("unapproved predictions")
    sys.addaudithook(guard)


def execute():
    cfg = read_config()
    sealed = json.loads(SEAL.read_text(encoding="utf-8"))
    if sealed["hashes"] != fingerprint(cfg):
        raise ValueError("sealed hash drift")
    if OUT.exists() or (REPORT / "result.json").exists():
        raise RuntimeError("exactly-once output exists")
    install_guard(cfg, sealed["hashes"])
    OUT.mkdir(parents=True)
    save(OUT / "ATTEMPT_LOCK.json", {"pid": os.getpid(), "new_fits": 0, "seal": sealed})
    started = time.monotonic()
    try:
        torch.set_num_threads(1)
        frame, labels = previous.load_data(previous.load_config())
        fold = np.full(len(frame), "", dtype="U20")
        for spec in previous.load_config()["folds"]:
            _, selected = base.restoration_masks(frame.time, spec, 7)
            fold[selected] = spec["id"]
        used = fold != ""
        query, truth, fold = frame.loc[used].reset_index(drop=True), labels[used], fold[used]
        key = (query.time.astype(str) + "|" + query.layer.astype(str)).to_numpy(str)
        old = np.load(previous.ARTIFACT / "03_training/raw_oof.npz", allow_pickle=False)
        for name, value in (("key", key), ("truth", truth), ("fold", fold)):
            if not np.array_equal(old[name], value):
                raise ValueError("historical evaluation alignment changed")
        assert len(truth) == 69850 and int((fold == "2024_sep_oct").sum()) == 26273
        models, intact = {}, {"C": np.full(len(truth), np.nan), "R": np.full(len(truth), np.nan)}
        for item in sealed["hashes"]["models"]:
            model = base.make_model("v23", 11)
            model.load_state_dict(torch.load(ROOT / item["path"], map_location="cpu", weights_only=True))
            model.eval()
            models[(item["arm"], item["fold"])] = model
            rows = fold == item["fold"]
            intact[item["arm"]][rows] = previous.predict_absolute(model, query.loc[rows].reset_index(drop=True))
        for arm in ("C", "R"):
            if not np.array_equal(intact[arm], old[f"{arm}_seed{cfg['seed']}"]):
                raise ValueError("frozen historical first-seed replay not exact")
        trigger = route_trigger(query)
        intact["conditional"] = conditional(intact["C"], intact["R"], trigger)
        outputs = {"key": key, "truth": truth, "fold": fold, "layer": query.layer.to_numpy(), "time": query.time.to_numpy(str), "intact_trigger": trigger,
                   **{f"intact_{arm}": pred for arm, pred in intact.items()}}
        result_intact = panel_metrics(truth, intact, fold, trigger)
        episodes = []
        for episode in cfg["episodes"]:
            altered, selected = episode_frame(query, episode)
            supported = np.isfinite(altered.baseline.to_numpy()) & (altered.public_temp_count.to_numpy() >= 2)
            name = episode["id"]
            current_trigger = route_trigger(altered)
            outputs[f"{name}_selected"], outputs[f"{name}_trigger"] = selected, current_trigger
            outputs[f"{name}_supported"] = supported
            summary = {**episode, "all_keys": len(truth), "injected_rows": int(selected.sum()), "supported_rows": int(supported.sum()), "unsupported_rows": int((~supported).sum()), "trigger_rows": int(current_trigger.sum()), "evaluation_rows_deleted": 0}
            if not supported.all():
                summary["status"] = "SUPPORT_BLOCKED"
                episodes.append(summary)
                continue
            predictions = {arm: intact[arm].copy() for arm in ("C", "R")}
            rows = fold == episode["fold"]
            for arm in ("C", "R"):
                predictions[arm][rows] = previous.predict_absolute(models[(arm, episode["fold"])], altered.loc[rows].reset_index(drop=True))
            predictions["conditional"] = conditional(predictions["C"], predictions["R"], current_trigger)
            if not all(np.array_equal(pred[~selected], intact[arm][~selected]) for arm, pred in predictions.items()):
                raise ValueError("outside-episode zero-dependency invariance failed")
            if not np.array_equal(predictions["conditional"][~current_trigger], predictions["C"][~current_trigger]):
                raise ValueError("conditional route altered supported C rows")
            if episode["development"]:
                for arm in ("C", "R"):
                    if not np.array_equal(predictions[arm][rows], old[f"stress_{arm}_seed{cfg['seed']}"][rows]):
                        raise ValueError("old development scenario replay differs")
            outputs.update({f"{name}_{arm}": pred for arm, pred in predictions.items()})
            summary.update(status="COMPLETE", metrics=panel_metrics(truth, predictions, fold, current_trigger, selected))
            episodes.append(summary)
            print(json.dumps({"episode_complete": name, "new_fits": 0, "rows": len(truth)}), flush=True)
        additional = [e for e in episodes if not e["development"] and e["status"] == "COMPLETE"]
        aggregated = {}
        for group in ("2024_sep_oct", "2025_jul_aug", "2025_nov_dec", "all_new_episodes"):
            selected_episodes = [e for e in additional if group == "all_new_episodes" or e["fold"] == group]
            n = sum(e["metrics"]["injected_episode"]["n"] for e in selected_episodes)
            values = {arm: {"n": n, "sse": sum(e["metrics"]["injected_episode"]["metrics"][arm]["sse"] for e in selected_episodes)} for arm in intact}
            for metrics in values.values():
                metrics["rmse"] = float(np.sqrt(metrics["sse"] / n)) if n else None
            aggregated[group] = values
        intact_delta = result_intact["autumn_primary"]["delta_conditional_minus_C_rmse_C"]
        autumn = aggregated["2024_sep_oct"]
        support_pass = all(e["status"] == "COMPLETE" for e in episodes)
        information = support_pass and intact_delta <= 0 and autumn["conditional"]["rmse"] < autumn["C"]["rmse"]
        prediction_path = OUT / "validation_predictions.npz"
        np.savez_compressed(prediction_path, **outputs)
        result = {"experiment_id": ID, "status": "INFORMATION_VALUE_INTERNAL_ONLY" if information else "RESEARCH_ONLY_NOT_READY", "intact": result_intact, "episodes": episodes, "additional_episode_metrics": aggregated,
                  "frozen_model_reuse_exact": True, "reused_models": 6, "new_backbone_fits": 0, "new_rule_fits": 0, "fulltrain_fits": 0, "official_access_rows": 0, "csv_written": 0, "upload": 0, "evaluation_rows_deleted": 0,
                  "prediction_sha256": base.file_hash(prediction_path), "hashes": sealed["hashes"], "runtime_seconds": time.monotonic()-started, "cpu_threads": 1, "gpu_used": False,
                  "official_control_rmse_C": 0.455143, "official_control_points": 27.622418, "expected_official_points": None,
                  "limitations": ["Single seed C/R; historical labels repeatedly exposed", "Rule motivated by old 14d development result", "Additional synthetic episodes are new manipulations, not fresh labels or independent confirmation", "Natural missingness may change intact predictions; only target-free public availability routes", "Unsupported masked scenarios retain keys and are not scored; no silent row deletion", "Official materialization and fulltrain not authorized in this validation"]}
        save(REPORT / "result.json", result)
        save(OUT / "terminal_result.json", {"status": "COMPLETE", "decision": result["status"], "new_fits": 0, "result_sha256": base.file_hash(REPORT / "result.json")})
        print(json.dumps({"status": result["status"], "new_fits": 0, "intact_primary_delta_C": intact_delta, "new_autumn_metrics": autumn, "runtime_seconds": result["runtime_seconds"]}))
    except Exception as error:
        save(OUT / "terminal_result.json", {"status": "TERMINAL_TECHNICAL_FAILURE", "exception_type": type(error).__name__, "message": str(error), "new_fits": 0, "automatic_restart": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.seal and args.execute:
        raise ValueError("seal must precede execution")
    if args.seal:
        cfg = read_config()
        save(SEAL, {"experiment_id": ID, "sealed_utc": pd.Timestamp.now(tz="UTC").isoformat(), "hashes": fingerprint(cfg), "fit_cap": 0})
        print(json.dumps({"status": "SEALED", "runner_sha256": base.file_hash(Path(__file__)), "config_sha256": base.file_hash(CONFIG)}))
    elif args.execute:
        with threadpool_limits(limits=1):
            execute()
