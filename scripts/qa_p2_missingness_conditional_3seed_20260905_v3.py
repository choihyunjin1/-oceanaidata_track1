"""Independent aggregate and lineage checks; no fitting or official input."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_p2_missingness_conditional_3seed_20260905_v3 as run  # noqa: E402


def audit():
    checks = []
    def check(name, condition):
        checks.append({"name": name, "pass": bool(condition)})
        if not condition:
            raise AssertionError(name)
    report, out = run.REPORT, run.OUT
    result = json.loads((report / "result.json").read_text(encoding="utf-8"))
    seal = json.loads(run.SEAL.read_text(encoding="utf-8"))
    lock = json.loads((out / "ATTEMPT_LOCK.json").read_text(encoding="utf-8"))
    terminal = json.loads((out / "terminal_result.json").read_text(encoding="utf-8"))
    replay = json.loads((report / "fresh-process-replay.json").read_text(encoding="utf-8"))
    fits = json.loads((out / "fit-receipts.json").read_text(encoding="utf-8"))
    check("new seal exact", seal["hashes"] == run.fingerprints())
    check("terminal result hash", terminal["result_sha256"] == run.base.file_hash(report / "result.json"))
    check("nine fits", len(fits) == result["new_historical_fits"] + result["new_full_fits"] == terminal["new_fits"] == 9)
    expected = {(s, f["id"]) for s in (20260902, 20260903) for f in run.previous.load_config()["folds"]} | {(s, "full") for s in run.read_config()["seeds"]}
    check("exact nine seed-fold contracts", {(f["seed"], f["fold"]) for f in fits} == expected)
    old_fits = json.loads((run.previous.ARTIFACT / "03_training/fit-receipts.json").read_text(encoding="utf-8"))
    for fit in fits:
        check(f"model hash {fit['fit_id']}", run.base.file_hash(ROOT / fit["path"]) == fit["sha256"])
        check(f"fixed recipe {fit['fit_id']}", fit["arm"] == "R" and fit["epochs"] == 60 and fit["state_reload_exact"])
        if fit["fold"] == "full":
            check(f"full rows {fit['fit_id']}", fit["training"]["original_rows"] == 166268)
        else:
            reference = next(f for f in old_fits if f["fit_id"] == f"R_20260901_{fit['fold']}")
            check(f"same fold training {fit['fit_id']}", fit["training"] == reference["training"])
    check("fit runtime sum", np.isclose(result["fit_runtime_seconds"], sum(f["runtime_seconds"] for f in fits), rtol=0, atol=1e-9))
    old_seal = json.loads((run.previous.REPORT / "preregistration-seal.json").read_text(encoding="utf-8"))["hashes"]
    check("original A training runner unchanged", old_seal["runner"] == run.base.file_hash(Path(run.previous.__file__)))
    check("original A training config unchanged", old_seal["config"] == run.base.file_hash(run.previous.CONFIG))
    for path, sha in old_seal["dependencies"].items():
        check(f"original A dependency unchanged {path}", run.base.file_hash(ROOT / path) == sha)
    full_c = json.loads((run.FULL_C / "train-result.json").read_text(encoding="utf-8"))
    for kind, path in (("runner", "scripts/run_p2_score_repair_deploy_20260905_v1.py"), ("config", "configs/experiments/p2_score_repair_deploy_20260905_v1.json"), ("research_runner", "scripts/run_p2_score_repair_20260905_v1.py"), ("research_config", "configs/experiments/p2_score_repair_20260905_v1.json")):
        check(f"original full C lineage {kind}", full_c[f"{kind}_sha256"] == run.base.file_hash(ROOT / path))
    check("fresh process different from training", replay["pid"] != lock["pid"])
    check("six model fresh process replay", replay["status"] == "PASS" and replay["full_models"] == 6 and replay["component_predictions_exact"] and replay["conditional_exact"])
    for name, path in (("prediction", out / "validation_predictions.npz"), ("fit_receipts", out / "fit-receipts.json"), ("replay_input", out / "replay_public_inputs.npz")):
        check(f"artifact {name} hash", result[f"{name}_sha256"] == run.base.file_hash(path))
    data = np.load(out / "validation_predictions.npz", allow_pickle=False)
    old = np.load(run.previous.ARTIFACT / "03_training/raw_oof.npz", allow_pickle=False)
    check("all 69850 historical keys", len(data["key"]) == len(np.unique(data["key"])) == 69850)
    for key in ("key", "truth", "fold"):
        check(f"exact old {key}", np.array_equal(data[key], old[key]))
    check("autumn primary 26273", int((data["fold"] == "2024_sep_oct").sum()) == 26273)
    for arm in ("C", "R"):
        mean = np.mean(np.stack([data[f"intact_{arm}_seed{s}"] for s in run.read_config()["seeds"]]), axis=0)
        check(f"three seed mean {arm}", np.array_equal(mean, data[f"intact_{arm}"]))
        check(f"old first seed exact {arm}", np.array_equal(data[f"intact_{arm}_seed20260901"], old[f"{arm}_seed20260901"]))
    check("C mean exact official recipe historical mean", np.array_equal(data["intact_C"], old["C_mean"]))
    natural = ~(data["natural_temp5_finite"] & data["natural_psal5_finite"])
    check("target-free OR natural trigger", np.array_equal(natural, data["intact_trigger"]))
    def metrics(prediction, mask):
        delta = prediction[mask] - data["truth"][mask]
        return {"n": int(mask.sum()), "sse": float(np.sum(delta * delta)), "rmse": float(np.sqrt(np.mean(delta * delta))), "bias": float(np.mean(delta))}
    def panel(name, panels, selected=None):
        trigger = data[f"{name}_trigger"]
        conditional = np.where(trigger, data[f"{name}_R"], data[f"{name}_C"])
        check(f"exact route {name}", np.array_equal(conditional, data[f"{name}_conditional"]))
        scopes = {"all_intact_keys": np.ones(69850, bool), "autumn_primary": data["fold"] == "2024_sep_oct", "trigger": trigger}
        if selected is not None:
            scopes.update(injected_episode=selected, outside_injected_episode=~selected)
        for scope, mask in scopes.items():
            check(f"denominator {name} {scope}", panels[scope]["n"] == int(mask.sum()))
            if not mask.any():
                continue
            for arm in ("C", "R", "conditional"):
                calc = metrics(data[f"{name}_{arm}"], mask)
                for key, value in calc.items():
                    check(f"metric {name} {scope} {arm} {key}", np.isclose(value, panels[scope]["metrics"][arm][key], rtol=1e-12, atol=1e-10))
            expected_delta = panels[scope]["metrics"]["conditional"]["rmse"] - panels[scope]["metrics"]["C"]["rmse"]
            check(f"delta {name} {scope}", expected_delta == panels[scope]["delta_conditional_minus_C_rmse_C"])
    panel("intact", result["intact"])
    for episode in result["episodes"]:
        name = episode["id"]
        selected, supported = data[f"{name}_selected"], data[f"{name}_supported"]
        check(f"episode retained all keys {name}", episode["all_keys"] == 69850 and len(selected) == 69850 and episode["evaluation_rows_deleted"] == 0)
        check(f"episode counts {name}", episode["injected_rows"] == int(selected.sum()) and episode["unsupported_rows"] == int((~supported).sum()))
        check(f"episode trigger {name}", np.array_equal(data[f"{name}_trigger"], natural | selected))
        changed = int((selected & (data["natural_temp5_finite"] | data["natural_psal5_finite"])).sum())
        check(f"actual new missingness {name}", changed == episode["newly_changed_public_availability_rows"])
        if name == "autumn_3d":
            check("known technical support blocked preserved", episode["status"] == "SUPPORT_BLOCKED" and episode["unsupported_rows"] == 4 and "metrics" not in episode)
            check("unsupported no fabricated predictions", f"{name}_conditional" not in data.files)
            continue
        check(f"scored scenario all supported {name}", supported.all() and episode["status"] == "COMPLETE")
        panel(name, episode["metrics"], selected)
        for arm in ("C", "R", "conditional"):
            check(f"outside invariance {name} {arm}", np.array_equal(data[f"{name}_{arm}"][~selected], data[f"intact_{arm}"][~selected]))
        if name.startswith("winter"):
            check(f"winter no-op flagged {name}", changed == 0)
    for group, values in result["additional_episode_metrics"].items():
        episodes = [e for e in result["episodes"] if not e["development"] and e["status"] == "COMPLETE" and (group == "all_new_episodes" or e["fold"] == group)]
        count = sum(int(data[f"{e['id']}_selected"].sum()) for e in episodes)
        for arm in ("C", "R", "conditional"):
            sse = sum(metrics(data[f"{e['id']}_{arm}"], data[f"{e['id']}_selected"])["sse"] for e in episodes)
            check(f"pooled sum {group} {arm}", values[arm]["n"] == count and np.isclose(sse, values[arm]["sse"], rtol=1e-12) and np.isclose(np.sqrt(sse/count), values[arm]["rmse"], rtol=1e-12))
    breakdown = {}
    for fold in np.unique(data["fold"]):
        breakdown[fold] = {}
        masks = {"temp5_only_missing": ~data["natural_temp5_finite"] & data["natural_psal5_finite"], "psal5_only_missing": data["natural_temp5_finite"] & ~data["natural_psal5_finite"], "both_missing": ~data["natural_temp5_finite"] & ~data["natural_psal5_finite"], "nontrigger": ~natural}
        for kind, mask in masks.items():
            mask = mask & (data["fold"] == fold)
            breakdown[fold][kind] = {"n": int(mask.sum()), "metrics": {arm: metrics(data[f"intact_{arm}"], mask) for arm in ("C", "R", "conditional")} if mask.any() else None}
    autumn = result["additional_episode_metrics"]["2024_sep_oct"]
    expected_info = result["intact"]["autumn_primary"]["delta_conditional_minus_C_rmse_C"] <= 0 and autumn["conditional"]["rmse"] < autumn["C"]["rmse"]
    check("information value rule independently recalculated", bool(expected_info) == result["information_value_positive"])
    for key in ("new_rule_fits", "official_access_rows", "csv_written", "upload", "evaluation_rows_deleted"):
        check(f"zero {key}", result[key] == 0)
    check("no CSV created", not list(out.rglob("*.csv")))
    check("all old zero-fit artifacts unchanged", run.base.file_hash(run.prior.REPORT / "result.json") == seal["hashes"]["files"][str((run.prior.REPORT / "result.json").relative_to(ROOT)).replace("\\", "/")])
    qa = {"status": "PASS", "experiment_id": run.ID, "checks": checks, "check_count": len(checks), "result_sha256": run.base.file_hash(report / "result.json"), "fresh_process_replay_sha256": run.base.file_hash(report / "fresh-process-replay.json"), "qa_script_sha256": run.base.file_hash(Path(__file__)), "natural_availability_breakdown": breakdown, "scientific_claim": "INFO_ONLY is distinct from numerical QA, reproducibility and guaranteed official gain", "access_evidence": "Runner audit hook source allowlist + immutable hashes + zero output receipts; no global OS-level read instrumentation claim"}
    run.save(report / "independent-qa.json", qa)
    print(json.dumps({"status": "PASS", "checks": len(checks), "result_sha256": qa["result_sha256"]}))


if __name__ == "__main__":
    audit()
