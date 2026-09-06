"""New-path orchestration of an immutable validated numeric P3 policy.

Only two full backbones plus one full-OOF router are trained. Historical fits
are not repeated; model-only replay is not a whole-cold regeneration claim.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import p3_forward_candidate_materialize_20260906_v1 as core  # noqa: E402
import run_p3_numeric_lead_forward_gpu_20260906_v2 as experiment  # noqa: E402

NAME = "p3_numeric_candidate_20260906_v1"
CONFIG = ROOT / "configs/experiments" / f"{NAME}.json"
OUT = ROOT / "artifacts" / NAME
REPORT = ROOT / "reports" / NAME
TEST = ROOT / "tests" / f"test_{NAME}.py"
CODE_QA = REPORT / "code-qa-v2.json"


def historical_checks_pass(qa):
    checks = qa.get("checks")
    return (
        qa.get("status") == "PASS"
        and qa.get("checks_count") == 149
        and qa.get("failed_checks") == []
        and isinstance(checks, list)
        and len(checks) == 149
        and all(isinstance(item, dict) and item.get("pass") is True for item in checks)
    )


def verify_code_qa():
    qa = core.read(CODE_QA)
    if (
        qa["status"] != "PASS"
        or qa["ruff"] != "PASS"
        or qa["focused_pytest_count"] < 19
        or qa["driver_sha256"] != core.sha(__file__)
        or qa["config_sha256"] != core.sha(CONFIG)
        or qa["tests_sha256"] != core.sha(TEST)
    ):
        raise ValueError("actual focused code QA or source hash missing")


def source_closure():
    """Pin loaded project code only, not environment/site-package payloads."""
    paths = {Path(__file__).resolve(), Path(core.__file__).resolve()}
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            path = Path(path).resolve()
            if path.suffix == ".py" and any(
                folder in path.parents for folder in (ROOT / "scripts", ROOT / "src")
            ):
                paths.add(path)
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): core.sha(path) for path in sorted(paths)
    }


def verify_contract(config):
    if config["experiment_id"] != NAME or config["historical_experiment"] != experiment.NAME:
        raise ValueError("candidate identity differs")
    if config["budget"] != {
        "new_backbone_fits": 2,
        "new_router_fits": 1,
        "cpu_threads": 2,
        "gpu_device": "0",
        "wall_seconds": 3600,
    }:
        raise ValueError("unapproved resource or fit budget")
    if config["policy"] != {
        "feature_count": 591,
        "numeric_lead": True,
        "hmax_removed": False,
        "full_seed": 20260817,
        "router_alpha": 10.0,
        "router_temperature": 2.0,
        "router_strength": 0.5,
        "long_persistence": 0.2,
        "active_leads": [12, 18, 24],
    }:
        raise ValueError("numeric policy changed or hmax policy mixed in")
    for name, expected in config["pins"].items():
        if core.sha(ROOT / name) != expected:
            raise ValueError("candidate input pin differs: " + name)
    cfg = core.read(experiment.CONFIG)
    if "removal" in cfg:
        raise ValueError("hmax removal is not this candidate")
    return cfg


def fingerprint(config):
    return {
        "driver_sha256": core.sha(__file__),
        "config_sha256": core.sha(CONFIG),
        "tests_sha256": core.sha(TEST),
        "source_closure_sha256": source_closure(),
        "historical_pins": config["pins"],
        "code_qa_sha256": core.sha(CODE_QA),
    }


def assert_preflight(config):
    receipt = core.read(REPORT / "preflight.json")
    if receipt["status"] != "PASS" or receipt["fingerprint"] != fingerprint(config):
        raise ValueError("scoped driver/config/dependency preflight changed")
    return receipt


def preflight(config, cfg):
    if OUT.exists():
        raise FileExistsError("candidate output already exists; no restart")
    verify_code_qa()
    accepted = core.verify_inputs(experiment, cfg)
    result = core.read(experiment.REPORT / "result.json")
    qa = core.read(experiment.REPORT / "independent-qa.json")
    replay = core.read(experiment.REPORT / "fresh-process-replay.json")
    recipe = core.read(ROOT / cfg["recipe"])
    single = experiment.parameters(recipe, "single", 20260817)
    multi = experiment.parameters(recipe, "multi", 20260817)
    checks = {
        "historical_149_checks_pass": historical_checks_pass(qa),
        "historical_15_plus_8": result["fit_count"] == 15 and result["router_fit_count"] == 8,
        "historical_103602_replay": replay["paired_rows"] == 103602
        and replay["max_abs_prediction_error_m"] == 0
        and replay["fresh_pid"] != replay["training_pid"],
        "single_cpu2_700": single["task_type"] == "CPU"
        and single["thread_count"] == 2
        and single["iterations"] == 700,
        "multi_gpu0_1200_plain": multi["task_type"] == "GPU"
        and multi["devices"] == "0"
        and multi["iterations"] == 1200
        and multi["boosting_type"] == "Plain",
        "no_hmax_removal": "removal" not in cfg,
    }
    failed = [name for name, ok in checks.items() if not ok]
    core.save(
        REPORT / "preflight.json",
        {
            "status": "FAIL" if failed else "PASS",
            "checks": checks,
            "failed_checks": failed,
            "fingerprint": fingerprint(config),
            "accepted_same_cycle": accepted,
            "pid": os.getpid(),
            "created_utc": datetime.now(UTC).isoformat(),
            "official_rows": 0,
            "new_fits": 0,
            "synthetic_test_receipt": CODE_QA.name,
        },
    )
    if failed:
        raise ValueError("preflight failed")


def fit(config, cfg):
    receipt = assert_preflight(config)
    started = datetime.now(UTC).isoformat()

    # Hard limit affects only this newly owned process. No other process is touched.
    def timeout():
        if OUT.exists() and not (OUT / "BUDGET_STOP.json").exists():
            core.save(
                OUT / "BUDGET_STOP.json",
                {
                    "status": "RESOURCE_STOP",
                    "pid": os.getpid(),
                    "wall_seconds": 3600,
                    "automatic_restart": False,
                },
            )
        os._exit(124)

    timer = threading.Timer(3600, timeout)
    timer.daemon = True
    timer.start()
    try:
        core.fit(experiment, cfg, OUT)
        core.save(
            OUT / "scoped-seal.json",
            {
                "fingerprint": receipt["fingerprint"],
                "started_utc": started,
                "preflight_sha256": core.sha(REPORT / "preflight.json"),
            },
        )
        for relative in receipt["fingerprint"]["source_closure_sha256"]:
            destination = OUT / "02_code" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        core.save(
            OUT / "04_logs/full-fit-terminal.json",
            {
                "status": "COMPLETE",
                "pid": os.getpid(),
                "gpu_work_complete": True,
                "new_backbone_fits": 2,
                "new_router_fits": 1,
                "training_result_sha256": core.sha(OUT / "04_logs/training-result.json"),
                "official_rows": 0,
                "whole_cold": False,
            },
        )
    finally:
        timer.cancel()


def numeric_qa(config, cfg):
    core.training_qa(experiment, cfg, OUT)
    row, models, columns = core.load(experiment, cfg, OUT)
    expected_single = ["station", "lead_h", "current_hs_for_residual", *columns]
    seal = core.read(OUT / "scoped-seal.json")
    copied = {
        name: core.sha(OUT / "02_code" / name)
        for name in seal["fingerprint"]["source_closure_sha256"]
    }
    checks = {
        "scoped_seal_exact": seal["fingerprint"] == fingerprint(config),
        "copied_sources_exact": copied == seal["fingerprint"]["source_closure_sha256"],
        "numeric_single_native_cat_indices_station_only": models[0].get_cat_feature_indices()
        == [0],
        "single_feature_order_exact": models[0].feature_names_ == expected_single,
        "multi_feature_order_exact": models[1].feature_names_ == ["station", *columns],
        "hmax_remains_both_backbones": "hmax_current" in columns
        and all("hmax_current" in model.feature_names_ for model in models[:2]),
        "hmax_remains_router": "hmax_current" in models[2].columns,
        "model_files_exactly_three": sorted(row["model_sha256"])
        == ["multi.cbm", "router.joblib", "single.cbm"],
        "full_router_own_historical_oof_only": row["same_cycle_prior_oof_rows"] == 103602,
        "not_training_probe_generalization_claim": row["replay_rows"] == 768,
    }
    original = core.read(OUT / "04_logs/training-independent-qa.json")
    checks.update({"materializer_" + name: ok for name, ok in original["checks"].items()})
    failed = [name for name, ok in checks.items() if not ok]
    core.save(
        OUT / "04_logs/numeric-independent-qa.json",
        {
            "status": "FAIL" if failed else "PASS",
            "checks": checks,
            "checks_count": len(checks),
            "failed_checks": failed,
            "training_result_sha256": core.sha(OUT / "04_logs/training-result.json"),
            "fresh_replay_sha256": core.sha(OUT / "04_logs/fresh-process-replay.json"),
            "scoped_seal_sha256": core.sha(OUT / "scoped-seal.json"),
            "historical_oof_qa_sha256": config["pins"][
                str((experiment.REPORT / "independent-qa.json").relative_to(ROOT)).replace(
                    "\\", "/"
                )
            ],
            "saved_model_replay_not_scratch_twice": True,
            "official_rows": 0,
        },
    )
    if failed:
        raise ValueError("numeric independent QA failed: " + ",".join(failed))


def require_numeric_qa():
    qa = core.read(OUT / "04_logs/numeric-independent-qa.json")
    if qa["status"] != "PASS" or qa["failed_checks"] or not all(qa["checks"].values()):
        raise ValueError("numeric QA required before official input")
    for name, field in (
        ("training-result.json", "training_result_sha256"),
        ("fresh-process-replay.json", "fresh_replay_sha256"),
    ):
        if core.sha(OUT / "04_logs" / name) != qa[field]:
            raise ValueError("numeric QA link differs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("preflight", "check", "fit", "replay", "qa", "infer", "verify-answer"),
        required=True,
    )
    args = parser.parse_args()
    config = core.read(CONFIG)
    cfg = verify_contract(config)
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    core.guard(source, OUT, cfg, experiment, official=args.stage in {"infer", "verify-answer"})
    begin = time.perf_counter()
    try:
        if args.stage == "preflight":
            preflight(config, cfg)
        else:
            assert_preflight(config)
            if args.stage == "check":
                verify_code_qa()
                core.verify_inputs(experiment, cfg)
            elif args.stage == "fit":
                fit(config, cfg)
            elif args.stage == "replay":
                core.replay(experiment, cfg, OUT)
            elif args.stage == "qa":
                numeric_qa(config, cfg)
            else:
                require_numeric_qa()
                core.inference(
                    experiment, cfg, OUT, source, verify_answer=args.stage == "verify-answer"
                )
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "status": "COMPLETE",
                    "pid": os.getpid(),
                    "seconds": time.perf_counter() - begin,
                }
            ),
            flush=True,
        )
    except Exception as exc:
        failure = REPORT / (args.stage + "-failure.json")
        if not failure.exists():
            core.save(
                failure,
                {
                    "status": "TERMINAL_TECHNICAL_FAILURE",
                    "stage": args.stage,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "pid": os.getpid(),
                    "automatic_restart": False,
                },
            )
        raise


if __name__ == "__main__":
    main()
