"""Budget-only v2: reuse two inner models from the preserved resource stop."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p3_forward_candidate_materialize_20260906_v1 as core  # noqa: E402
import run_p3_numeric_lead_forward_gpu_20260906_v2 as e  # noqa: E402
from p3_forward_independent_statistics_20260906_v2 import independent_bootstrap  # noqa: E402

np, pd, joblib = core.np, core.pd, core.joblib
NAME = "p3_numeric_training_curve_budget90_20260906_v2"
CONFIG = ROOT / "configs/experiments" / (NAME + ".json")
OUT, REPORT = ROOT / "artifacts" / NAME, ROOT / "reports" / NAME
OLD_OUT = e.OUT
REUSE_OUT = ROOT / "artifacts/p3_numeric_training_curve_20260906_v1"
REUSE_REPORT = ROOT / "reports/p3_numeric_training_curve_20260906_v1"


def now():
    return datetime.now(UTC).isoformat()


def key_sha(frame, columns):
    return hashlib.sha256(frame[columns].to_json(orient="split", index=False).encode()).hexdigest()


def prediction_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


@contextmanager
def owned_output():
    """Only this process's helper output variable changes; no source is edited."""
    previous = e.OUT
    e.OUT = OUT
    try:
        yield
    finally:
        e.OUT = previous


def validate_config(cfg):
    expected = (
        ("baseline", 700, 0.035, 1200, 0.03),
        ("compact", 525, 0.035 / 0.75, 900, 0.04),
        ("gentle", 1050, 0.035 / 1.5, 1800, 0.02),
    )
    actual = [
        (
            r["id"],
            r["single_iterations"],
            r["single_learning_rate"],
            r["multi_iterations"],
            r["multi_learning_rate"],
        )
        for r in cfg["recipes"]
    ]
    if actual != list(expected) or cfg["experiment_id"] != NAME:
        raise ValueError("frozen recipes or identity differ")
    if (
        cfg["budget"]["max_total_historical_fits_including_reused"] != 20
        or cfg["budget"]["wall_seconds"] != 5400
    ):
        raise ValueError("fit/time budget differs")
    for key, expected_value in {
        "max_new_backbones": 14,
        "max_new_routers": 4,
        "max_new_fits": 18,
        "cpu_threads": 2,
        "gpu_device": "0",
    }.items():
        if cfg["budget"][key] != expected_value:
            raise ValueError("resource amendment contract differs: " + key)
    if cfg["fixed"]["hmax_removed"] or not cfg["fixed"]["numeric_lead"]:
        raise ValueError("numeric/hmax policy differs")


def verify(cfg, *, sealed=False):
    validate_config(cfg)
    for path, digest in cfg["pins"].items():
        if core.sha(ROOT / path) != digest:
            raise ValueError("input hash changed: " + path)
    original = core.read(ROOT / "configs/experiments/p3_numeric_training_curve_20260906_v1.json")
    for field in ("recipes", "held_inner", "outer", "fixed", "baseline"):
        if cfg[field] != original[field]:
            raise ValueError("scientific contract changed: " + field)
    if cfg["budget"]["max_new_fits"] != 18 or cfg["reuse"]["model_count"] != 2:
        raise ValueError("two reused plus eighteen new maximum required")
    for kind, item in cfg["reuse"]["models"].items():
        path = (ROOT / item["source"]).resolve()
        if path != (REUSE_OUT / "models/baseline_INNER" / (kind + ".cbm")).resolve():
            raise ValueError("reuse source outside explicit two models")
        if core.sha(path) != item["sha256"]:
            raise ValueError("reuse source model hash changed")
    previous = core.read(e.CONFIG)
    core.verify_inputs(e, previous)
    if sealed:
        seal = core.read(OUT / "seal.json")
        if seal["runner_sha256"] != core.sha(__file__) or seal["config_sha256"] != core.sha(CONFIG):
            raise ValueError("sealed source changed")
    return previous


def load_inputs(previous):
    features, cache_anchors, columns = e.load_cache(previous)
    anchors = pd.read_parquet(OLD_OUT / "anchors.parquet")
    anchors.anchor_time = pd.to_datetime(anchors.anchor_time, utc=True)
    if not anchors[["anchor_id", "station"]].equals(cache_anchors[["anchor_id", "station"]]):
        raise ValueError("raw episode anchors/cache keys differ")
    return features, anchors, columns, e.cv.load_contract(ROOT / previous["contract"])


def inner_masks(anchors, contract, settings):
    start, end = pd.Timestamp(settings["start"]), pd.Timestamp(settings["end"])
    parent = e.cv.p3_split(anchors, settings["parent_train_fold"], contract)["train"]
    times = pd.to_datetime(anchors.anchor_time, utc=True)
    validation = parent & times.ge(start).to_numpy() & times.lt(end).to_numpy()
    pairs = list(zip(anchors.station, anchors.episode_id, strict=True))
    held = {pair for pair, kept in zip(pairs, validation, strict=True) if kept}
    train = parent & times.lt(start - pd.Timedelta(hours=78)).to_numpy()
    train &= np.array([pair not in held for pair in pairs])
    if not train.any() or not validation.any() or np.any(train & validation):
        raise ValueError("inner support/overlap")
    if times[train].max() + pd.Timedelta(hours=24) >= times[validation].min() - pd.Timedelta(
        hours=48
    ):
        raise ValueError("inner target/context overlap")
    outer_start = pd.Timestamp(
        next(
            f["start"] for f in contract["P3"]["folds"] if f["id"] == settings["parent_train_fold"]
        )
    )
    if times[validation].max() + pd.Timedelta(hours=24) >= outer_start - pd.Timedelta(hours=48):
        raise ValueError("inner labels not ready before outer context")
    return {"train": train, "validation": validation}


def modified_recipe(previous, arm):
    recipe = core.read(ROOT / previous["recipe"])
    for kind in ("single", "multi"):
        recipe["model"][kind]["iterations"] = arm[kind + "_iterations"]
        recipe["model"][kind]["learning_rate"] = arm[kind + "_learning_rate"]
    return recipe


def native_matches(path, kind, expected, columns):
    """Inspect native metadata on CPU; never predict or fit in acceptance."""
    model = core.CatBoostRegressor().load_model(path)
    actual = model.get_params()
    for key, value in expected.items():
        if key not in actual or actual[key] != value:
            raise ValueError("native configured parameter differs: " + key)
    names = (
        ["station", "lead_h", "current_hs_for_residual", *columns]
        if kind == "numeric_single"
        else ["station", *columns]
    )
    if model.feature_names_ != names or model.get_cat_feature_indices() != [0]:
        raise ValueError("native feature order/categorical contract differs")
    if model.tree_count_ != expected["iterations"] or "hmax_current" not in names:
        raise ValueError("native completed trees or hmax contract differs")
    return True


def accept_models(cfg, previous, columns, anchors, split):
    pilot = core.read(REUSE_REPORT / "resource-pilot.json")
    failure = core.read(REUSE_REPORT / "execute-failure.json")
    prior_preflight = core.read(REUSE_REPORT / "preflight.json")
    old_receipts = core.read(REUSE_OUT / "fit-receipts.json")["fits"]
    if (
        pilot["status"] != "RESOURCE_STOP"
        or pilot["metrics_inspected"] is not False
        or failure["status"] != "RESOURCE_STOP"
        or len(old_receipts) != 1
    ):
        raise ValueError("prior uninspected resource stop not verified")
    if any((REUSE_OUT / p).exists() for p in ("selection.json", "outer-start.json")):
        raise ValueError("prior selection/outer already started")
    for name in ("train", "validation"):
        digest = key_sha(
            anchors.loc[split[name]], ["anchor_id", "station", "anchor_time", "episode_id"]
        )
        if digest != prior_preflight["inner"][name]["key_sha256"]:
            raise ValueError("reused model training/validation population differs")
    single = old_receipts[0]
    recipe = modified_recipe(previous, cfg["recipes"][0])
    accepted = {}
    for kind, item in cfg["reuse"]["models"].items():
        source = ROOT / item["source"]
        if core.sha(source) != item["sha256"]:
            raise ValueError("reuse source hash differs")
        model_kind = "single" if kind == "numeric_single" else "multi"
        params = e.parameters(recipe, model_kind, cfg["held_inner"]["seed"])
        native_matches(source, kind, params, columns)
        if kind == "numeric_single" and (
            single["model_sha256"] != item["sha256"] or single["parameters"] != params
        ):
            raise ValueError("original single receipt differs")
        record = {
            "fold": "baseline_INNER",
            "kind": kind,
            "recipe": "baseline",
            "evaluation_fold": "INNER",
            "seed": cfg["held_inner"]["seed"],
            "rows": int(split["train"].sum()) * (6 if model_kind == "single" else 1),
            "train_anchors": int(split["train"].sum()),
            "validation_anchors": int(split["validation"].sum()),
            "iterations": params["iterations"],
            "seconds": single["seconds"]
            if model_kind == "single"
            else pilot["seconds_first_two"] - single["seconds"],
            "seconds_evidence": "original_receipt"
            if model_kind == "single"
            else "pilot_total_minus_original_single",
            "model_path": "models/baseline_INNER/" + kind + ".cbm",
            "source_path": item["source"],
            "model_sha256": item["sha256"],
            "parameters": params,
            "cat_features": ["station"],
            "lead_dtype": "float64" if model_kind == "single" else "not a multi input",
            "train_anchor_sha256": key_sha(anchors.loc[split["train"]], ["anchor_id", "station"]),
            "validation_anchor_sha256": key_sha(
                anchors.loc[split["validation"]], ["anchor_id", "station"]
            ),
            "reused": True,
            "hash_evidence": item["hash_evidence"],
        }
        if model_kind == "single":
            record["original_prediction_sha256"] = single["validation_prediction_sha256"]
        accepted[kind] = record
    return {
        "status": "PASS",
        "receipts": accepted,
        "new_fits": 0,
        "prediction_calls": 0,
        "prior_pilot_sha256": core.sha(REUSE_REPORT / "resource-pilot.json"),
        "prior_failure_sha256": core.sha(REUSE_REPORT / "execute-failure.json"),
        "prior_preflight_sha256": core.sha(REUSE_REPORT / "preflight.json"),
        "created_utc": now(),
        "pid": os.getpid(),
    }


def copy_accepted_models(accepted):
    for record in accepted["receipts"].values():
        target = (OUT / record["model_path"]).resolve()
        if OUT.resolve() not in target.parents or target.exists():
            raise ValueError("reuse copy target not empty owned path")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / record["source_path"], target)
        if core.sha(target) != record["model_sha256"]:
            raise ValueError("reused model copy hash mismatch")


def verify_accepted_copies(accepted):
    if accepted["status"] != "PASS" or set(accepted["receipts"]) != {"numeric_single", "multi"}:
        raise ValueError("two accepted models required")
    for record in accepted["receipts"].values():
        target = (OUT / record["model_path"]).resolve()
        if OUT.resolve() not in target.parents or core.sha(target) != record["model_sha256"]:
            raise ValueError("accepted copy differs or escapes owned directory")


def preflight(cfg, previous):
    if OUT.exists():
        raise FileExistsError("preserve existing trial")
    code_qa = core.read(REPORT / "code-qa.json")
    if (
        code_qa["status"] != "PASS"
        or code_qa["ruff"] != "PASS"
        or code_qa["runner_sha256"] != core.sha(__file__)
        or code_qa["config_sha256"] != core.sha(CONFIG)
        or code_qa["tests_sha256"] != core.sha(ROOT / "tests" / ("test_" + NAME + ".py"))
    ):
        raise ValueError("matching synthetic code QA required")
    features, anchors, columns, contract = load_inputs(previous)
    split = inner_masks(anchors, contract, cfg["held_inner"])
    masks = e.fold_masks(anchors, contract)
    baseline = pd.read_parquet(OLD_OUT / "candidate_oof.parquet")
    expected_keys = (
        pd.concat(
            [
                e.component_frame(
                    anchors,
                    s["validation"],
                    f["id"],
                    np.zeros((int(s["validation"].sum()), 6)),
                    np.zeros((int(s["validation"].sum()), 6)),
                )[e.KEYS]
                for f, s in masks
            ],
            ignore_index=True,
        )
        .sort_values(["anchor_id", "lead_h"])
        .reset_index(drop=True)
    )
    baseline = baseline.sort_values(["anchor_id", "lead_h"]).reset_index(drop=True)
    if len(baseline) != 103602 or not expected_keys.equals(baseline[e.KEYS]):
        raise ValueError("prior numeric population differs")
    qa = core.read(e.REPORT / "independent-qa.json")
    if qa["checks_count"] != 149 or not all(item["pass"] is True for item in qa["checks"]):
        raise ValueError("historical independent QA incomplete")
    accepted = accept_models(cfg, previous, columns, anchors, split)
    OUT.mkdir(parents=True)
    copy_accepted_models(accepted)
    core.save(OUT / "accepted-reuse.json", accepted)
    support = {
        name: {
            "anchors": int(mask.sum()),
            "rows": int(mask.sum()) * 6,
            "station_anchors": anchors.loc[mask].groupby("station").size().to_dict(),
            "key_sha256": key_sha(
                anchors.loc[mask], ["anchor_id", "station", "anchor_time", "episode_id"]
            ),
        }
        for name, mask in split.items()
    }
    receipt = {
        "status": "PASS",
        "pid": os.getpid(),
        "inner": support,
        "outer_rows": len(baseline),
        "outer_key_sha256": key_sha(expected_keys, e.KEYS),
        "outer_train_counts": [int(s["train"].sum()) for _, s in masks],
        "outer_validation_counts": [int(s["validation"].sum()) for _, s in masks],
        "features": len(columns),
        "numeric_lead": True,
        "hmax_present": "hmax_current" in columns,
        "all_inner_targets_before_first_outer_context": True,
        "created_utc": now(),
        "official_rows": 0,
        "new_fits": 0,
        "reused_models": 2,
        "accepted_reuse_sha256": core.sha(OUT / "accepted-reuse.json"),
        "original_resource_stop_preserved": True,
        "prediction_calls": 0,
    }
    core.save(REPORT / "preflight.json", receipt)
    core.save(
        OUT / "seal.json",
        {
            "runner_sha256": core.sha(__file__),
            "config_sha256": core.sha(CONFIG),
            "preflight_sha256": core.sha(REPORT / "preflight.json"),
            "created_utc": now(),
            "code_qa_sha256": core.sha(REPORT / "code-qa.json"),
        },
    )


def inner_prediction(frame):
    equal = (frame.single_prediction.to_numpy() + frame.multi_prediction.to_numpy()) * 0.5
    return e.apply_long_lead_persistence_shrink(
        equal,
        frame.persistence.to_numpy(),
        frame.lead_h.to_numpy(),
        config=e.LongLeadPersistenceShrink(0.2, (12, 18, 24)),
    )


def select_inner(rows):
    if len(rows) != 3 or [r["recipe"] for r in rows] != ["baseline", "compact", "gentle"]:
        raise ValueError("all three frozen inner recipes required")
    if not all(math.isfinite(r["sse_m2"]) and r["sse_m2"] >= 0 for r in rows):
        raise ValueError("nonfinite inner metric")
    return min(range(3), key=lambda i: (rows[i]["sse_m2"], i))


def fit_component(
    arm, fold, number, split, features, anchors, columns, recipe, previous, deadline, kind
):
    fit_fold = {"id": arm["id"] + "_" + fold["id"]}
    with owned_output():
        prediction, receipt = e.fit_one(
            kind, fit_fold, number, split, features, anchors, columns, recipe, previous, deadline
        )
    receipt.update(
        recipe=arm["id"],
        evaluation_fold=fold["id"],
        validation_prediction_sha256=prediction_sha(prediction),
        validation_anchor_sha256=key_sha(
            anchors.loc[split["validation"]], ["anchor_id", "station"]
        ),
        train_anchor_sha256=key_sha(anchors.loc[split["train"]], ["anchor_id", "station"]),
    )
    return prediction, receipt


def execute(cfg, previous):
    begin = time.perf_counter()
    verify(cfg, sealed=True)
    core.save(
        OUT / "ATTEMPT_LOCK.json",
        {"pid": os.getpid(), "created_utc": now(), "budget": cfg["budget"]},
    )
    accepted = core.read(OUT / "accepted-reuse.json")
    verify_accepted_copies(accepted)
    fits, inner_scores, inner_components = [], [], []
    deadline = begin + 5400
    features, anchors, columns, contract = load_inputs(previous)
    split = inner_masks(anchors, contract, cfg["held_inner"])
    masks = e.fold_masks(anchors, contract)

    def progress(stage):
        e.write_json(
            OUT / "progress.json",
            {
                "stage": stage,
                "pid": os.getpid(),
                "completed_backbones": len(fits),
                "reused_backbones": sum(r.get("reused", False) for r in fits),
                "new_backbones": sum(not r.get("reused", False) for r in fits),
                "maximum_backbones": 16,
                "elapsed_seconds": time.perf_counter() - begin,
            },
            progress=True,
        )
        e.write_json(OUT / "fit-receipts.json", {"fits": fits}, progress=True)
        print(
            json.dumps(
                {
                    "stage": stage,
                    "completed_backbones": len(fits),
                    "elapsed_seconds": time.perf_counter() - begin,
                }
            ),
            flush=True,
        )
        if time.perf_counter() >= deadline:
            raise TimeoutError("90min fixed fit budget expired")

    timer = threading.Timer(max(0.001, deadline - time.perf_counter()), lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    try:
        for arm in cfg["recipes"]:
            recipe = modified_recipe(previous, arm)
            predictions = []
            for kind in ("numeric_single", "multi"):
                progress("INNER_" + arm["id"] + "_" + kind)
                if arm["id"] == "baseline":
                    receipt = copy.deepcopy(accepted["receipts"][kind])
                    ids = anchors.loc[split["validation"], "anchor_id"].to_numpy(int)
                    prediction = predict_saved(receipt, features, anchors, columns, ids)
                    digest = prediction_sha(prediction)
                    original_digest = receipt.pop("original_prediction_sha256", None)
                    if original_digest is not None and digest != original_digest:
                        raise ValueError("reused original single prediction differs")
                    receipt["validation_prediction_sha256"] = digest
                    receipt["prediction_hash_evidence"] = (
                        "original_fit_hash_exact"
                        if original_digest
                        else "first_persisted_after_resource_amendment"
                    )
                else:
                    prediction, receipt = fit_component(
                        arm,
                        {"id": "INNER"},
                        0,
                        split,
                        features,
                        anchors,
                        columns,
                        recipe,
                        previous,
                        deadline,
                        kind,
                    )
                fits.append(receipt)
                predictions.append(prediction)
                progress("INNER_" + arm["id"] + "_" + kind + "_COMPLETE")
            frame = e.component_frame(anchors, split["validation"], "INNER", *predictions)
            inner_components.append(frame)
            if arm["id"] == "baseline":
                # Outcomes are not evaluated until resource forecast is accepted.
                scales = sum(r["scale"] for r in cfg["recipes"])
                ratio = sum(int(s["train"].sum()) for _, s in masks) / int(split["train"].sum())
                estimate = 1.2 * (fits[0]["seconds"] + fits[1]["seconds"]) * (scales + 1.5 * ratio)
                core.save(
                    REPORT / "resource-pilot.json",
                    {
                        "status": "CONTINUE" if estimate < 5400 else "RESOURCE_STOP",
                        "seconds_first_two": sum(r["seconds"] for r in fits),
                        "estimated_worst_seconds": estimate,
                        "metrics_inspected": False,
                        "actual_planned_fits_reused": True,
                        "resource_amendment_seconds": 5400,
                        "original_two_fits_already_completed": True,
                        "previous_pilot_sha256": core.sha(REUSE_REPORT / "resource-pilot.json"),
                    },
                )
                if estimate >= 5400:
                    raise TimeoutError("pilot forecast exceeds90min before outcome inspection")
        for arm, frame in zip(cfg["recipes"], inner_components, strict=True):
            pred = inner_prediction(frame)
            sse = math.fsum(float(v) ** 2 for v in pred - frame.target_hs.to_numpy())
            inner_scores.append(
                {
                    "recipe": arm["id"],
                    "rows": len(frame),
                    "sse_m2": sse,
                    "rmse_m": math.sqrt(sse / len(frame)),
                }
            )
        selected = cfg["recipes"][select_inner(inner_scores)]
        core.save(
            OUT / "selection.json",
            {
                "status": "FROZEN_BEFORE_OUTER",
                "recipe": selected,
                "scores": inner_scores,
                "created_utc": now(),
                "completed_backbones": len(fits),
                "selection_uses_outer_labels": False,
            },
        )
        for arm, frame in zip(cfg["recipes"], inner_components, strict=True):
            frame.to_parquet(OUT / ("inner_" + arm["id"] + ".parquet"), index=False)
        baseline = (
            pd.read_parquet(OLD_OUT / "candidate_oof.parquet")
            .sort_values(["anchor_id", "lead_h"])
            .reset_index(drop=True)
        )
        routers = []
        if selected["id"] == "baseline":
            candidate = baseline.copy()
        else:
            parts = []
            recipe = modified_recipe(previous, selected)
            core.save(
                OUT / "outer-start.json",
                {"created_utc": now(), "selection_sha256": core.sha(OUT / "selection.json")},
            )
            for number, (fold, outer_split) in enumerate(masks):
                predictions = []
                for kind in ("numeric_single", "multi"):
                    progress("OUTER_" + fold["id"] + "_" + kind)
                    prediction, receipt = fit_component(
                        selected,
                        fold,
                        number,
                        outer_split,
                        features,
                        anchors,
                        columns,
                        recipe,
                        previous,
                        deadline,
                        kind,
                    )
                    predictions.append(prediction)
                    fits.append(receipt)
                    progress("OUTER_" + fold["id"] + "_" + kind + "_COMPLETE")
                parts.append(
                    e.component_frame(anchors, outer_split["validation"], fold["id"], *predictions)
                )
            progress("OUTER_PRIOR_ROUTER")
            with owned_output():
                candidate, routers = e.fixed_policy(
                    pd.concat(parts, ignore_index=True),
                    features,
                    anchors,
                    contract,
                    arm="selected",
                    fit=True,
                )
        candidate.to_parquet(OUT / "candidate_oof.parquet", index=False)
        if not candidate[e.KEYS].equals(baseline[e.KEYS]) or not np.array_equal(
            candidate.target_hs, baseline.target_hs
        ):
            raise ValueError("outer comparison keys/targets differ")
        paired = baseline[[*e.KEYS, "anchor_time", "episode_id", "target_hs"]].copy()
        paired["control"] = baseline.final_prediction.to_numpy()
        paired["candidate"] = candidate.final_prediction.to_numpy()
        wind_columns = [
            c
            for c in columns
            if c.startswith(("wspd_", "gust_", "wdir_sin_", "wdir_cos_")) and "_valid_" in c
        ]
        paired["wind_observed"] = (
            features.set_index("anchor_id")
            .loc[paired.anchor_id, wind_columns]
            .gt(0)
            .any(axis=1)
            .to_numpy()
        )
        paired.to_parquet(OUT / "paired_oof.parquet", index=False)
        comparison = e.summary(paired, contract)
        progress("COMPLETE")
        core.save(
            REPORT / "result.json",
            {
                "status": "MEAN_IMPROVEMENT_CANDIDATE_RETAINED"
                if comparison["candidate_retained"]
                else "NO_MEAN_IMPROVEMENT_RETAIN_NUMERIC_BASELINE",
                "pid": os.getpid(),
                "selected_recipe": selected,
                "inner_scores": inner_scores,
                "comparison": comparison,
                "backbone_fits": len(fits),
                "backbone_fits_definition": "unique completed models including two v1 reused fits",
                "new_backbone_fits": sum(not r.get("reused", False) for r in fits),
                "reused_backbone_fits": 2,
                "accepted_reuse_sha256": core.sha(OUT / "accepted-reuse.json"),
                "router_fits": sum(r["model_path"] is not None for r in routers),
                "fit_receipts": fits,
                "router_receipts": routers,
                "execution_seconds": time.perf_counter() - begin,
                "selection_sha256": core.sha(OUT / "selection.json"),
                "preflight_sha256": core.sha(REPORT / "preflight.json"),
                "runner_sha256": core.sha(__file__),
                "config_sha256": core.sha(CONFIG),
                "artifacts": {p.name: core.sha(p) for p in OUT.glob("*.parquet")},
                "baseline_oof_sha256": core.sha(OLD_OUT / "candidate_oof.parquet"),
                "authorized_prior_oof_rows": 103602,
                "old_model_loads": 2,
                "old_model_loads_scope": "only same-hypothesis v1 baseline INNER exact-hash models",
                "official_rows": 0,
                "hidden_rows": 0,
                "csv_rows": 0,
                "uploads": 0,
                "whole_cold": False,
                "saved_replay": "PENDING",
                "full_fits": 0,
            },
        )
    finally:
        timer.cancel()


def predict_saved(receipt, features, anchors, columns, ids):
    path = OUT / receipt["model_path"]
    if core.sha(path) != receipt["model_sha256"]:
        raise ValueError("saved model hash differs")
    model = core.CatBoostRegressor().load_model(path)
    if receipt["kind"] == "numeric_single":
        x, _, meta = e.expand_leads(features, anchors, ids, columns)
        prediction = np.clip(
            meta.current_hs.to_numpy() + model.predict(e.matrix(x, True), thread_count=2), 0, 30
        )
        frame = meta[["anchor_id", "lead_h"]].copy()
        frame["value"] = prediction
        return frame.sort_values(["anchor_id", "lead_h"]).value.to_numpy().reshape(-1, 6)
    x, _, current = e.multi_matrix(features, anchors, ids, columns)
    return np.clip(current[:, None] + model.predict(x, thread_count=2), 0, 30)


def replay(cfg, previous):
    begin = time.perf_counter()
    result = core.read(REPORT / "result.json")
    if result["pid"] == os.getpid():
        raise ValueError("new process required")
    features, anchors, columns, contract = load_inputs(previous)
    inner = inner_masks(anchors, contract, cfg["held_inner"])
    masks = {fold["id"]: split for fold, split in e.fold_masks(anchors, contract)}
    maximum = 0.0
    for receipt in result["fit_receipts"]:
        split = (
            inner if receipt["evaluation_fold"] == "INNER" else masks[receipt["evaluation_fold"]]
        )
        ids = anchors.loc[split["validation"], "anchor_id"].to_numpy(int)
        prediction = predict_saved(receipt, features, anchors, columns, ids)
        if prediction_sha(prediction) != receipt["validation_prediction_sha256"]:
            raise ValueError("full saved component array replay differs")
    if result["selected_recipe"]["id"] != "baseline":
        component = pd.read_parquet(OUT / "candidate_oof.parquet")
        with owned_output():
            reproduced, _ = e.fixed_policy(
                component.drop(columns="final_prediction"),
                features,
                anchors,
                contract,
                arm="selected",
                fit=False,
                expected_receipts=result["router_receipts"],
            )
        maximum = float(np.max(np.abs(reproduced.final_prediction - component.final_prediction)))
        if maximum != 0:
            raise ValueError("full selected OOF router replay differs")
    core.save(
        REPORT / "fresh-process-replay.json",
        {
            "status": "PASS",
            "pid": os.getpid(),
            "training_pid": result["pid"],
            "backbone_models": result["backbone_fits"],
            "router_models": result["router_fits"],
            "all_saved_component_arrays_exact": True,
            "outer_rows": 103602,
            "max_abs_final_difference_m": maximum,
            "result_sha256": core.sha(REPORT / "result.json"),
            "seconds": time.perf_counter() - begin,
            "official_rows": 0,
            "new_fits": 0,
            "whole_cold": False,
        },
    )


def qa(cfg, previous):
    result = core.read(REPORT / "result.json")
    replay_receipt = core.read(REPORT / "fresh-process-replay.json")
    pre = core.read(REPORT / "preflight.json")
    features, anchors, columns, contract = load_inputs(previous)
    inner = inner_masks(anchors, contract, cfg["held_inner"])
    paired = pd.read_parquet(OUT / "paired_oof.parquet")
    baseline = (
        pd.read_parquet(OLD_OUT / "candidate_oof.parquet")
        .sort_values(["anchor_id", "lead_h"])
        .reset_index(drop=True)
    )
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    for name, digest in result["artifacts"].items():
        check("artifact/" + name, core.sha(OUT / name) == digest)
    check(
        "source/config",
        result["runner_sha256"] == core.sha(__file__)
        and result["config_sha256"] == core.sha(CONFIG),
    )
    check("preflight/link", result["preflight_sha256"] == core.sha(REPORT / "preflight.json"))
    check("selection/link", result["selection_sha256"] == core.sha(OUT / "selection.json"))
    selection = core.read(OUT / "selection.json")
    check(
        "selection/all_inner_only",
        selection["completed_backbones"] == 6 and not selection["selection_uses_outer_labels"],
    )
    recomputed = []
    for arm in cfg["recipes"]:
        frame = pd.read_parquet(OUT / ("inner_" + arm["id"] + ".parquet"))
        y = (
            anchors.set_index("anchor_id")
            .loc[frame.anchor_id.to_numpy(), [f"target_{lead}" for lead in e.LEADS]]
            .to_numpy()
        )
        expected_y = y[np.arange(len(frame)), np.tile(np.arange(6), len(frame) // 6)]
        check("inner/" + arm["id"] + "/target", np.array_equal(expected_y, frame.target_hs))
        prediction = inner_prediction(frame)
        sse = math.fsum(float(v) ** 2 for v in prediction - frame.target_hs.to_numpy())
        recomputed.append({"recipe": arm["id"], "sse_m2": sse})
        check("inner/" + arm["id"] + "/count", len(frame) == int(inner["validation"].sum()) * 6)
    check(
        "selection/exact_argmin",
        cfg["recipes"][select_inner(recomputed)]["id"] == result["selected_recipe"]["id"],
    )
    check("population/103602", len(paired) == 103602 and paired.anchor_id.nunique() == 17267)
    check(
        "keys/exact_prior",
        paired[e.KEYS].equals(baseline[e.KEYS]) and not paired.duplicated(e.KEYS).any(),
    )
    check("targets/prior_exact", np.array_equal(paired.target_hs, baseline.target_hs))
    check("control/prior_exact", np.array_equal(paired.control, baseline.final_prediction))
    check(
        "inner/parent_key_hash",
        key_sha(anchors.loc[inner["train"]], ["anchor_id", "station", "anchor_time", "episode_id"])
        == pre["inner"]["train"]["key_sha256"],
    )
    check(
        "finite/range",
        np.isfinite(paired[["target_hs", "control", "candidate"]]).all().all()
        and paired[["control", "candidate"]].ge(0).all().all()
        and paired[["control", "candidate"]].le(30).all().all(),
    )
    for arm in ("control", "candidate"):
        sse = math.fsum(float(v) ** 2 for v in paired[arm] - paired.target_hs)
        check(
            "sse/" + arm,
            math.isclose(sse, result["comparison"][arm + "_sse_m2"], rel_tol=0, abs_tol=1e-8),
        )
        check(
            "rmse/" + arm,
            math.isclose(
                math.sqrt(sse / len(paired)), result["comparison"][arm], rel_tol=0, abs_tol=1e-12
            ),
        )
    independent = independent_bootstrap(paired, contract["common"]["bootstrap"])
    check(
        "bootstrap/CI",
        np.allclose(independent["ci90"], result["comparison"]["ci90"], rtol=0, atol=1e-12),
    )
    expected_fits = 6 if result["selected_recipe"]["id"] == "baseline" else 16
    check("fits/exact", result["backbone_fits"] == len(result["fit_receipts"]) == expected_fits)
    check("routers/exact", result["router_fits"] == (0 if expected_fits == 6 else 4))
    check("allfits/cap", result["backbone_fits"] + result["router_fits"] <= 20)
    check("newfits/cap", result["new_backbone_fits"] + result["router_fits"] <= 18)
    check("newfits/exact", result["new_backbone_fits"] == expected_fits - 2)
    check(
        "reused/exact_two",
        result["reused_backbone_fits"] == 2
        and sum(r.get("reused", False) for r in result["fit_receipts"]) == 2,
    )
    accepted = core.read(OUT / "accepted-reuse.json")
    verify_accepted_copies(accepted)
    check(
        "reused/acceptance_hash",
        result["accepted_reuse_sha256"] == core.sha(OUT / "accepted-reuse.json"),
    )
    check(
        "reused/native_parameters",
        all(
            native_matches(OUT / r["model_path"], r["kind"], r["parameters"], columns)
            for r in accepted["receipts"].values()
        ),
    )
    check("time/cap", result["execution_seconds"] < 5400)
    check(
        "replay/new_pid",
        replay_receipt["status"] == "PASS"
        and replay_receipt["pid"] != result["pid"]
        and replay_receipt["result_sha256"] == core.sha(REPORT / "result.json"),
    )
    outer_masks = {fold["id"]: split for fold, split in e.fold_masks(anchors, contract)}
    if expected_fits == 16:
        outer_start = core.read(OUT / "outer-start.json")
        check(
            "selection/before_outer",
            pd.Timestamp(selection["created_utc"]) < pd.Timestamp(outer_start["created_utc"])
            and outer_start["selection_sha256"] == result["selection_sha256"],
        )
    for receipt in result["fit_receipts"]:
        label = receipt["recipe"] + "/" + receipt["evaluation_fold"] + "/" + receipt["kind"]
        model = core.CatBoostRegressor().load_model(OUT / receipt["model_path"])
        arm = next(r for r in cfg["recipes"] if r["id"] == receipt["recipe"])
        kind = "single" if receipt["kind"] == "numeric_single" else "multi"
        split = (
            inner
            if receipt["evaluation_fold"] == "INNER"
            else outer_masks[receipt["evaluation_fold"]]
        )
        number = (
            0
            if receipt["evaluation_fold"] == "INNER"
            else list(outer_masks).index(receipt["evaluation_fold"])
        )
        params = e.parameters(modified_recipe(previous, arm), kind, previous["fold_seeds"][number])
        check("model/" + label + "/exact_parameters", receipt["parameters"] == params)
        check(
            "model/" + label + "/train_keys",
            receipt["train_anchor_sha256"]
            == key_sha(anchors.loc[split["train"]], ["anchor_id", "station"]),
        )
        check(
            "model/" + label + "/validation_keys",
            receipt["validation_anchor_sha256"]
            == key_sha(anchors.loc[split["validation"]], ["anchor_id", "station"]),
        )
        check(
            "model/" + label + "/row_counts",
            receipt["train_anchors"] == int(split["train"].sum())
            and receipt["validation_anchors"] == int(split["validation"].sum()),
        )
        check(
            "model/" + label + "/sha",
            core.sha(OUT / receipt["model_path"]) == receipt["model_sha256"],
        )
        check("model/" + label + "/iterations", model.tree_count_ == arm[kind + "_iterations"])
        check("model/" + label + "/cats", model.get_cat_feature_indices() == [0])
        check("model/" + label + "/hmax", "hmax_current" in model.feature_names_)
    if expected_fits == 16:
        component = pd.read_parquet(OUT / "candidate_oof.parquet")
        _, meta, _, _ = e.router_material(component, features)
        completed = []
        for fold, split in e.fold_masks(anchors, contract):
            prior = meta.fold.isin(completed) & meta.anchor_id.isin(
                anchors.loc[split["train"], "anchor_id"]
            )
            receipt = next(r for r in result["router_receipts"] if r["fold"] == fold["id"])
            check(
                "router/" + fold["id"] + "/keys",
                receipt["prior_key_sha256"] == key_sha(meta.loc[prior], e.KEYS),
            )
            check("router/" + fold["id"] + "/rows", receipt["past_fit_rows"] == int(prior.sum()))
            if completed:
                check(
                    "router/" + fold["id"] + "/target_availability",
                    (
                        meta.loc[prior, "anchor_time"] + pd.Timedelta(hours=24)
                        < pd.Timestamp(fold["start"]) - pd.Timedelta(hours=48)
                    ).all(),
                )
            completed.append(fold["id"])
    check(
        "official/zero",
        result["official_rows"]
        == result["hidden_rows"]
        == result["csv_rows"]
        == result["uploads"]
        == result["full_fits"]
        == 0,
    )
    failed = [name for name, ok in checks.items() if not ok]
    core.save(
        REPORT / "independent-qa.json",
        {
            "status": "FAIL" if failed else "PASS",
            "checks": checks,
            "checks_count": len(checks),
            "failed_checks": failed,
            "result_sha256": core.sha(REPORT / "result.json"),
            "fresh_replay_sha256": core.sha(REPORT / "fresh-process-replay.json"),
            "independent_bootstrap": independent,
            "official_rows": 0,
        },
    )
    if failed:
        raise ValueError("independent QA failed: " + ",".join(failed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "execute", "replay", "qa"), required=True)
    parser.add_argument("--gpu-released-by-root", action="store_true")
    args = parser.parse_args()
    cfg = core.read(CONFIG)
    previous = verify(cfg, sealed=args.stage != "preflight")
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    guard_config = copy.deepcopy(previous)
    guard_config["inputs"].update(
        {item["source"]: item["sha256"] for item in cfg["reuse"]["models"].values()}
    )
    core.guard(source, OUT, guard_config, e, official=False)
    if args.stage == "execute" and not args.gpu_released_by_root:
        raise PermissionError("root GPU ownership signal required")
    try:
        {"preflight": preflight, "execute": execute, "replay": replay, "qa": qa}[args.stage](
            cfg, previous
        )
        print(
            json.dumps({"stage": args.stage, "status": "COMPLETE", "pid": os.getpid()}), flush=True
        )
    except Exception as exc:
        target = REPORT / (args.stage + "-failure.json")
        if not target.exists():
            core.save(
                target,
                {
                    "status": "RESOURCE_STOP"
                    if isinstance(exc, TimeoutError)
                    else "TERMINAL_TECHNICAL_FAILURE",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "pid": os.getpid(),
                    "automatic_restart": False,
                    "created_utc": now(),
                },
            )
        raise


if __name__ == "__main__":
    main()
