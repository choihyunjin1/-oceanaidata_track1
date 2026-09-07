"""Portable CPU ensemble cold runner. Historical helpers remain byte unchanged."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import run as b  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

P = b.OUT
D = b.DOCS
W = b.WORK
M = b.MODELS
E = b.module("numeric")
ORIGINAL_PARAMETERS = E.parameters


def settings():
    manifest = b.read(CODE / "cpudet-manifest.json")
    for name, expected in manifest.items():
        path = (P / name).resolve()
        if not path.is_relative_to(P) or b.sha(path) != expected:
            raise ValueError("source pin mismatch: " + name)
    cfg = b.read(CODE / "cpudet-config.json")
    if cfg["seeds"] != [20260817, 20260818, 20260819] or cfg["cpu_threads"] != 4:
        raise ValueError("fixed seed/resource contract")
    c = b.read(CODE / "frozen.json")
    c["budget"]["total_seconds"] = cfg["maximum_seconds_per_cold"]
    b.postprocessing_contract(c["recipe"])
    return cfg, c


def cpu_parameters(recipe, kind, seed, *, synthetic=False):
    params = ORIGINAL_PARAMETERS(recipe, kind, seed, synthetic=synthetic)
    params.pop("devices", None)
    params.update(task_type="CPU", thread_count=4)
    return params


def elapsed():
    return time.time() - b.read(D / "prepare.json")["started_unix"]


def log_fit(done):
    item = {
        "stage": "CPU_TRAINING",
        "completed_backbone_fits": done,
        "maximum_backbone_fits": 36,
        "seconds": elapsed(),
        "pid": os.getpid(),
    }
    E.write_json(P / "progress.json", item, progress=True)
    print(json.dumps(item), flush=True)


def fit_component(kind, seed, name, ids, features, anchors, columns, c, deadline):
    params = cpu_parameters(c["recipe"], kind, seed)
    if kind == "single":
        x, y, meta = E.expand_leads(features, anchors, ids, columns)
        x, current = E.matrix(x, True), meta.current_hs.to_numpy()
    else:
        x, y, current = E.multi_matrix(features, anchors, ids, columns)
    start = time.monotonic()
    model = CatBoostRegressor(**params)
    model.fit(
        x,
        y,
        sample_weight=E.threshold_case_weights(current),
        cat_features=["station"],
        callbacks=[E.Deadline(deadline)],
        verbose=False,
    )
    if model.tree_count_ != params["iterations"] or time.perf_counter() >= deadline:
        raise TimeoutError("full CPU fit not completed; no partial promotion")
    path = M / name
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    return {
        "kind": kind,
        "seed": seed,
        "model": path.relative_to(P).as_posix(),
        "sha256": b.sha(path),
        "iterations": model.tree_count_,
        "rows": len(x),
        "seconds": time.monotonic() - start,
        "parameters": params,
        "resolved_parameters": model.get_all_params(),
    }


def component_predictions(features, anchors, ids, columns, model_paths):
    singles, multis = [], []
    x, _, meta = E.expand_leads(features, anchors, ids, columns)
    x = E.matrix(x, True)
    mx, _, current = E.multi_matrix(features, anchors, ids, columns)
    for single_path, multi_path in model_paths:
        single = CatBoostRegressor().load_model(single_path)
        multi = CatBoostRegressor().load_model(multi_path)
        singles.append(b.single_case_major(meta, single.predict(x, thread_count=4)))
        multis.append(np.clip(current[:, None] + multi.predict(mx, thread_count=4), 0, 30))
    return np.mean(singles, axis=0), np.mean(multis, axis=0)


def paths(scope, cfg):
    return [
        (M / scope / f"{seed}_single.cbm", M / scope / f"{seed}_multi.cbm") for seed in cfg["seeds"]
    ]


def historical(cfg, c, features, anchors, columns):
    blocks = []
    for fold, split in E.fold_masks(
        anchors, b.read(CODE / "configs/evaluation/ocean_forward_v5.json")
    ):
        ids = anchors.loc[split["validation"], "anchor_id"].to_numpy(int)
        single, multi = component_predictions(
            features, anchors, ids, columns, paths(fold["id"], cfg)
        )
        blocks.append(E.component_frame(anchors, split["validation"], fold["id"], single, multi))
    return pd.concat(blocks, ignore_index=True)


def train(cfg, c):
    p, features, anchors, columns = b.prepared(c)
    if any(M.iterdir()) or len(columns) != 591:
        raise ValueError("fresh models and 591 features required")
    b.save(
        P / "CPU_TRAIN_LOCK.json", {"pid": os.getpid(), "backbone_budget": 36, "router_budget": 5}
    )
    timer = b.watchdog(c, p["started_unix"])
    deadline = time.perf_counter() + cfg["maximum_seconds_per_cold"] - elapsed()
    fits = []
    splits = E.fold_masks(anchors, b.read(CODE / "configs/evaluation/ocean_forward_v5.json"))
    for scope, ids in [
        (fold["id"], anchors.loc[split["train"], "anchor_id"].to_numpy(int))
        for fold, split in splits
    ] + [("full", anchors.anchor_id.to_numpy(int))]:
        for seed in cfg["seeds"]:
            for kind in ["single", "multi"]:
                fits.append(
                    fit_component(
                        kind,
                        seed,
                        f"{scope}/{seed}_{kind}.cbm",
                        ids,
                        features,
                        anchors,
                        columns,
                        c,
                        deadline,
                    )
                )
                E.write_json(P / "fit-progress.json", fits, progress=True)
                log_fit(len(fits))
    components = historical(cfg, c, features, anchors, columns)
    evaluation = b.read(CODE / "configs/evaluation/ocean_forward_v5.json")
    E.OUT = M / "routers"
    final, routers = E.fixed_policy(
        components, features, anchors, evaluation, arm="candidate", fit=True
    )
    x, meta, _, loss = E.router_material(components, features)
    if len(meta) != 103602:
        raise ValueError("router OOF population")
    router = E.ComponentLossRouter(E.RouterConfig(10.0, 2.0, 0.5, "smooth_medium")).fit(x, loss)
    joblib.dump(router, M / "full_router.joblib")
    cases = (
        features.iloc[np.unique(np.linspace(0, len(features) - 1, 128, dtype=int))]
        .copy()
        .reset_index(drop=True)
    )
    cases.insert(0, "case_id", [f"INTERNAL_{i:04d}" for i in range(len(cases))])
    _, probe = predict(cfg, cases, columns)
    cases.to_parquet(W / "full_probe_cases.parquet", index=False)
    np.save(W / "full_probe.npy", probe, allow_pickle=False)
    final.to_parquet(W / "candidate_oof.parquet", index=False)
    r = {
        "status": "CPU_S3_36_BACKBONE_5_ROUTER_COMPLETE",
        "pid": os.getpid(),
        "fits": fits,
        "backbone_fits": len(fits),
        "router_fits": 5,
        "historical_routers": routers,
        "internal_rmse": float(np.sqrt(np.mean((final.target_hs - final.final_prediction) ** 2))),
        "oof_sha256": b.sha(W / "candidate_oof.parquet"),
        "seconds": elapsed(),
        "official_rows": 0,
        "old_cache_model_prediction_inputs": 0,
        "hidden_rows": 0,
        "config_sha256": b.sha(CODE / "cpudet-config.json"),
        "models": {x.relative_to(P).as_posix(): b.sha(x) for x in M.rglob("*") if x.is_file()},
    }
    assert r["backbone_fits"] == 36 and sum(x.endswith(".joblib") for x in r["models"]) == 5
    b.save(D / "cpudet-training.json", r)
    timer.cancel()
    print(json.dumps({"status": r["status"], "seconds": r["seconds"]}), flush=True)


def training_receipt():
    r = b.read(D / "cpudet-training.json")
    assert r["pid"] != os.getpid() and r["backbone_fits"] == 36 and r["router_fits"] == 5
    for name, expected in r["models"].items():
        assert b.sha(P / name) == expected, name
    return r


def qa(cfg, c):
    r = training_receipt()
    p, features, anchors, columns = b.prepared(c)
    timer = b.watchdog(c, p["started_unix"])
    components = historical(cfg, c, features, anchors, columns)
    E.OUT = M / "routers"
    final, _ = E.fixed_policy(
        components,
        features,
        anchors,
        b.read(CODE / "configs/evaluation/ocean_forward_v5.json"),
        arm="candidate",
        fit=False,
        expected_receipts=r["historical_routers"],
    )
    previous = pd.read_parquet(W / "candidate_oof.parquet")
    assert previous[E.KEYS].equals(final[E.KEYS]) and np.array_equal(
        previous.final_prediction, final.final_prediction
    )
    assert (
        abs(
            float(np.sqrt(np.mean((final.target_hs - final.final_prediction) ** 2)))
            - r["internal_rmse"]
        )
        < 1e-12
    )
    _, probe = predict(cfg, pd.read_parquet(W / "full_probe_cases.parquet"), columns)
    assert np.array_equal(probe, np.load(W / "full_probe.npy", allow_pickle=False))
    b.save(
        D / "cpudet-training-qa.json",
        {
            "status": "PASS",
            "pid": os.getpid(),
            "training_pid": r["pid"],
            "training_sha256": b.sha(D / "cpudet-training.json"),
            "oof_rows": len(final),
            "oof_exact": True,
            "new_fits": 0,
            "official_rows": 0,
            "hidden_rows": 0,
            "seconds": elapsed(),
        },
    )
    timer.cancel()


class SeedEnsemble:
    """Average clipped component forecasts, expressed as residual for original predictor."""

    def __init__(self, models, multi):
        self.models, self.multi = models, multi

    def predict(self, frame, thread_count=4):
        current = (
            frame["hs_current"].to_numpy()
            if self.multi
            else frame["current_hs_for_residual"].to_numpy()
        )
        if self.multi:
            current = current[:, None]
        values = [np.clip(current + m.predict(frame, thread_count=4), 0, 30) for m in self.models]
        return np.mean(values, axis=0) - current


def predict(cfg, cases, columns):
    single = SeedEnsemble([CatBoostRegressor().load_model(a) for a, _ in paths("full", cfg)], False)
    multi = SeedEnsemble(
        [CatBoostRegressor().load_model(bpath) for _, bpath in paths("full", cfg)], True
    )
    router = joblib.load(M / "full_router.joblib")
    return b.materializer.predict_cases(E, {}, cases, columns, (single, multi, router))


def inference(cfg, c, source, verify):
    r = training_receipt()
    q = b.read(D / "cpudet-training-qa.json")
    assert q["status"] == "PASS" and q["training_sha256"] == b.sha(D / "cpudet-training.json")
    if not verify:
        b.save(P / "CPU_INFER_LOCK.json", {"pid": os.getpid(), "uploads": 0})
    columns = b.read(W / "columns.json")["columns"]
    hashes = {n: b.sha(source / n) for n in ["test_context.parquet", "test_index.csv"]}
    context = pd.read_parquet(
        source / "test_context.parquet",
        columns=["case_id", "station", "step_minute", *E.BASE_COLUMNS, *E.DIRECTION_COLUMNS],
    )
    index = pd.read_csv(source / "test_index.csv", usecols=b.materializer.KEYS)
    assert (
        len(context) == 57800
        and len(index) == 1200
        and not index.duplicated(b.materializer.KEYS).any()
    )
    records = []
    for case_id, group in context.groupby("case_id", sort=False):
        group = group.sort_values("step_minute")
        assert (
            len(group) == 289
            and group.station.nunique() == 1
            and np.array_equal(group.step_minute, np.arange(-2880, 1, 10))
        )
        records.append(
            {
                "case_id": case_id,
                "station": str(group.station.iloc[0]),
                **E.summarize_context(group),
            }
        )
    cases = (
        index[["case_id", "station"]]
        .drop_duplicates()
        .merge(
            pd.DataFrame(records),
            on=["case_id", "station"],
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
    )
    assert len(cases) == 200 and cases._merge.eq("both").all()
    keys, prediction = predict(cfg, cases.drop(columns="_merge"), columns)
    keys["hs_pred"] = prediction
    answer = index.merge(keys, on=b.materializer.KEYS, how="left", validate="one_to_one")
    assert (
        answer[b.materializer.KEYS].equals(index)
        and np.isfinite(answer.hs_pred).all()
        and answer.hs_pred.between(0, 30).all()
    )
    payload = answer.to_csv(index=False, lineterminator="\n").encode()
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    if verify:
        prev = b.read(D / "cpudet-answer.json")
        assert (
            prev["pid"] != os.getpid()
            and prev["sha256"] == digest
            and b.sha(P / "05_answer/submission.csv") == digest
        )
    else:
        (P / "05_answer/submission.csv").write_bytes(payload)
    assert hashes == {n: b.sha(source / n) for n in hashes}
    b.save(
        D / ("cpudet-replay.json" if verify else "cpudet-answer.json"),
        {
            "status": "EXACT_REPLAY_PASS" if verify else "LOCAL_NOT_UPLOADED",
            "pid": os.getpid(),
            "training_pid": r["pid"],
            "sha256": digest,
            "rows": 1200,
            "seconds": elapsed(),
            "source_sha256": hashes,
            "new_fits": 0,
            "hidden_rows": 0,
            "uploads": 0,
            "old_answer_values_read": 0,
        },
    )
    print(
        json.dumps(
            {"stage": "replay" if verify else "infer", "sha256": digest, "seconds": elapsed()}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train", "qa", "infer", "replay", "smoke"])
    args = parser.parse_args()
    cfg, c = settings()
    E.parameters = cpu_parameters
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    b.access_guard(source, args.stage in ["infer", "replay"])
    try:
        with threadpool_limits(limits=2):
            if args.stage == "smoke":
                x = pd.DataFrame(
                    {
                        "station": ["A", "B"] * 20,
                        "hs_current": np.linspace(1.5, 3, 40),
                        "lead_h": np.tile([3.0, 6.0, 9.0, 12.0, 18.0], 8),
                        "current_hs_for_residual": np.linspace(1.5, 3, 40),
                    }
                )
                for kind in ["single", "multi"]:
                    y = np.sin(x.hs_current.to_numpy())
                    if kind == "multi":
                        y = np.tile(y[:, None], (1, 6))
                    ps = []
                    for _ in range(2):
                        model = CatBoostRegressor(
                            **cpu_parameters(c["recipe"], kind, 20260817, synthetic=True)
                        )
                        model.fit(x, y, cat_features=["station"], verbose=False)
                        ps.append(model.predict(x, thread_count=4))
                    assert np.array_equal(*ps)
                b.save(
                    D / "cpudet-smoke.json",
                    {
                        "status": "PASS",
                        "synthetic_fits": 4,
                        "task_type": "CPU",
                        "production_fits": 0,
                    },
                )
            elif args.stage == "prepare":
                b.prepare(c, source)
            elif args.stage == "train":
                train(cfg, c)
            elif args.stage == "qa":
                qa(cfg, c)
            else:
                inference(cfg, c, source, args.stage == "replay")
    except BaseException as exc:
        target = D / ("cpudet-failure-" + args.stage + ".json")
        if not target.exists():
            b.save(
                target,
                {
                    "status": "TECHNICAL_FAILURE",
                    "error": str(exc),
                    "type": type(exc).__name__,
                    "pid": os.getpid(),
                    "restart": False,
                },
            )
        raise


if __name__ == "__main__":
    main()
