"""Approved improved-policy full fit and local answer; no uploads or new tuning.

Reuses only this cycle's validated candidate OOF and source-derived feature cache.
This is not a second uninterrupted whole cold-start proof.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

from p3_wave.loss_router import expand_case_router_features  # noqa: E402

EXPERIMENTS = ("p3_numeric_lead_forward_gpu_20260906_v2", "p3_hmax_removed_forward_20260906_v1")
KEYS = ["case_id", "station", "lead_h"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")


def read(path):
    return json.loads(path.read_text())


def guard(source, out, cfg, experiment, *, official):
    allowed = {(ROOT / p).resolve() for p in cfg["inputs"]}
    allowed.add((experiment.OUT / "candidate_oof.parquet").resolve())
    allowed.add((experiment.OUT / "anchors.parquet").resolve())
    if "baseline" in cfg:
        allowed.add((ROOT / cfg["baseline"]["out"] / "baseline_oof.parquet").resolve())
    source_allowed = {source / p for p in cfg["source_files"]}
    if official:
        source_allowed |= {source / "test_context.parquet", source / "test_index.csv"}

    def hook(event, args):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise PermissionError("network denied by Python hook; not OS isolation")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1], args[2]
        writing = (
            isinstance(mode, str)
            and any(c in mode for c in "wax+")
            or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        )
        if any(
            word in path.name.lower()
            for word in ("hidden", "credentials", "refined_alpha", "axis_contract")
        ):
            raise PermissionError("forbidden data or coefficient ancestry")
        owned = out in path.parents
        if source in path.parents and (path not in source_allowed or writing):
            raise PermissionError("outside immutable authorized source allowlist")
        if path in allowed and writing:
            raise PermissionError("same-cycle source inputs immutable")
        if (
            path.suffix.lower() in {".cbm", ".joblib", ".parquet", ".npz", ".pt", ".csv"}
            and not owned
            and path not in allowed
            and path not in source_allowed
        ):
            raise PermissionError("unapproved data, model or prediction input")
        if (
            owned
            and path.suffix.lower() == ".csv"
            and (path.parent != out / "05_answer" or not official)
        ):
            raise PermissionError("answer read/write requires approved inference phase")

    sys.addaudithook(hook)


def verify_inputs(e, cfg):
    result = read(e.REPORT / "result.json")
    qa, replay = (
        read(e.REPORT / name) for name in ("independent-qa.json", "fresh-process-replay.json")
    )
    digest = sha(e.REPORT / "result.json")
    if (
        not result["comparison"]["candidate_retained"]
        or result["comparison"]["delta_candidate_minus_control"] >= 0
    ):
        raise ValueError("no mean-improved complete policy; no full training")
    if (
        qa["status"] != "PASS"
        or qa["failed_checks"]
        or replay["status"] != "PASS"
        or qa["result_sha256"] != digest
        or replay["result_sha256"] != digest
    ):
        raise ValueError("independent historical QA/replay required")
    e.verify_pins(cfg, Path(os.environ["P3_DATA_DIR"]).resolve(), sealed=True)
    if sha(e.OUT / "candidate_oof.parquet") != result["artifacts"]["candidate_oof.parquet"]:
        raise ValueError("candidate OOF hash changed")
    return {
        str(path.relative_to(ROOT)): sha(path)
        for path in (
            e.REPORT / "result.json",
            e.REPORT / "independent-qa.json",
            e.REPORT / "fresh-process-replay.json",
            e.OUT / "candidate_oof.parquet",
            e.CONFIG,
            Path(e.__file__),
        )
    }


def policy_columns(e, cfg, columns):
    return e.selected_columns(columns) if "removal" in cfg else columns


def predict_cases(e, cfg, cases, columns, models):
    single, multi, router = models
    n = len(cases)
    x = cases[columns].iloc[np.repeat(np.arange(n), 6)].reset_index(drop=True)
    x.insert(0, "station", np.repeat(cases.station.astype(str).to_numpy(), 6))
    x.insert(1, "lead_h", np.tile(e.LEADS, n))
    current = np.repeat(cases.hs_current.to_numpy(float), 6)
    x.insert(2, "current_hs_for_residual", current)
    p_single = np.clip(
        current + single.predict(e.matrix(x, "removal" not in cfg), thread_count=2), 0, 30
    ).reshape(n, 6)
    mx = cases[["station", *columns]].copy()
    mx.station = mx.station.astype(str)
    persistence = np.repeat(cases.hs_current.to_numpy()[:, None], 6, axis=1)
    p_multi = np.clip(persistence + multi.predict(mx, thread_count=2), 0, 30)
    components = np.stack([p_single, p_multi, persistence], axis=2)
    observed = e.build_inference_router_features(
        cases[list(e.OBSERVED_FEATURES)],
        cases.station.to_numpy(),
        cases.hs_current.to_numpy(),
        components,
    )
    if "removal" in cfg:
        observed = observed.drop(columns=["hmax_current"])
    metadata = pd.DataFrame(
        {
            "fold": "full",
            "anchor_id": np.arange(n),
            "station": cases.station.to_numpy(),
            "anchor_time": pd.NaT,
        }
    )
    row_x, row_meta, row_components = expand_case_router_features(observed, metadata, components)
    weights = router.predict_weights(row_x)
    weights[~row_meta.lead_h.isin([12, 18, 24]).to_numpy()] = [0.5, 0.5, 0.0]
    routed = e.route_row_predictions(row_components, weights)
    prediction = e.apply_long_lead_persistence_shrink(
        routed,
        persistence.reshape(-1),
        row_meta.lead_h.to_numpy(),
        config=e.LongLeadPersistenceShrink(0.2, (12, 18, 24)),
    )
    keys = pd.DataFrame(
        {
            "case_id": np.repeat(cases.case_id.to_numpy(), 6),
            "station": x.station,
            "lead_h": x.lead_h,
        }
    )
    if not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 30)):
        raise ValueError("prediction finite/range failure")
    return keys, prediction


def fit(e, cfg, out):
    begin = time.perf_counter()
    if out.exists():
        raise FileExistsError("new full-fit output must not exist; preserve prior attempts")
    accepted = verify_inputs(e, cfg)
    for name in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
        (out / name).mkdir(parents=True)
    save(
        out / "FULL_FIT_LOCK.json",
        {
            "pid": os.getpid(),
            "created_utc": datetime.now(UTC).isoformat(),
            "full_backbone_budget": 2,
            "full_router_budget": 1,
            "empty_03_model": True,
        },
    )
    save(
        out / "seal.json",
        {
            "runner_sha256": sha(__file__),
            "accepted_same_cycle": accepted,
            "source_sha256": cfg["source_files"],
            "whole_cold_start": False,
        },
    )
    features, anchors, columns = e.load_cache(cfg)
    columns = policy_columns(e, cfg, columns)
    component = pd.read_parquet(e.OUT / "candidate_oof.parquet")
    x_router, meta_router, _, loss = e.router_material(component, features)
    if len(component) != 103602 or meta_router.anchor_id.nunique() != 17267:
        raise ValueError("fixed full-router OOF population differs")
    recipe = read(ROOT / cfg["recipe"])
    seed = recipe["model"]["full_train_seed"]
    ids = anchors.anchor_id.to_numpy(int)
    x, y, meta = e.expand_leads(features, anchors, ids, columns)
    single = CatBoostRegressor(**e.parameters(recipe, "single", seed))
    t = time.perf_counter()
    single.fit(
        e.matrix(x, "removal" not in cfg),
        y,
        sample_weight=e.threshold_case_weights(meta.current_hs.to_numpy()),
        cat_features=["station"] if "removal" not in cfg else ["station", "lead_h"],
        callbacks=[e.Deadline(begin + 3600)],
        verbose=False,
    )
    if single.tree_count_ != 700:
        raise TimeoutError("partial single model never promoted")
    single.save_model(out / "03_model/single.cbm")
    seconds_single = time.perf_counter() - t
    if time.perf_counter() - begin >= 3600:
        raise TimeoutError("one-hour full-fit budget")
    mx, my, current = e.multi_matrix(features, anchors, ids, columns)
    multi = CatBoostRegressor(**e.parameters(recipe, "multi", seed))
    t = time.perf_counter()
    multi.fit(
        mx,
        my,
        sample_weight=e.threshold_case_weights(current),
        cat_features=["station"],
        verbose=False,
    )
    multi.save_model(out / "03_model/multi.cbm")
    seconds_multi = time.perf_counter() - t
    router = e.ComponentLossRouter(e.RouterConfig(10.0, 2.0, 0.5, "smooth_medium")).fit(
        x_router, loss
    )
    joblib.dump(router, out / "03_model/router.joblib")
    selected = np.unique(np.linspace(0, len(features) - 1, 128, dtype=int))
    cases = features.iloc[selected].copy().reset_index(drop=True)
    cases.insert(0, "case_id", [f"INTERNAL_{i:04d}" for i in range(len(cases))])
    keys, prediction = predict_cases(e, cfg, cases, columns, (single, multi, router))
    cases.to_parquet(out / "04_logs/replay_cases.parquet", index=False)
    np.savez_compressed(out / "04_logs/replay_expected.npz", prediction=prediction)
    save(out / "04_logs/feature_columns.json", {"columns": columns})
    row = {
        "status": "FULL_TRAINING_COMPLETE",
        "pid": os.getpid(),
        "new_backbone_fits": 2,
        "new_router_fits": 1,
        "rows_single": len(x),
        "rows_multi": len(mx),
        "router_rows": len(x_router),
        "full_seed": seed,
        "single_seconds": seconds_single,
        "multi_seconds": seconds_multi,
        "seconds": time.perf_counter() - begin,
        "same_cycle_prior_oof_rows": len(component),
        "whole_cold_start": False,
        "empty_full_model_directory": True,
        "feature_count": len(columns),
        "numeric_lead": "removal" not in cfg,
        "router_hmax_removed": "removal" in cfg,
        "model_sha256": {p.name: sha(p) for p in (out / "03_model").iterdir()},
        "replay_sha256": {p.name: sha(p) for p in (out / "04_logs").iterdir()},
        "accepted_same_cycle": accepted,
        "source_sha256": cfg["source_files"],
        "runner_sha256": sha(__file__),
        "replay_rows": len(keys),
        "official_rows": 0,
        "sample_rows": 0,
        "hidden_rows": 0,
        "uploads": 0,
    }
    if row["seconds"] >= 3600 or multi.tree_count_ != 1200:
        raise TimeoutError("full-fit execution budget/iterations failure")
    save(out / "04_logs/training-result.json", row)
    print(
        json.dumps(
            {
                k: row[k]
                for k in ("status", "pid", "seconds", "new_backbone_fits", "new_router_fits")
            }
        ),
        flush=True,
    )


def load(e, cfg, out):
    row = read(out / "04_logs/training-result.json")
    if row["runner_sha256"] != sha(__file__) or row["accepted_same_cycle"] != verify_inputs(e, cfg):
        raise ValueError("full source/experiment provenance changed")
    for folder, field in (("03_model", "model_sha256"), ("04_logs", "replay_sha256")):
        for name, expected in row[field].items():
            if sha(out / folder / name) != expected:
                raise ValueError("saved full model/replay artifact changed")
    models = (
        CatBoostRegressor().load_model(out / "03_model/single.cbm"),
        CatBoostRegressor().load_model(out / "03_model/multi.cbm"),
        joblib.load(out / "03_model/router.joblib"),
    )
    return row, models, read(out / "04_logs/feature_columns.json")["columns"]


def replay(e, cfg, out):
    begin = time.perf_counter()
    row, models, columns = load(e, cfg, out)
    if row["pid"] == os.getpid():
        raise ValueError("new process required")
    cases = pd.read_parquet(out / "04_logs/replay_cases.parquet")
    _, prediction = predict_cases(e, cfg, cases, columns, models)
    with np.load(out / "04_logs/replay_expected.npz", allow_pickle=False) as expected:
        difference = float(np.max(np.abs(prediction - expected["prediction"])))
    if difference != 0:
        raise ValueError("saved full-model replay differs")
    save(
        out / "04_logs/fresh-process-replay.json",
        {
            "status": "PASS",
            "pid": os.getpid(),
            "training_pid": row["pid"],
            "rows": len(prediction),
            "max_abs_difference_m": difference,
            "seconds": time.perf_counter() - begin,
            "training_result_sha256": sha(out / "04_logs/training-result.json"),
            "official_rows": 0,
            "whole_cold_start": False,
        },
    )


def training_qa(e, cfg, out):
    row, models, columns = load(e, cfg, out)
    r = read(out / "04_logs/fresh-process-replay.json")
    checks = {
        "two_backbone_one_router": row["new_backbone_fits"] == 2 and row["new_router_fits"] == 1,
        "full_population": row["rows_single"] == 146160
        and row["rows_multi"] == 24360
        and row["router_rows"] == 103602,
        "full_seed": row["full_seed"] == 20260817,
        "iterations": models[0].tree_count_ == 700 and models[1].tree_count_ == 1200,
        "single_lead_type": models[0].get_cat_feature_indices()
        == ([0] if row["numeric_lead"] else [0, 1]),
        "multi_station_type": models[1].get_cat_feature_indices() == [0],
        "feature_count": len(columns) == (591 if row["numeric_lead"] else 527),
        "no_hmax_candidate_leak": row["numeric_lead"]
        or all(
            not name.startswith("hmax_") for model in models[:2] for name in model.feature_names_
        ),
        "no_hmax_router_leak": row["numeric_lead"]
        or not any(name.startswith("hmax_") for name in models[2].columns),
        "router_target_free": not {"target_hs", "truth", "label", "anomaly_type"}.intersection(
            models[2].columns
        ),
        "fresh_replay": r["status"] == "PASS"
        and r["pid"] != row["pid"]
        and r["max_abs_difference_m"] == 0
        and r["training_result_sha256"] == sha(out / "04_logs/training-result.json"),
        "hour_budget": row["seconds"] < 3600,
        "not_whole_cold_claim": row["whole_cold_start"] is False,
        "zero_official_before_qa": row["official_rows"]
        == row["sample_rows"]
        == row["hidden_rows"]
        == row["uploads"]
        == 0,
    }
    failed = [k for k, v in checks.items() if not v]
    save(
        out / "04_logs/training-independent-qa.json",
        {
            "status": "PASS" if not failed else "FAIL",
            "checks": checks,
            "checks_count": len(checks),
            "failed_checks": failed,
            "training_result_sha256": sha(out / "04_logs/training-result.json"),
            "whole_cold_retraining_package_validation": "PENDING_NOT_RUN",
        },
    )
    if failed:
        raise ValueError("full training QA failed: " + ",".join(failed))


def inference(e, cfg, out, source, *, verify_answer):
    start = time.perf_counter()
    row, models, columns = load(e, cfg, out)
    qa = read(out / "04_logs/training-independent-qa.json")
    if qa["status"] != "PASS" or qa["training_result_sha256"] != sha(
        out / "04_logs/training-result.json"
    ):
        raise ValueError("full independent QA required before official input")
    if not verify_answer:
        save(
            out / "INFERENCE_LOCK.json",
            {
                "pid": os.getpid(),
                "created_utc": datetime.now(UTC).isoformat(),
                "uploads_authorized": False,
            },
        )
    public_input_sha256 = {
        name: sha(source / name) for name in ("test_context.parquet", "test_index.csv")
    }
    context = pd.read_parquet(
        source / "test_context.parquet",
        columns=["case_id", "station", "step_minute", *e.BASE_COLUMNS, *e.DIRECTION_COLUMNS],
    )
    index = pd.read_csv(source / "test_index.csv", usecols=KEYS)
    if len(index) != 1200 or index.duplicated(KEYS).any() or len(context) != 57800:
        raise ValueError("official public key/context counts invalid")
    records = []
    for case_id, group in context.groupby("case_id", sort=False):
        group = group.sort_values("step_minute")
        if (
            len(group) != 289
            or group.station.nunique() != 1
            or not np.array_equal(group.step_minute, np.arange(-2880, 1, 10))
        ):
            raise ValueError("anonymous public case context grid invalid")
        records.append(
            {
                "case_id": case_id,
                "station": str(group.station.iloc[0]),
                **e.summarize_context(group),
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
    if len(cases) != 200 or not cases._merge.eq("both").all():
        raise ValueError("official case population invalid")
    keys, prediction = predict_cases(e, cfg, cases.drop(columns="_merge"), columns, models)
    keys["hs_pred"] = prediction
    frame = index.merge(keys, on=KEYS, how="left", validate="one_to_one")
    if (
        not frame[KEYS].equals(index[KEYS])
        or not np.isfinite(frame.hs_pred).all()
        or not frame.hs_pred.between(0, 30).all()
    ):
        raise ValueError("official output schema/key/order/finite invalid")
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, lineterminator="\n")
    payload = stream.getvalue().encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if public_input_sha256 != {name: sha(source / name) for name in public_input_sha256}:
        raise ValueError("public input changed during frozen inference")
    if verify_answer:
        previous = read(out / "04_logs/answer-qa.json")
        if (
            previous["pid"] == os.getpid()
            or previous["public_input_sha256"] != public_input_sha256
            or digest != previous["sha256"]
            or digest != sha(out / "05_answer/submission.csv")
        ):
            raise ValueError("new process CSV bytes differ")
    else:
        with (out / "05_answer/submission.csv").open("xb") as output:
            output.write(payload)
    receipt = {
        "status": "EXACT_ANSWER_REPLAY_PASS"
        if verify_answer
        else "LOCAL_CANDIDATE_READY_NOT_UPLOADED",
        "pid": os.getpid(),
        "training_pid": row["pid"],
        "rows": 1200,
        "cases": 200,
        "sha256": digest,
        "schema": [*KEYS, "hs_pred"],
        "key_order_exact": True,
        "finite_0_30": True,
        "duplicate_keys": 0,
        "seconds": time.perf_counter() - start,
        "source_training_result_sha256": sha(out / "04_logs/training-result.json"),
        "experiment_result_sha256": sha(e.REPORT / "result.json"),
        "official_context_rows": len(context),
        "official_index_rows": len(index),
        "public_input_sha256": public_input_sha256,
        "public_input_hash_scope": "public context/index whole bytes; parsed columns allowlisted",
        "sample_rows": 0,
        "hidden_rows": 0,
        "uploads": 0,
        "official_score": None,
        "whole_cold_start_package_validation": "PENDING_NOT_RUN",
    }
    save(
        out / "04_logs" / ("answer-replay-qa.json" if verify_answer else "answer-qa.json"), receipt
    )
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ("fit", "replay", "qa", "infer", "verify-answer"):
        group.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    e = importlib.import_module("run_" + args.experiment)
    cfg = read(e.CONFIG)
    out = ROOT / "artifacts" / (args.experiment + "_full")
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    guard(source, out, cfg, e, official=args.infer or args.verify_answer)
    try:
        if args.fit:
            fit(e, cfg, out)
        elif args.replay:
            replay(e, cfg, out)
        elif args.qa:
            training_qa(e, cfg, out)
        else:
            inference(e, cfg, out, source, verify_answer=args.verify_answer)
    except Exception as exc:
        if out.exists() and not (out / "FAILURE_RECEIPT.json").exists():
            save(
                out / "FAILURE_RECEIPT.json",
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "pid": os.getpid(),
                    "auto_restart": False,
                },
            )
        raise


if __name__ == "__main__":
    main()
