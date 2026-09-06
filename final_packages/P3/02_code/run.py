"""Source-only fixed-candidate full cold path. No old OOF/models/answers accepted."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import threading
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"
CODE = Path(__file__).resolve().parent
OUT = CODE.parent
WORK, MODELS, DOCS = OUT / "04_logs", OUT / "03_model", OUT / "06_docs"
sys.path[:0] = [str(CODE / "scripts"), str(CODE / "src")]

import joblib
import numpy as np
import p3_forward_candidate_materialize_20260906_v1 as materializer
import pandas as pd
from catboost import CatBoostRegressor

from p3_wave.features import build_training_features

MODULE_NAMES = {"numeric": "run_p3_numeric_lead_forward_gpu_20260906_v2",
                "hmax": "run_p3_hmax_removed_forward_20260906_v1"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=str)


def module(variant):
    e = importlib.import_module(MODULE_NAMES[variant])
    # Relocate outputs in the copied helper module, never the immutable source file.
    e.OUT, e.REPORT = MODELS / "historical", DOCS
    return e


def contract():
    manifest = read(CODE / "manifest.json")
    for name, expected in manifest.items():
        path = (CODE / name).resolve()
        if not path.is_relative_to(CODE) or sha(path) != expected:
            raise ValueError("portable source changed: " + name)
    c = read(CODE / "frozen.json")
    if c["budget"] != {"cpu_threads": 2, "exclusive_gpu_device": 0,
                       "backbone_fits": 12, "router_fits": 5, "total_seconds": 21600}:
        raise ValueError("fixed resource budget changed")
    postprocessing_contract(c["recipe"])
    if sha(DOCS / "shrink-provenance.json") != c["shrink_provenance_sha256"]:
        raise ValueError("fixed historical shrink provenance changed")
    return c


def postprocessing_contract(recipe):
    router = recipe["router"]
    actual = {key: router[key] for key in ("alpha", "temperature_multiplier", "strength", "name", "active_leads")}
    expected = {"alpha": 10.0, "temperature_multiplier": 2.0, "strength": 0.5,
                "name": "smooth_medium", "active_leads": [12, 18, 24]}
    if (actual != expected or recipe["shrink"] != {"active_leads": [12, 18, 24], "persistence_weight": 0.2}
            or recipe["model"]["single_weight"] != 0.5 or recipe["model"]["multi_weight"] != 0.5):
        raise ValueError("copied helper fixed postprocessing differs from recipe")
    return {"router": actual, "shrink": recipe["shrink"], "short_lead_weights": [0.5, 0.5, 0.0]}


def access_guard(source, official=False):
    allowed = {source / name for name in ("train_wave.csv", "train_atmos.csv")}
    if official:
        allowed |= {source / "test_context.parquet", source / "test_index.csv"}
    deny = os.environ.get("P3_DENY_REPO")

    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("network denied; not OS isolation")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1], args[2]
        writing = bool(isinstance(mode, str) and any(k in mode for k in "wax+")) or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if "external_data" in path.parts or any(k in path.name.lower() for k in
                ("hidden", "credentials", "refined_alpha", "axis_contract")):
            raise PermissionError("forbidden source ancestry")
        if path.is_relative_to(source) and (path not in allowed or writing):
            raise PermissionError("source is immutable and stage-allowlisted")
        if deny and path.is_relative_to(Path(deny)) and not path.is_relative_to(Path(sys.prefix)):
            raise PermissionError("original repository denied")
        if path.suffix.lower() in {".csv", ".parquet", ".npz", ".npy", ".cbm", ".joblib", ".pt", ".ckpt"}:
            if not path.is_relative_to(OUT) and path not in allowed:
                raise PermissionError("old data/model/OOF/answer inputs forbidden")
        if path.is_relative_to(OUT / "05_answer") and not official:
            raise PermissionError("answer access before official phase")
    sys.addaudithook(audit)


def watchdog(c, started):
    remaining = c["budget"]["total_seconds"] - (time.time() - started)
    if remaining <= 0:
        raise TimeoutError("whole cold six-hour clock exhausted; no restart")
    timer = threading.Timer(remaining, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    return timer


def policy_config(variant):
    return {"removal": {"base_prefix": "hmax_"}} if variant == "hmax" else {}


def chosen_columns(e, c, variant, columns):
    result = materializer.policy_columns(e, policy_config(variant), columns)
    if len(result) != c["variants"][variant]["features"]:
        raise ValueError("fixed variant column count mismatch")
    return result


def source_hashes(c, source):
    actual = {name: sha(source / name) for name in c["source_files"]}
    if actual != c["source_files"]:
        raise ValueError("organizer train bytes differ")
    return actual


def single_case_major(meta, residual):
    # expand_leads is lead-major; fit_one sorts before its case-major reshape.
    frame = meta[["anchor_id", "station", "lead_h"]].copy()
    frame["value"] = np.clip(meta.current_hs.to_numpy() + residual, 0, 30)
    if frame.duplicated(["anchor_id", "lead_h"]).any() or not np.isfinite(frame.value).all():
        raise ValueError("single replay key/finite mismatch")
    return frame.sort_values(["anchor_id", "lead_h"]).value.to_numpy().reshape(-1, 6)


def attach_episodes(e, anchors, wave):
    episodes = e.wave_anchors(wave[["station", "time", "hs"]])
    anchors = anchors.merge(episodes, on=["station", "anchor_time"], how="outer",
                            validate="one_to_one", indicator=True)
    if len(anchors) != 24360 or not anchors._merge.eq("both").all():
        raise ValueError("raw episode/anchor support mismatch")
    return anchors.drop(columns="_merge").sort_values("anchor_id").reset_index(drop=True)


def support_checks(e, c, wave, features, anchors, columns, evaluation):
    lookup = wave.set_index(["station", "time"]).hs
    for lead in (0, *e.LEADS):
        keys = pd.MultiIndex.from_arrays([anchors.station, anchors.anchor_time + pd.Timedelta(hours=lead)])
        target = lookup.reindex(keys).to_numpy()
        actual = anchors.current_hs.to_numpy() if lead == 0 else anchors[f"target_{lead}"].to_numpy()
        if not np.isfinite(target).all() or not np.array_equal(target, actual):
            raise ValueError("current/six-target alignment mismatch")
    if not features[["anchor_id", "station"]].equals(anchors[["anchor_id", "station"]]):
        raise ValueError("fresh feature/anchor order mismatch")
    rows, probes = [], set()
    for i, (fold, masks) in enumerate(e.fold_masks(anchors, evaluation)):
        train, valid = anchors.loc[masks["train"]], anchors.loc[masks["validation"]]
        if (len(train), len(valid)) != (c["expected_train"][i], c["expected_validation"][i]):
            raise ValueError("fixed v5 fold support changed")
        rows.append({"fold": fold["id"], "train": len(train), "validation": len(valid)})
        for _, part in valid.groupby("station"):
            probes.update(part.sort_values("anchor_time").iloc[[0, len(part) // 2, -1]].anchor_id)
    if len(columns) != 591 or np.isinf(features[columns].to_numpy()).any():
        raise ValueError("feature count/infinite value mismatch")
    return rows, probes


def prepare(c, source):
    if any(MODELS.iterdir()) or any(WORK.iterdir()) or any((OUT / "05_answer").iterdir()):
        raise FileExistsError("fresh empty models/work/answers required")
    started = time.time()
    save(OUT / "PREPARE_LOCK.json", {"pid": os.getpid(), "started_unix": started,
                                   "manifest_sha256": sha(CODE / "manifest.json"), "fits": 0})
    timer = watchdog(c, started)
    source_hashes(c, source)
    e = module("numeric")  # Shared prepare has no lead-type or hmax hypothesis.
    wave, atmos = [pd.read_csv(source / name) for name in ("train_wave.csv", "train_atmos.csv")]
    for frame, count in ((wave, 118152), (atmos, 130896)):
        frame.time = pd.to_datetime(frame.time, utc=True)
        if len(frame) != count or frame.duplicated(["station", "time"]).any():
            raise ValueError("train count/key mismatch")
    data = e.P3Data(wave, atmos, *[pd.DataFrame() for _ in range(4)])
    def progress(done, total):
        if done % 3000 == 0 or done == total:
            print(json.dumps({"stage": "SOURCE_FEATURES", "done": done, "total": total,
                              "fits": 0, "seconds": time.time() - started}), flush=True)
    built = build_training_features(data, dense_spacing_minutes=20, progress=progress)
    columns = e.compact_feature_columns(list(built.feature_columns))
    anchors = attach_episodes(e, built.anchors, wave)
    evaluation = read(CODE / "configs/evaluation/ocean_forward_v5.json")
    rows, probes = support_checks(e, c, wave, built.features, anchors, columns, evaluation)
    grid = e.build_training_grid(data)
    grids = {name: part for name, part in grid.groupby("station")}
    indexed = built.features.set_index("anchor_id")
    for row in anchors.loc[anchors.anchor_id.isin(probes)].itertuples(index=False):
        actual = e.summarize_context(e.raw_context(grids[row.station], row))
        if not np.allclose([actual[k] for k in columns], indexed.loc[row.anchor_id, columns].to_numpy(float),
                           rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError("raw case context feature check failed")
    built.features.to_parquet(WORK / "features.parquet", index=False)
    anchors.to_parquet(WORK / "anchors.parquet", index=False)
    save(WORK / "columns.json", {"columns": columns})
    result = {"status": "PREPARED_SOURCE_ONLY", "pid": os.getpid(), "started_unix": started,
              "seconds": time.time() - started, "anchors": len(anchors), "features": len(columns),
              "folds": rows, "raw_context_checks": len(probes), "source_sha256": source_hashes(c, source),
              "train_sources_unchanged_before_after": True,
              "manifest_sha256": sha(CODE / "manifest.json"), "fits": 0, "old_cache_reads": 0,
              "official_rows": 0, "files": {p.name: sha(p) for p in WORK.iterdir() if p.is_file()},
              "empty_03_model": not any(MODELS.iterdir()), "variant_selected": False}
    save(DOCS / "prepare.json", result)
    timer.cancel()
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "source_sha256", "folds")}), flush=True)


def prepared(c):
    p = read(DOCS / "prepare.json")
    if p["status"] != "PREPARED_SOURCE_ONLY" or p["manifest_sha256"] != sha(CODE / "manifest.json"):
        raise ValueError("prepare/source seal mismatch")
    for name, expected in p["files"].items():
        if sha(WORK / name) != expected:
            raise ValueError("own source-generated cache changed")
    return p, pd.read_parquet(WORK / "features.parquet"), pd.read_parquet(WORK / "anchors.parquet"), read(WORK / "columns.json")["columns"]


def full_fit(e, c, variant, features, anchors, columns, component, deadline):
    recipe = c["recipe"]
    seed = recipe["model"]["full_train_seed"]
    ids = anchors.anchor_id.to_numpy(int)
    x, y, meta = e.expand_leads(features, anchors, ids, columns)
    if len(x) != 146160 or len(ids) != 24360:
        raise ValueError("full training population changed")
    single = CatBoostRegressor(**e.parameters(recipe, "single", seed))
    single.fit(e.matrix(x, variant == "numeric"), y,
               sample_weight=e.threshold_case_weights(meta.current_hs.to_numpy()),
               cat_features=["station"] if variant == "numeric" else ["station", "lead_h"],
               callbacks=[e.Deadline(deadline)], verbose=False)
    if single.tree_count_ != 700:
        raise TimeoutError("partial full single not accepted")
    single.save_model(MODELS / "full_single.cbm")
    mx, my, current = e.multi_matrix(features, anchors, ids, columns)
    multi = CatBoostRegressor(**e.parameters(recipe, "multi", seed))
    multi.fit(mx, my, sample_weight=e.threshold_case_weights(current), cat_features=["station"], verbose=False)
    if multi.tree_count_ != 1200 or time.perf_counter() >= deadline:
        raise TimeoutError("full multi iterations/time mismatch")
    multi.save_model(MODELS / "full_multi.cbm")
    router_x, router_meta, _, losses = e.router_material(component, features)
    if len(router_x) != 103602 or router_meta.anchor_id.nunique() != 17267:
        raise ValueError("new full-router OOF population changed")
    router = e.ComponentLossRouter(e.RouterConfig(10.0, 2.0, 0.5, "smooth_medium")).fit(router_x, losses)
    joblib.dump(router, MODELS / "full_router.joblib")
    cases = features.iloc[np.unique(np.linspace(0, len(features)-1, 128, dtype=int))].copy().reset_index(drop=True)
    cases.insert(0, "case_id", [f"INTERNAL_{i:04d}" for i in range(len(cases))])
    _, probe = materializer.predict_cases(e, policy_config(variant), cases, columns, (single, multi, router))
    cases.to_parquet(WORK / "full_probe_cases.parquet", index=False)
    np.save(WORK / "full_probe.npy", probe, allow_pickle=False)


def train(c, variant):
    p, features, anchors, original_columns = prepared(c)
    if any(MODELS.iterdir()):
        raise FileExistsError("empty 03_model before first fit required")
    save(OUT / "TRAIN_LOCK.json", {"variant": variant, "pid": os.getpid(), "max_backbone": 12,
                                 "max_router": 5, "manifest_sha256": sha(CODE / "manifest.json")})
    timer = watchdog(c, p["started_unix"])
    deadline = time.perf_counter() + 21600 - (time.time() - p["started_unix"])
    e = module(variant)
    columns = chosen_columns(e, c, variant, original_columns)
    evaluation = read(CODE / "configs/evaluation/ocean_forward_v5.json")
    blocks, fits = [], []
    for number, (fold, split) in enumerate(e.fold_masks(anchors, evaluation)):
        pair = []
        for kind in (c["variants"][variant]["single_kind"], "multi"):
            prediction, receipt = e.fit_one(kind, fold, number, split, features, anchors,
                                           columns, c["recipe"], c, deadline)
            pair.append(prediction)
            fits.append(receipt)
            print(json.dumps({"stage": "HISTORICAL", "completed_backbone_fits": len(fits),
                              "maximum_backbone_fits": 12, "seconds": time.time()-p["started_unix"]}), flush=True)
        blocks.append(e.component_frame(anchors, split["validation"], fold["id"], *pair))
    component = pd.concat(blocks, ignore_index=True)
    if len(component) != c["expected_oof_rows"] or component.duplicated(e.KEYS).any():
        raise ValueError("new OOF population mismatch")
    final, routers = e.fixed_policy(component, features, anchors, evaluation, arm="candidate", fit=True)
    if sum(r["model_path"] is not None for r in routers) != 4:
        raise ValueError("historical router fit count mismatch")
    final.to_parquet(WORK / "candidate_oof.parquet", index=False)
    full_fit(e, c, variant, features, anchors, columns, component, deadline)
    result = {"status": "TWELVE_BACKBONE_FIVE_ROUTER_COMPLETE", "pid": os.getpid(),
              "variant": variant, "new_backbone_fits": 12, "new_router_fits": 5,
              "historical_fits": fits, "historical_routers": routers,
              "oof_rows": len(final), "internal_rmse_m": float(np.sqrt(np.square(final.target_hs-final.final_prediction).mean())),
              "source_sha256": c["source_files"], "manifest_sha256": sha(CODE / "manifest.json"),
              "postprocessing_contract": postprocessing_contract(c["recipe"]),
              "prepare_sha256": sha(DOCS / "prepare.json"), "elapsed_seconds": time.time()-p["started_unix"],
              "models": {x.relative_to(OUT).as_posix(): sha(x) for x in MODELS.rglob("*") if x.is_file()},
              "work": {x.name: sha(x) for x in WORK.iterdir() if x.is_file()},
              "old_oof_model_answer_reads": 0, "official_rows": 0, "whole_cold_qa": "PENDING"}
    save(DOCS / "training-result.json", result)
    timer.cancel()
    print(json.dumps({k: result[k] for k in ("status", "variant", "new_backbone_fits", "new_router_fits", "elapsed_seconds")}), flush=True)


def trained(c):
    p, features, anchors, columns = prepared(c)
    r = read(DOCS / "training-result.json")
    if (r["status"] != "TWELVE_BACKBONE_FIVE_ROUTER_COMPLETE" or r["pid"] == os.getpid()
            or r["manifest_sha256"] != sha(CODE / "manifest.json")
            or r["prepare_sha256"] != sha(DOCS / "prepare.json")):
        raise ValueError("complete sealed training in another PID required")
    for name, digest in r["models"].items():
        if sha(OUT / name) != digest:
            raise ValueError("trained model changed")
    if (sum(name.endswith('.cbm') for name in r['models']) != 12
            or sum(name.endswith('.joblib') for name in r['models']) != 5
            or (r['new_backbone_fits'], r['new_router_fits']) != (12, 5)):
        raise ValueError("actual twelve-backbone/five-router artifact count mismatch")
    for name, digest in r["work"].items():
        if sha(WORK / name) != digest:
            raise ValueError("own training/OOF work changed")
    e = module(r["variant"])
    return p, r, e, features, anchors, chosen_columns(e, c, r["variant"], columns)


def full_models():
    return (CatBoostRegressor().load_model(MODELS / "full_single.cbm"),
            CatBoostRegressor().load_model(MODELS / "full_multi.cbm"),
            joblib.load(MODELS / "full_router.joblib"))


def qa(c, source):
    p, r, e, features, anchors, columns = trained(c)
    timer = watchdog(c, p["started_unix"])
    evaluation = read(CODE / "configs/evaluation/ocean_forward_v5.json")
    blocks = []
    for fold, split in e.fold_masks(anchors, evaluation):
        ids = anchors.loc[split["validation"], "anchor_id"].to_numpy(int)
        pair = []
        for kind in (c["variants"][r["variant"]]["single_kind"], "multi"):
            model = CatBoostRegressor().load_model(e.OUT / "models" / fold["id"] / (kind + ".cbm"))
            expected_cats = [0] if kind == 'multi' or r['variant'] == 'numeric' else [0, 1]
            if (model.tree_count_ != (1200 if kind == 'multi' else 700)
                    or model.get_cat_feature_indices() != expected_cats):
                raise ValueError("historical iterations/native categorical contract differs")
            if kind == "multi":
                x, _, current = e.multi_matrix(features, anchors, ids, columns)
                prediction = np.clip(current[:, None] + model.predict(x, thread_count=2), 0, 30)
            else:
                x, _, meta = e.expand_leads(features, anchors, ids, columns)
                residual = model.predict(e.matrix(x, r["variant"] == "numeric"), thread_count=2)
                prediction = single_case_major(meta, residual)
            pair.append(prediction)
        blocks.append(e.component_frame(anchors, split["validation"], fold["id"], *pair))
    fresh, routers = e.fixed_policy(pd.concat(blocks, ignore_index=True), features, anchors,
        evaluation, arm="candidate", fit=False, expected_receipts=r["historical_routers"])
    stored = pd.read_parquet(WORK / "candidate_oof.parquet")
    if not fresh[e.KEYS].equals(stored[e.KEYS]) or not np.array_equal(fresh.final_prediction, stored.final_prediction):
        raise ValueError("whole historical OOF replay differs")
    cases = pd.read_parquet(WORK / "full_probe_cases.parquet")
    _, probe = materializer.predict_cases(e, policy_config(r["variant"]), cases, columns, full_models())
    if not np.array_equal(probe, np.load(WORK / "full_probe.npy", allow_pickle=False)):
        raise ValueError("full-model probe replay differs")
    error = fresh.target_hs.to_numpy()-fresh.final_prediction.to_numpy()
    rmse = float(np.sqrt(np.dot(error, error)/len(error)))
    if abs(rmse-r["internal_rmse_m"]) > 1e-12:
        raise ValueError("independent pooled SSE/N mismatch")
    receipt = {"status": "PASS", "pid": os.getpid(), "training_pid": r["pid"],
               "train_source_sha256_rechecked": source_hashes(c, source),
               "training_result_sha256": sha(DOCS / "training-result.json"),
               "historical_models_replayed": 10, "prior_routers_replayed": 4,
               "full_models_replayed": 2, "full_router_replayed": 1,
               "oof_keys_probabilities_exact": True, "full_probe_exact": True,
               "independent_pooled_rmse_m": rmse, "rows": len(error),
               "full_probe_quality_calculated": False, "new_fits": 0, "official_rows": 0,
               "elapsed_seconds": time.time()-p["started_unix"], "fresh_OS_verified": False}
    save(DOCS / "training-qa.json", receipt)
    timer.cancel()
    print(json.dumps(receipt), flush=True)


def infer(c, source, verify=False):
    p, r, e, _, _, columns = trained(c)
    q = read(DOCS / "training-qa.json")
    if q["status"] != "PASS" or q["pid"] == r["pid"] or q["training_result_sha256"] != sha(DOCS / "training-result.json"):
        raise ValueError("fresh training QA before official reads required")
    if not verify:
        save(OUT / "INFERENCE_LOCK.json", {"pid": os.getpid(), "uploads": 0,
                                          "training_qa_sha256": sha(DOCS / "training-qa.json")})
    timer = watchdog(c, p["started_unix"])
    hashes = {name: sha(source / name) for name in ("test_context.parquet", "test_index.csv")}
    context = pd.read_parquet(source / "test_context.parquet", columns=["case_id", "station", "step_minute", *e.BASE_COLUMNS, *e.DIRECTION_COLUMNS])
    index = pd.read_csv(source / "test_index.csv", usecols=materializer.KEYS)
    if len(context) != 57800 or len(index) != 1200 or index.duplicated(materializer.KEYS).any():
        raise ValueError("official count/unique keys mismatch")
    records = []
    for case_id, group in context.groupby("case_id", sort=False):
        group = group.sort_values("step_minute")
        if len(group) != 289 or group.station.nunique() != 1 or not np.array_equal(group.step_minute, np.arange(-2880, 1, 10)):
            raise ValueError("anonymous case context grid mismatch")
        records.append({"case_id": case_id, "station": str(group.station.iloc[0]), **e.summarize_context(group)})
    cases = index[["case_id", "station"]].drop_duplicates().merge(pd.DataFrame(records),
        on=["case_id", "station"], how="outer", validate="one_to_one", indicator=True)
    if len(cases) != 200 or not cases._merge.eq("both").all():
        raise ValueError("case set mismatch")
    keys, prediction = materializer.predict_cases(e, policy_config(r["variant"]), cases.drop(columns="_merge"), columns, full_models())
    keys["hs_pred"] = prediction
    answer = index.merge(keys, on=materializer.KEYS, how="left", validate="one_to_one")
    if not answer[materializer.KEYS].equals(index) or not np.isfinite(answer.hs_pred).all() or not answer.hs_pred.between(0, 30).all():
        raise ValueError("answer schema/keys/order/finite/range mismatch")
    if hashes != {name: sha(source / name) for name in hashes}:
        raise ValueError("public input changed during inference")
    payload = answer.to_csv(index=False, lineterminator="\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    if verify:
        previous = read(DOCS / "answer-qa.json")
        if previous["pid"] == os.getpid() or previous["sha256"] != digest or previous["public_input_sha256"] != hashes or sha(OUT / "05_answer/submission.csv") != digest:
            raise ValueError("fresh answer bytes/input hashes differ")
    else:
        with (OUT / "05_answer/submission.csv").open("xb") as stream:
            stream.write(payload)
    receipt = {"status": "EXACT_ANSWER_REPLAY_PASS" if verify else "LOCAL_ANSWER_NOT_UPLOADED",
               "pid": os.getpid(), "training_pid": r["pid"], "sha256": digest, "rows": 1200,
               "schema": [*materializer.KEYS, "hs_pred"], "public_input_sha256": hashes,
               "training_result_sha256": sha(DOCS / "training-result.json"),
               "training_qa_sha256": sha(DOCS / "training-qa.json"),
               "elapsed_seconds": time.time()-p["started_unix"], "old_answer_values_read": 0,
               "sample_rows": 0, "hidden_rows": 0, "uploads": 0, "new_fits": 0,
               "new_OS_offline_verified": False, "whole_training_runs": 1}
    save(DOCS / ("answer-replay-qa.json" if verify else "answer-qa.json"), receipt)
    timer.cancel()
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train", "qa", "infer", "verify-answer"))
    parser.add_argument("--prepare-approved", action="store_true")
    parser.add_argument("--training-approved", action="store_true")
    parser.add_argument("--gpu-approved", action="store_true")
    parser.add_argument("--official-approved", action="store_true")
    parser.add_argument("--variant", choices=tuple(MODULE_NAMES))
    args = parser.parse_args()
    if args.stage == "prepare" and not args.prepare_approved:
        parser.error("separate source-only prepare authorization required")
    if args.stage == "train" and not (args.training_approved and args.gpu_approved and args.variant):
        parser.error("explicit variant, training and exclusive GPU authorization required")
    if args.stage in ("infer", "verify-answer") and not args.official_approved:
        parser.error("separate official input authorization required")
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    access_guard(source, args.stage in ("infer", "verify-answer"))
    c = contract()
    try:
        if args.stage == "prepare":
            prepare(c, source)
        elif args.stage == "train":
            train(c, args.variant)
        elif args.stage == "qa":
            qa(c, source)
        else:
            infer(c, source, args.stage == "verify-answer")
    except Exception as exc:
        target = DOCS / ("failure_"+args.stage+".json")
        if not target.exists():
            save(target, {"status": "FAIL", "type": type(exc).__name__, "message": str(exc),
                          "pid": os.getpid(), "restart": False})
        raise


if __name__ == "__main__":
    main()
