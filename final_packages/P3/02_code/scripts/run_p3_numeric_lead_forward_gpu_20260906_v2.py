"""Frozen mixed-device lead representation ablation on the v5 forward surface.

No official data, previous predictions/models, full-fit or answer branch exists.
Exact source-derived feature cache reuse is distinct from OOF/model reuse.
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

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import ocean_evaluation_contract_v5 as cv  # noqa: E402
import pandas as pd  # noqa: E402
from audit_ocean_forward_support_20260906_v1 import wave_anchors  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

from p3_wave.data import P3Data, build_training_grid  # noqa: E402
from p3_wave.features import BASE_COLUMNS, DIRECTION_COLUMNS, summarize_context  # noqa: E402
from p3_wave.loss_router import (  # noqa: E402
    OBSERVED_FEATURES,
    ComponentLossRouter,
    RouterConfig,
    build_inference_router_features,
    expand_case_router_rows,
    route_row_predictions,
)
from p3_wave.models import compact_feature_columns, threshold_case_weights  # noqa: E402
from p3_wave.persistence_shrink import (  # noqa: E402
    LongLeadPersistenceShrink,
    apply_long_lead_persistence_shrink,
)
from p3_wave.validation import expand_leads  # noqa: E402

NAME = "p3_numeric_lead_forward_gpu_20260906_v2"
CONFIG = ROOT / "configs/experiments" / f"{NAME}.json"
OUT = ROOT / "artifacts" / NAME
REPORT = ROOT / "reports" / NAME
LEADS = (3, 6, 9, 12, 18, 24)
KEYS = ["fold", "anchor_id", "station", "lead_h"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data, *, progress=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if progress else "x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False, default=str)


def now():
    return datetime.now(UTC).isoformat()


def zero_access():
    return {
        name: 0
        for name in (
            "official_rows",
            "sample_rows",
            "hidden_rows",
            "csv_rows",
            "uploads",
            "external_rows",
            "old_model_reads",
            "old_prediction_reads",
            "full_fits",
        )
    }


def access_guard(source, config):
    allowed = {(ROOT / path).resolve() for path in config["inputs"]}
    source_files = {source / name for name in config["source_files"]}

    def audit(event, args):
        if event == "socket.connect":
            raise PermissionError("network forbidden")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode = args[1]
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        )
        owned = OUT in path.parents or REPORT in path.parents
        if "external_data" in path.parts or "hidden" in path.name.lower():
            raise PermissionError("external/hidden file forbidden")
        if source in path.parents and (path not in source_files or writing):
            raise PermissionError("only immutable distributed training inputs allowed")
        if path.suffix.lower() == ".csv" and path not in source_files:
            raise PermissionError("CSV read/write forbidden")
        if path in allowed and writing:
            raise PermissionError("pinned input immutable")
        if (
            path.suffix.lower()
            in {".parquet", ".npz", ".cbm", ".joblib", ".pt", ".ckpt", ".tabpfn_fit"}
            and not owned
            and path not in allowed
        ):
            raise PermissionError("old model/prediction or unapproved data forbidden")

    sys.addaudithook(audit)


def verify_pins(config, source, *, sealed=False):
    verified = {}
    for name, expected in config["inputs"].items():
        actual = sha(ROOT / name)
        if actual != expected:
            raise ValueError(f"input hash mismatch: {name}")
        verified[name] = actual
    for name, expected in config["source_files"].items():
        if sha(source / name) != expected:
            raise ValueError("source train hash mismatch")
    if sealed:
        seal = json.loads((OUT / "seal.json").read_text())
        if seal["runner_sha256"] != sha(__file__) or seal["config_sha256"] != sha(CONFIG):
            raise ValueError("sealed runner/config changed")
    return verified


def load_cache(config):
    cache = ROOT / config["cache"]
    feature = pd.read_parquet(cache / "train_features.parquet")
    anchor = pd.read_parquet(cache / "train_anchors.parquet")
    columns = json.loads((cache / "feature_columns.json").read_text())["columns"]
    if len(columns) != 591 or columns != compact_feature_columns(feature):
        raise ValueError("fixed compact feature allowlist mismatch")
    if len(anchor) != 24360 or not feature[["anchor_id", "station"]].equals(
        anchor[["anchor_id", "station"]]
    ):
        raise ValueError("cache alignment/population mismatch")
    if any(c in columns for c in ("anchor_time", "time", "target_hs", "episode_id")) or any(
        c.startswith("target_") for c in columns
    ):
        raise ValueError("target/time/episode feature leak")
    if np.isinf(feature[columns].to_numpy()).any():
        raise ValueError("infinite feature")
    anchor.anchor_time = pd.to_datetime(anchor.anchor_time, utc=True)
    return feature, anchor, columns


def matrix(frame, numeric):
    if list(frame.columns[:3]) != ["station", "lead_h", "current_hs_for_residual"]:
        raise ValueError("single feature schema differs")
    if not frame.lead_h.isin(LEADS).all():
        raise ValueError("unexpected lead")
    result = frame.copy()
    result.station = result.station.astype(str)
    result.lead_h = result.lead_h.astype(float if numeric else str)
    return result


def parameters(recipe, kind, seed, *, synthetic=False):
    params = dict(recipe["model"][kind])
    params.pop("devices", None)
    params.update(
        task_type="GPU" if kind == "multi" else "CPU",
        thread_count=2,
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
    )
    if kind == "multi":
        params.update(devices="0", boosting_type="Plain")
    if synthetic:
        params.update(iterations=3, depth=2)
    return params


def raw_context(grid, anchor):
    time_value = pd.Timestamp(anchor.anchor_time)
    frame = grid.loc[
        (grid.station == anchor.station)
        & grid.time.between(time_value - pd.Timedelta(hours=48), time_value)
    ].sort_values("time")
    expected = pd.date_range(time_value - pd.Timedelta(hours=48), time_value, freq="10min")
    if len(frame) != 289 or not np.array_equal(frame.time.to_numpy(), expected.to_numpy()):
        raise ValueError("48h context grid not exact")
    return frame


def fold_masks(anchors, contract):
    return [
        (fold, cv.p3_split(anchors, fold["id"], contract))
        for fold in contract["P3"]["folds"]
        if not fold["warmup"]
    ]


def safe_meta_mask(row_meta, anchors, current_fold, contract):
    valid_base_ids = anchors.loc[cv.p3_split(anchors, current_fold, contract)["train"], "anchor_id"]
    return row_meta.anchor_id.isin(valid_base_ids).to_numpy()


def preflight(config, source):
    if OUT.exists():
        raise FileExistsError("preexisting isolated output preserved")
    begin = time.perf_counter()
    verified = verify_pins(config, source)
    contract = cv.load_contract(ROOT / config["contract"])
    support = json.loads((ROOT / config["source_support"]).read_text())["P3"]
    prepare = json.loads((ROOT / config["cache_prepare"]).read_text())
    if (
        prepare["source_sha256"] != config["source_files"]
        or prepare["old_cache_reads"] != 0
        or prepare["official_rows"] != 0
    ):
        raise ValueError("cache source-only provenance mismatch")
    for filename in ("train_features.parquet", "train_anchors.parquet", "feature_columns.json"):
        if prepare["files"][filename] != config["inputs"][f"{config['cache']}/{filename}"]:
            raise ValueError("feature hash not linked to fresh prepare")
    feature, anchors, columns = load_cache(config)
    wave = pd.read_csv(
        source / "train_wave.csv", usecols=["station", "time", "hs", "tp", "hmax", "wvdir"]
    )
    atmos = pd.read_csv(
        source / "train_atmos.csv",
        usecols=["station", "time", "wspd", "gust", "wdir", "airt", "relh", "caph"],
    )
    for frame, count in ((wave, 118152), (atmos, 130896)):
        frame.time = pd.to_datetime(frame.time, utc=True)
        if len(frame) != count or frame.duplicated(["station", "time"]).any():
            raise ValueError("source train schema/key/count mismatch")
    episodes = wave_anchors(wave[["station", "time", "hs"]])
    anchors = anchors.merge(
        episodes, on=["station", "anchor_time"], validate="one_to_one", how="outer", indicator=True
    )
    if not anchors._merge.eq("both").all() or len(anchors) != 24360:
        raise ValueError("raw20min high-run episode anchor mismatch")
    anchors = anchors.drop(columns="_merge").sort_values("anchor_id").reset_index(drop=True)
    lookup = wave.set_index(["station", "time"]).hs
    for lead in (0, *LEADS):
        target_keys = pd.MultiIndex.from_arrays(
            [anchors.station, anchors.anchor_time + pd.Timedelta(hours=lead)]
        )
        truth = lookup.reindex(target_keys).to_numpy()
        saved = anchors.current_hs.to_numpy() if lead == 0 else anchors[f"target_{lead}"].to_numpy()
        if not np.isfinite(truth).all() or not np.array_equal(truth, saved):
            raise ValueError("source target/current alignment mismatch")
    masks = fold_masks(anchors, contract)
    rows, sample_ids, oof_ids = [], set(), []
    for number, (fold, split) in enumerate(masks):
        train_ids = anchors.loc[split["train"], "anchor_id"].to_numpy()
        valid = anchors.loc[split["validation"]]
        if (
            len(train_ids) != config["expected_train"][number]
            or len(valid) != config["expected_validation"][number]
        ):
            raise ValueError("support fold count mismatch")
        if support["folds"][number]["fold"] != fold["id"] or support["folds"][number][
            "validation_anchors"
        ] != len(valid):
            raise ValueError("source support receipt mismatch")
        prior = anchors.anchor_id.isin(oof_ids) & split["train"]
        rows.append(
            {
                "fold": fold["id"],
                "train_anchors": len(train_ids),
                "validation_anchors": len(valid),
                "meta_prior_safe_anchors": int(prior.sum()),
                "prior_anchors_excluded_by_78h_episode_rule": len(oof_ids) - int(prior.sum()),
                "station_episodes": len(set(cv.bootstrap_groups(valid, "P3", contract))),
            }
        )
        oof_ids.extend(valid.anchor_id.to_list())
        for _, group in valid.groupby("station"):
            ordered = group.sort_values("anchor_time")
            sample_ids.update(ordered.iloc[[0, len(ordered) // 2, -1]].anchor_id.to_list())
    data = P3Data(wave, atmos, *[pd.DataFrame() for _ in range(4)])
    grid = build_training_grid(data)
    station_grids = {name: group for name, group in grid.groupby("station")}
    feature_lookup = feature.set_index("anchor_id")
    maximum_error = 0.0
    for row in anchors.loc[anchors.anchor_id.isin(sample_ids)].itertuples(index=False):
        context = raw_context(station_grids[row.station], row)
        recomputed = summarize_context(context)
        values = np.array([recomputed[c] for c in columns])
        stored = feature_lookup.loc[row.anchor_id, columns].to_numpy(float)
        if not np.allclose(values, stored, rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError("raw context cached feature mismatch")
        finite = np.isfinite(values) & np.isfinite(stored)
        maximum_error = max(maximum_error, float(np.max(np.abs(values[finite] - stored[finite]))))
    # Target changes outside the anchor's context cannot change case-local inputs.
    example = anchors.loc[anchors.anchor_id.isin(sample_ids)].iloc[0]
    example_grid = station_grids[example.station].copy()
    before = summarize_context(raw_context(example_grid, example))
    outside = (example_grid.time > example.anchor_time) | (
        example_grid.time < example.anchor_time - pd.Timedelta(hours=48)
    )
    example_grid.loc[outside, [*BASE_COLUMNS, *DIRECTION_COLUMNS]] = -9876.0
    after = summarize_context(raw_context(example_grid, example))
    if not np.allclose(list(before.values()), list(after.values()), rtol=0, atol=0, equal_nan=True):
        raise ValueError("outside-context future data leaked")
    recipe = json.loads((ROOT / config["recipe"]).read_text())
    if (
        recipe["router"]["alpha"] != 10
        or recipe["shrink"]["persistence_weight"] != 0.2
        or recipe["model"]["single_weight"] != 0.5
        or recipe["model"]["multi_weight"] != 0.5
    ):
        raise ValueError("fixed clean postprocessing recipe differs")
    OUT.mkdir(parents=True)
    anchors.to_parquet(OUT / "anchors.parquet", index=False)
    receipt = {
        "status": "FEATURE_AND_TARGET_SUPPORT_PASS",
        "seconds": time.perf_counter() - begin,
        "source_rows": {"wave": len(wave), "atmos": len(atmos)},
        "anchors": len(anchors),
        "features": len(columns),
        "validation_anchors": len(oof_ids),
        "validation_rows": len(oof_ids) * 6,
        "folds": rows,
        "raw_context_recomputed_cases": len(sample_ids),
        "raw_context_max_feature_error": maximum_error,
        "all_anchor_six_targets_source_recomputed": True,
        "future_outside_context_perturbation_no_effect": True,
        "episode_source": "raw20min high hs>=1.5 runs; missing/low/gap break; target eligibility does not split runs",
        "dense_validation_not_sparse_official_case_distribution": True,
        "onset": "NOT_ENABLED",
        "cpu_threads": 2,
        "gpu": 1,
        "verified_inputs": verified,
        "anchors_sha256": sha(OUT / "anchors.parquet"),
        **zero_access(),
    }
    write_json(REPORT / "preflight.json", receipt)
    write_json(
        OUT / "seal.json",
        {
            "runner_sha256": sha(__file__),
            "config_sha256": sha(CONFIG),
            "preflight_sha256": sha(REPORT / "preflight.json"),
            "created_utc": now(),
        },
    )
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key not in {"verified_inputs"}}
        ),
        flush=True,
    )


class Deadline:
    def __init__(self, end):
        self.end = end

    def after_iteration(self, info):
        return time.perf_counter() < self.end


def multi_matrix(features, anchors, ids, columns):
    frame = features.set_index("anchor_id").loc[ids, ["station", *columns]].reset_index(drop=True)
    frame.station = frame.station.astype(str)
    lookup = anchors.set_index("anchor_id").loc[ids]
    target = (
        lookup[[f"target_{lead}" for lead in LEADS]].to_numpy(float)
        - lookup.current_hs.to_numpy()[:, None]
    )
    return frame, target, lookup.current_hs.to_numpy()


def fit_one(kind, fold, number, split, features, anchors, columns, recipe, config, deadline):
    started = time.perf_counter()
    ids = anchors.loc[split["train"], "anchor_id"].to_numpy(int)
    valid_ids = anchors.loc[split["validation"], "anchor_id"].to_numpy(int)
    model_kind = "multi" if kind == "multi" else "single"
    params = parameters(recipe, model_kind, config["fold_seeds"][number])
    model = CatBoostRegressor(**params)
    if model_kind == "single":
        x, y, meta = expand_leads(features, anchors, ids, columns)
        vx, _, vm = expand_leads(features, anchors, valid_ids, columns)
        numeric = kind == "numeric_single"
        x, vx = matrix(x, numeric), matrix(vx, numeric)
        current, cats = (
            meta.current_hs.to_numpy(),
            ["station"] if numeric else ["station", "lead_h"],
        )
    else:
        x, y, current = multi_matrix(features, anchors, ids, columns)
        vx, _, valid_current = multi_matrix(features, anchors, valid_ids, columns)
        cats = ["station"]
    if time.perf_counter() >= deadline:
        raise TimeoutError("fixed execution budget expired before fit")
    fit_options = {"callbacks": [Deadline(deadline)]} if params["task_type"] == "CPU" else {}
    model.fit(
        x,
        y,
        sample_weight=threshold_case_weights(current),
        cat_features=cats,
        verbose=False,
        **fit_options,
    )
    if model.tree_count_ != params["iterations"]:
        raise TimeoutError("resource deadline stopped a fit; partial model not promoted")
    predict = np.asarray(model.predict(vx, thread_count=2), float)
    if model_kind == "single":
        frame = vm[["anchor_id", "station", "lead_h"]].copy()
        frame["value"] = np.clip(vm.current_hs.to_numpy() + predict, 0, 30)
        frame = frame.sort_values(["anchor_id", "lead_h"])
        prediction = frame.value.to_numpy().reshape(-1, 6)
    else:
        prediction = np.clip(valid_current[:, None] + predict, 0, 30)
    if not np.isfinite(prediction).all() or prediction.shape != (len(valid_ids), 6):
        raise ValueError("component prediction shape/finite mismatch")
    path = OUT / "models" / fold["id"] / f"{kind}.cbm"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    receipt = {
        "fold": fold["id"],
        "kind": kind,
        "seed": config["fold_seeds"][number],
        "rows": len(x),
        "train_anchors": len(ids),
        "validation_anchors": len(valid_ids),
        "iterations": model.tree_count_,
        "seconds": time.perf_counter() - started,
        "model_path": path.relative_to(OUT).as_posix(),
        "model_sha256": sha(path),
        "parameters": params,
        "cat_features": cats,
        "lead_dtype": "float64"
        if kind == "numeric_single"
        else "string"
        if kind == "categorical_single"
        else "not a multi input",
    }
    del model, x, y, vx
    gc.collect()
    return prediction, receipt


def component_frame(anchors, mask, fold, single, multi):
    part = anchors.loc[mask].sort_values("anchor_id")
    frame = part.loc[
        part.index.repeat(6), ["anchor_id", "station", "anchor_time", "episode_id", "current_hs"]
    ].reset_index(drop=True)
    frame.insert(0, "fold", fold)
    frame["lead_h"] = np.tile(LEADS, len(part))
    frame["target_hs"] = part[[f"target_{lead}" for lead in LEADS]].to_numpy().reshape(-1)
    frame["single_prediction"], frame["multi_prediction"] = single.reshape(-1), multi.reshape(-1)
    frame["persistence"] = frame.current_hs
    return frame


def router_material(frame, features):
    frame = frame.sort_values(["anchor_id", "lead_h"]).reset_index(drop=True)
    if (
        frame.duplicated(["anchor_id", "lead_h"]).any()
        or not frame.groupby("anchor_id").lead_h.agg(tuple).map(lambda value: value == LEADS).all()
    ):
        raise ValueError("router keys not six complete ordered leads")
    cases = frame.iloc[::6].reset_index(drop=True)
    components = (
        frame[["single_prediction", "multi_prediction", "persistence"]].to_numpy().reshape(-1, 6, 3)
    )
    observed = (
        features.set_index("anchor_id")
        .loc[cases.anchor_id, list(OBSERVED_FEATURES)]
        .reset_index(drop=True)
    )
    observed_x = build_inference_router_features(
        observed, cases.station.to_numpy(), cases.current_hs.to_numpy(), components
    )
    metadata = cases[["fold", "anchor_id", "station", "anchor_time", "episode_id"]]
    x, meta, component_rows, loss = expand_case_router_rows(
        observed_x, metadata, components, frame.target_hs.to_numpy().reshape(-1, 6)
    )
    return x, meta, component_rows, loss


def fixed_policy(component, features, anchors, contract, *, arm, fit, expected_receipts=None):
    x, meta, components, loss = router_material(component, features)
    weights = np.tile([0.5, 0.5, 0.0], (len(meta), 1))
    completed = []
    receipts = []
    for index, (fold, _) in enumerate(fold_masks(anchors, contract)):
        current = meta.fold.eq(fold["id"]).to_numpy()
        prior = meta.fold.isin(completed).to_numpy() & safe_meta_mask(
            meta, anchors, fold["id"], contract
        )
        path = OUT / "models" / "router" / f"{arm}_{fold['id']}.joblib"
        if index:
            if not prior.any():
                raise ValueError("later fold lacks safe prior OOF")
            start = pd.Timestamp(fold["start"])
            if not (
                meta.loc[prior, "anchor_time"] + pd.Timedelta(hours=24)
                < start - pd.Timedelta(hours=48)
            ).all():
                raise ValueError("router target/context footprint overlap")
            if fit:
                model = ComponentLossRouter(RouterConfig(10.0, 2.0, 0.5, "smooth_medium")).fit(
                    x.loc[prior], loss[prior]
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, path)
            else:
                model = joblib.load(path)
            weights[current] = model.predict_weights(x.loc[current])
        key_payload = meta.loc[prior, ["fold", "anchor_id", "station", "lead_h"]].to_json(
            orient="split", index=False
        )
        receipts.append(
            {
                "arm": arm,
                "fold": fold["id"],
                "past_fit_rows": int(prior.sum()),
                "past_fit_anchors": int(meta.loc[prior, "anchor_id"].nunique()),
                "prior_key_sha256": hashlib.sha256(key_payload.encode()).hexdigest(),
                "current_fold_labels_used": False,
                "target_context_separation_verified": True,
                "model_path": path.relative_to(OUT).as_posix() if index else None,
                "model_sha256": sha(path) if index else None,
            }
        )
        completed.append(fold["id"])
    weights[~meta.lead_h.isin([12, 18, 24]).to_numpy()] = [0.5, 0.5, 0.0]
    routed = route_row_predictions(components, weights)
    final = apply_long_lead_persistence_shrink(
        routed,
        components[:, 2],
        meta.lead_h.to_numpy(),
        config=LongLeadPersistenceShrink(0.2, (12, 18, 24)),
    )
    if expected_receipts is not None and receipts != expected_receipts:
        raise ValueError("saved router hash/training key lineage changed")
    out = component.sort_values(["anchor_id", "lead_h"]).reset_index(drop=True)
    if not out[KEYS].equals(meta[KEYS]):
        raise ValueError("router output alignment mismatch")
    out["final_prediction"] = final
    return out, receipts


def summary(frame, contract):
    if len(frame) != 103602 or frame.duplicated(KEYS).any():
        raise ValueError("OOF population/keys invalid")
    output = cv.paired_bootstrap(
        frame.target_hs,
        frame.control,
        frame.candidate,
        cv.bootstrap_groups(frame, "P3", contract),
        "rmse",
        contract,
    )
    output["control_sse_m2"] = float(np.square(frame.target_hs - frame.control).sum())
    output["candidate_sse_m2"] = float(np.square(frame.target_hs - frame.candidate).sum())
    slices = []
    for dimension in ("fold", "station", "lead_h", "wind_observed"):
        for key, part in frame.groupby(dimension):
            control = cv.pooled_metric(part.target_hs, part.control, "rmse")
            candidate = cv.pooled_metric(part.target_hs, part.candidate, "rmse")
            slices.append(
                {
                    "dimension": dimension,
                    "key": str(key),
                    "rows": len(part),
                    "anchors": int(part.anchor_id.nunique()),
                    "control_rmse_m": control,
                    "candidate_rmse_m": candidate,
                    "delta_candidate_minus_control_m": candidate - control,
                }
            )
    output["slices"] = slices
    output["worst_quarter_delta_m"] = max(
        row["delta_candidate_minus_control_m"] for row in slices if row["dimension"] == "fold"
    )
    output["maximum_slice_harm_m"] = max(
        row["delta_candidate_minus_control_m"] for row in slices if row["dimension"] != "fold"
    )
    output["onset_diagnostic"] = "NOT_ENABLED"
    output["greedy_diagnostic"] = "NOT_ENABLED"
    output["cluster_caveat"] = (
        "raw station high-runs, not independent meteorological storms; dense adjacent anchors and cross-station storm dependence remain"
    )
    output["official_score_projection"] = None
    return output


def execute(config, source):
    begin = time.perf_counter()
    verify_pins(config, source, sealed=True)
    if (OUT / "ATTEMPT_LOCK.json").exists():
        raise FileExistsError("consumed attempt preserved; no restart")
    pre = json.loads((REPORT / "preflight.json").read_text())
    if sha(OUT / "anchors.parquet") != pre["anchors_sha256"]:
        raise ValueError("prepared anchors changed")
    write_json(
        OUT / "ATTEMPT_LOCK.json",
        {
            "pid": os.getpid(),
            "created_utc": now(),
            "max_backbone_fits": 15,
            "max_router_fits": 8,
            "wall_seconds": 5400,
            "gpu": 1,
        },
    )
    features, _, columns = load_cache(config)
    anchors = pd.read_parquet(OUT / "anchors.parquet")
    contract = cv.load_contract(ROOT / config["contract"])
    recipe = json.loads((ROOT / config["recipe"]).read_text())
    deadline = begin + config["budget"]["wall_seconds"]
    fits, baseline_parts, numeric_parts, multis = [], [], [], {}
    masks = fold_masks(anchors, contract)

    def progress(stage):
        receipt = {
            "stage": stage,
            "pid": os.getpid(),
            "completed_backbone_fits": len(fits),
            "max_backbone_fits": 15,
            "elapsed_seconds": time.perf_counter() - begin,
            "cpu_threads": 2,
            "gpu": 1,
        }
        write_json(OUT / "progress.json", receipt, progress=True)
        write_json(OUT / "fit-receipts.json", {"fits": fits}, progress=True)
        print(json.dumps(receipt), flush=True)
        if time.perf_counter() >= deadline:
            raise TimeoutError("90-minute execution cap reached")

    for number, (fold, split) in enumerate(masks):
        progress(f"BASELINE_{fold['id']}_SINGLE")
        single, receipt = fit_one(
            "categorical_single",
            fold,
            number,
            split,
            features,
            anchors,
            columns,
            recipe,
            config,
            deadline,
        )
        fits.append(receipt)
        progress(f"BASELINE_{fold['id']}_MULTI")
        multi, receipt = fit_one(
            "multi", fold, number, split, features, anchors, columns, recipe, config, deadline
        )
        fits.append(receipt)
        multis[fold["id"]] = multi
        baseline_parts.append(
            component_frame(anchors, split["validation"], fold["id"], single, multi)
        )
        if number == 0:
            ratio = sum(config["expected_train"]) / config["expected_train"][0]
            estimate = (
                config["budget"]["pilot_estimate_margin"]
                * ratio
                * (2 * fits[0]["seconds"] + fits[1]["seconds"])
            )
            pilot = {
                "status": "CONTINUE"
                if estimate < config["budget"]["wall_seconds"]
                else "RESOURCE_CAP_PREDICTED_STOP_BEFORE_OUTCOME_REVIEW",
                "planned_first_two_fits_reused": True,
                "single_seconds": fits[0]["seconds"],
                "multi_seconds": fits[1]["seconds"],
                "train_anchor_scale_sum": ratio,
                "margin": 1.2,
                "estimated_total_fit_seconds": estimate,
                "wall_cap_seconds": 5400,
                "model_metrics_inspected": False,
            }
            write_json(REPORT / "resource-pilot.json", pilot)
            if estimate >= config["budget"]["wall_seconds"]:
                write_json(OUT / "fit-receipts.json", {"fits": fits}, progress=True)
                raise TimeoutError(
                    "new mixed-device planned fits forecast exceeds90min; no outcome-driven restart"
                )
    progress("BASELINE_COMPLETE_FIXED_ROUTER")
    baseline_components = pd.concat(baseline_parts, ignore_index=True)
    baseline, base_router = fixed_policy(
        baseline_components, features, anchors, contract, arm="control", fit=True
    )
    baseline.to_parquet(OUT / "baseline_oof.parquet", index=False)
    write_json(
        REPORT / "baseline-complete.json",
        {
            "status": "FRESH_MIXED_DEVICE_BASELINE_COMPLETE_BEFORE_CANDIDATE",
            "backbone_fits": 10,
            "router_fits": 4,
            "rows": len(baseline),
            "oof_sha256": sha(OUT / "baseline_oof.parquet"),
            "old_gpu_score_inherited": False,
            "router_receipts": base_router,
            **zero_access(),
        },
    )
    for number, (fold, split) in enumerate(masks):
        progress(f"NUMERIC_CANDIDATE_{fold['id']}")
        single, receipt = fit_one(
            "numeric_single",
            fold,
            number,
            split,
            features,
            anchors,
            columns,
            recipe,
            config,
            deadline,
        )
        fits.append(receipt)
        numeric_parts.append(
            component_frame(anchors, split["validation"], fold["id"], single, multis[fold["id"]])
        )
    progress("BOTH_ARMS_COMPLETE_FIXED_ROUTER")
    numeric, candidate_router = fixed_policy(
        pd.concat(numeric_parts, ignore_index=True),
        features,
        anchors,
        contract,
        arm="candidate",
        fit=True,
    )
    if not baseline[KEYS].equals(numeric[KEYS]) or not np.array_equal(
        baseline.target_hs, numeric.target_hs
    ):
        raise ValueError("paired arm target/key mismatch")
    numeric.to_parquet(OUT / "candidate_oof.parquet", index=False)
    final = baseline[KEYS + ["anchor_time", "episode_id", "target_hs"]].copy()
    final["control"], final["candidate"] = baseline.final_prediction, numeric.final_prediction
    wind_columns = [
        c
        for c in columns
        if c.startswith(("wspd_", "gust_", "wdir_sin_", "wdir_cos_")) and "_valid_" in c
    ]
    final["wind_observed"] = (
        features.set_index("anchor_id")
        .loc[final.anchor_id, wind_columns]
        .gt(0)
        .any(axis=1)
        .to_numpy()
    )
    final.to_parquet(OUT / "paired_oof.parquet", index=False)
    stats = summary(final, contract)
    verify_pins(config, source, sealed=True)
    result = {
        "status": "MEAN_IMPROVEMENT_CANDIDATE_RETAINED"
        if stats["candidate_retained"]
        else "NO_OP_NUMERIC_LEAD_NOT_IMPROVED",
        "pid": os.getpid(),
        "comparison": stats,
        "fit_count": 15,
        "router_fit_count": 8,
        "fit_receipts": fits,
        "router_receipts": {"control": base_router, "candidate": candidate_router},
        "execution_seconds": time.perf_counter() - begin,
        "preflight_seconds": pre["seconds"],
        "runner_sha256": sha(__file__),
        "config_sha256": sha(CONFIG),
        "preflight_sha256": sha(REPORT / "preflight.json"),
        "source_sha256": config["source_files"],
        "artifacts": {
            name: sha(OUT / name)
            for name in (
                "anchors.parquet",
                "baseline_oof.parquet",
                "candidate_oof.parquet",
                "paired_oof.parquet",
            )
        },
        "baseline_is_new_v5_not_inherited_old_score": True,
        "baseline_regeneration_check": "NOT_RUN_FULL_MATERIALIZATION_OUT_OF_SCOPE",
        "same_environment_saved_model_replay": "PENDING",
        **zero_access(),
    }
    write_json(REPORT / "result.json", result)
    write_json(
        OUT / "TERMINAL.json",
        {
            "status": result["status"],
            "result_sha256": sha(REPORT / "result.json"),
            "fits": 15,
            "router_fits": 8,
        },
    )
    progress("TERMINAL")
    print(
        json.dumps(
            {
                "status": result["status"],
                "seconds": result["execution_seconds"],
                "comparison": {k: v for k, v in stats.items() if k != "slices"},
            }
        ),
        flush=True,
    )


def replay(config, source):
    start = time.perf_counter()
    verify_pins(config, source, sealed=True)
    result = json.loads((REPORT / "result.json").read_text())
    if os.getpid() == result["pid"]:
        raise ValueError("replay must use new process")
    for name, expected in result["artifacts"].items():
        if sha(OUT / name) != expected:
            raise ValueError("saved evaluation artifact hash mismatch")
    feature, _, columns = load_cache(config)
    anchors = pd.read_parquet(OUT / "anchors.parquet")
    contract = cv.load_contract(ROOT / config["contract"])
    blocks = {"control": [], "candidate": []}
    fit_lookup = {(r["fold"], r["kind"]): r for r in result["fit_receipts"]}
    checks, max_error = 0, 0.0
    for fold, split in fold_masks(anchors, contract):
        ids = anchors.loc[split["validation"], "anchor_id"].to_numpy(int)
        mx, _, current = multi_matrix(feature, anchors, ids, columns)
        x, _, meta = expand_leads(feature, anchors, ids, columns)
        values = {}
        for kind in ("multi", "categorical_single", "numeric_single"):
            receipt = fit_lookup[(fold["id"], kind)]
            path = OUT / receipt["model_path"]
            if sha(path) != receipt["model_sha256"]:
                raise ValueError("saved backbone changed")
            model = CatBoostRegressor().load_model(path)
            if kind == "multi":
                values[kind] = np.clip(current[:, None] + model.predict(mx, thread_count=2), 0, 30)
            else:
                prediction = np.clip(
                    meta.current_hs.to_numpy()
                    + model.predict(matrix(x, kind == "numeric_single"), thread_count=2),
                    0,
                    30,
                )
                ordered = (
                    meta[["anchor_id", "lead_h"]]
                    .assign(prediction=prediction)
                    .sort_values(["anchor_id", "lead_h"])
                )
                values[kind] = ordered.prediction.to_numpy().reshape(-1, 6)
            checks += 1
        for arm, kind in (("control", "categorical_single"), ("candidate", "numeric_single")):
            blocks[arm].append(
                component_frame(
                    anchors, split["validation"], fold["id"], values[kind], values["multi"]
                )
            )
        gc.collect()
    for arm, filename in (
        ("control", "baseline_oof.parquet"),
        ("candidate", "candidate_oof.parquet"),
    ):
        predicted, _ = fixed_policy(
            pd.concat(blocks[arm], ignore_index=True),
            feature,
            anchors,
            contract,
            arm=arm,
            fit=False,
            expected_receipts=result["router_receipts"][arm],
        )
        stored = pd.read_parquet(OUT / filename)
        if not predicted[KEYS].equals(stored[KEYS]):
            raise ValueError("replay row key mismatch")
        for col in ("single_prediction", "multi_prediction", "final_prediction"):
            error = float(np.max(np.abs(predicted[col] - stored[col])))
            max_error = max(error, max_error)
            if error != 0:
                raise ValueError("saved model exact replay failed")
    receipt = {
        "status": "PASS",
        "fresh_pid": os.getpid(),
        "training_pid": result["pid"],
        "backbone_models": checks,
        "router_models": 8,
        "paired_rows": 103602,
        "max_abs_prediction_error_m": max_error,
        "seconds": time.perf_counter() - start,
        "result_sha256": sha(REPORT / "result.json"),
        "same_environment_only": True,
        "full_regeneration_or_portable_package": False,
        **zero_access(),
    }
    write_json(REPORT / "fresh-process-replay.json", receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    access_guard(source, config)
    try:
        if args.preflight:
            preflight(config, source)
        elif args.execute:
            execute(config, source)
        else:
            replay(config, source)
    except Exception as exc:
        if (
            OUT.exists()
            and not (OUT / "TERMINAL.json").exists()
            and not (OUT / "FAILURE.json").exists()
        ):
            write_json(
                OUT / "FAILURE.json",
                {
                    "status": "RESOURCE_STOP_NO_RESTART"
                    if isinstance(exc, TimeoutError)
                    else "TECHNICAL_FAILURE_NO_RESTART",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "pid": os.getpid(),
                    "created_utc": now(),
                    **zero_access(),
                },
            )
        raise


if __name__ == "__main__":
    main()
