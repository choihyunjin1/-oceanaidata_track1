"""Original P1 recipe: source-only training and general-rule inference.

Frozen historical OOF-selected settings are declared model hyperparameters;
this does not rerun historical HPO or recover the lost cell-selector program.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "02_code"
KEYS = ["station", "year", "layer", "time"]
BASE = [*KEYS, "temp", "psal", "depth"]
SEEDS = [20260813, 20260829, 20260847]
POST = {"high_threshold": .2, "low_threshold": .1,
        "close_gap_rows": 0, "minimum_positive_run": 12}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def verify_manifest():
    for name, expected in read(ROOT / "source-manifest.json")["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Source manifest escapes package")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source pin mismatch: {name}")


def initialize(data, kind):
    verify_manifest()
    helper = module("owned_ms_runtime", CODE / "ms_driver.py")
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = "8"
    helper.configure_owned_runtime(ROOT)
    sys.path.insert(0, str(CODE / "source/src"))
    helper.install_io_guard(ROOT, data / f"{kind}.csv")
    return helper


def tree_train(data):
    helper = initialize(data, "train")
    import joblib
    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    from p1_qc.config import load_config
    from p1_qc.features import build_features
    from p1_qc.pipeline import TabularEncoder, save_model, train_full_model

    target = ROOT / "03_model/tree"
    target.mkdir(exist_ok=False)
    started = time.time()
    write(target / "ATTEMPT_LOCK.json", {"pid": os.getpid(), "started": started, "fits_max": 4})
    if helper.sha(data / "train.csv") != read(ROOT / "contract.json")["train_sha256"]:
        raise ValueError("Training input drift")
    frame = pd.read_csv(data / "train.csv", low_memory=False)
    cfg = load_config(CODE / "tree_config.toml", env={})
    bundle = build_features(frame[BASE].copy(), config=cfg)
    # This is the original full-data training entrypoint, not the modified clean one.
    selection = read(CODE / "tree_recipe.json")["O_selection"]
    original = train_full_model(frame, bundle, cfg, selection)
    save_model(original, target / "O.joblib")
    encoder = TabularEncoder().fit(bundle, np.arange(len(frame)))
    x = encoder.transform(bundle)
    y = frame.label.to_numpy(dtype=np.int8)
    weight_code = module("owned_event_weight", CODE / "event_weight.py")
    weights = weight_code._event_day_weight(frame[KEYS], y)
    packages = []
    parameters = read(CODE / "tree_recipe.json")["B_parameters"]
    for seed in SEEDS:
        params = dict(parameters, objective="binary", random_state=seed, n_jobs=8,
                      verbosity=-1, deterministic=True, force_row_wise=True,
                      feature_fraction_seed=seed, bagging_seed=seed,
                      data_random_seed=seed, extra_seed=seed)
        model = lgb.LGBMClassifier(**params)
        model.fit(x, y, sample_weight=weights)
        packages.append({"seed": seed, "global": model})
    saved = {"branch": "event_day_balanced_binary_lgbm", "encoder": encoder,
             "packages": packages, "seeds": SEEDS, "postprocess": POST}
    joblib.dump(saved, target / "B.joblib", compress=3)
    ids = np.linspace(0, len(frame)-1, 256, dtype=int)
    # A train-only numerical replay probe, not an in-sample quality score.
    o_x = original.encoder.transform(bundle)[ids]
    np.savez_compressed(target / "own_probe.npz", o_x=o_x, b_x=x[ids],
                        o=original.model.predict_proba(o_x)[:, 1],
                        b=np.mean(np.vstack([p["global"].predict_proba(x[ids])[:, 1]
                                             for p in packages]), axis=0))
    write(target / "training-result.json", {"status": "TRAINED_REPLAY_PENDING",
          "pid": os.getpid(), "fits": 4, "rows": len(frame), "features": len(bundle.feature_columns),
          "O_sha256": helper.sha(target / "O.joblib"), "B_sha256": helper.sha(target / "B.joblib"),
          "runtime_seconds": time.time()-started, "old_model_cache_prediction_reads": 0,
          "official_rows": 0, "tree_parameters": parameters, "postprocess": POST})


def tree_qa(data):
    helper = initialize(data, "train")
    import joblib
    import numpy as np

    folder = ROOT / "03_model/tree"
    receipt = read(folder / "training-result.json")
    checks = {"fresh_pid": receipt["pid"] != os.getpid(), "fits": receipt["fits"] == 4}
    for name in ["O", "B"]:
        checks[f"{name}_sha"] = helper.sha(folder / f"{name}.joblib") == receipt[f"{name}_sha256"]
    o, b = joblib.load(folder / "O.joblib"), joblib.load(folder / "B.joblib")
    with np.load(folder / "own_probe.npz", allow_pickle=False) as probe:
        checks["O_probe_exact"] = np.array_equal(o.model.predict_proba(probe["o_x"])[:, 1], probe["o"])
        probability = np.mean(np.vstack([p["global"].predict_proba(probe["b_x"])[:, 1]
                                         for p in b["packages"]]), axis=0)
        checks["B_probe_exact"] = np.array_equal(probability, probe["b"])
    write(folder / "qa.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks})
    if not all(checks.values()):
        raise ValueError("Tree numerical replay failed")


def infer_tree(data):
    helper = initialize(data, "test")
    import joblib
    import numpy as np
    import pandas as pd

    from p1_qc.config import load_config
    from p1_qc.features import build_features
    from p1_qc.pipeline import apply_postprocess, predict_submission
    from p1_qc.rules import detect_plateaus, detect_singleton_spikes

    if read(ROOT / "03_model/tree/qa.json")["status"] != "PASS":
        raise ValueError("Tree QA required")
    if helper.sha(data / "test.csv") != read(ROOT / "contract.json")["test_sha256"]:
        raise ValueError("Official covariate pin mismatch")
    frame = pd.read_csv(data / "test.csv", low_memory=False)
    bundle = build_features(frame[BASE], config=load_config(CODE / "tree_config.toml", env={}))
    o, _ = predict_submission(joblib.load(ROOT / "03_model/tree/O.joblib"), frame, bundle)
    b_model = joblib.load(ROOT / "03_model/tree/B.joblib")
    x = b_model["encoder"].transform(bundle)
    probability = np.mean(np.vstack([p["global"].predict_proba(x)[:, 1]
                                     for p in b_model["packages"]]), axis=0)
    plateau, spike = detect_plateaus(frame).to_numpy(), detect_singleton_spikes(frame).to_numpy()
    b = apply_postprocess(frame, probability, plateau, spike, POST)
    types = np.full(len(frame), "", dtype=object)
    types[plateau & b.astype(bool)] = "flatline"
    types[spike & b.astype(bool)] = "spike"
    composition = module("owned_general_composition", CODE / "composition.py")
    router, gi = composition.compose_tree(frame.station, frame.layer, o.label, b,
                    add_cells=composition.ADD_CELLS, remove_cells=composition.REMOVE_CELLS)
    added = (gi == 1) & (b == 0)
    types[added] = o.anomaly_type.to_numpy()[added]
    np.savez_compressed(ROOT / "04_logs/tree_inference.npz", router=router, gi=gi,
                        gi_type=types.astype(str))


def infer_ms(data):
    # Load the module from the MS-owned snapshot: avoids the previous cold QA path bug.
    folder = ROOT / "03_model/mstcn"
    driver = module("owned_ms_predict", folder / "02_code/mstcn.py")
    driver.configure_owned_runtime(ROOT)
    driver.install_io_guard(ROOT, data / "test.csv")
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(data / "test.csv", usecols=BASE, low_memory=False)
    keys, proposal, _ = driver.predict_proposal(frame, model_dir=folder / "03_model")
    if not keys[KEYS].astype(str).equals(frame[KEYS].astype(str)):
        raise ValueError("MS key order drift")
    np.save(ROOT / "04_logs/ms_proposal.npy", proposal, allow_pickle=False)


def combine(data, name):
    helper = initialize(data, "test")
    import numpy as np
    import pandas as pd

    from p1_qc.submission import validate_submission

    compose = module("owned_composition", CODE / "composition.py")
    test = pd.read_csv(data / "test.csv", usecols=KEYS)
    with np.load(ROOT / "04_logs/tree_inference.npz", allow_pickle=False) as trees:
        ms = np.load(ROOT / "04_logs/ms_proposal.npy", allow_pickle=False)
        test["label"] = compose.compose_mstcn_spike(trees["router"], trees["gi"], ms, trees["gi_type"])
    validation = validate_submission(test, test[KEYS])
    target = ROOT / "05_answer" / name
    with target.open("x", encoding="utf-8", newline="") as handle:
        test.to_csv(handle, index=False, lineterminator="\n")
    expected = read(ROOT / "contract.json")["historical_answer_sha256"]
    write(ROOT / "06_docs" / f"{name}.json", {"status": "SCHEMA_PASS",
          "validator": validation, "sha256": helper.sha(target),
          "historical_answer_exact": helper.sha(target) == expected,
          "old_prediction_inputs": 0, "fixed_row_patch_inputs": 0,
          "official_score": "Only historical SHA match links the historical score; no new scoring"})


def pipeline(data):
    verify_manifest()
    started = time.time()
    write(ROOT / "ATTEMPT_LOCK.json", {"started": started, "pid": os.getpid(), "cap_seconds": 21600})
    if any((ROOT / "03_model").iterdir()):
        raise ValueError("Start with empty 03_model; never delete an existing run")
    commands = [("tree_train", [sys.executable, str(__file__), "tree-train", "--data", str(data)]),
                ("tree_qa", [sys.executable, str(__file__), "tree-qa", "--data", str(data)]),
                ("ms_train", [sys.executable, str(CODE / "ms_driver.py"), "train",
                 "--source-root", str(CODE / "source"), "--train-csv", str(data / "train.csv"),
                 "--output", str(ROOT / "03_model/mstcn"), "--training-approved", "--gpu-approved"]),
                ("ms_qa", [sys.executable, str(ROOT / "03_model/mstcn/02_code/mstcn.py"),
                 "replay", "--output", str(ROOT / "03_model/mstcn")])]
    commands.extend((stage, [sys.executable, str(__file__), stage, "--data", str(data)])
                    for stage in ["infer-tree", "infer-ms", "combine"])
    completed = []
    try:
        for stage, command in commands:
            verify_manifest()
            print(json.dumps({"stage": stage, "elapsed": time.time()-started}), flush=True)
            with (ROOT / "04_logs" / f"{stage}.log").open("x", encoding="utf-8") as log:
                subprocess.run(command, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               timeout=max(1, 21600-(time.time()-started)))
            completed.append(stage)
        answer = read(ROOT / "06_docs/P1_submission.csv.json")
        write(ROOT / "terminal.json", {"status": "TRAIN_TO_ANSWER_COMPLETE",
              "historical_answer_exact": answer["historical_answer_exact"], "completed": completed,
              "fits": 7, "runtime_seconds": time.time()-started, "upload": 0})
    except BaseException as error:
        write(ROOT / "terminal.json", {"status": "TERMINAL_TECHNICAL_FAILURE", "completed": completed,
              "error": str(error), "runtime_seconds": time.time()-started, "auto_restart": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["all", "tree-train", "tree-qa", "infer-tree", "infer-ms", "combine"])
    parser.add_argument("--data", type=Path, default=os.environ.get("P1_DATA_DIR"))
    args = parser.parse_args()
    if args.data is None:
        parser.error("Set P1_DATA_DIR or pass --data")
    functions = {"all": pipeline, "tree-train": tree_train, "tree-qa": tree_qa,
                 "infer-tree": infer_tree, "infer-ms": infer_ms,
                 "combine": lambda data: combine(data, "P1_submission.csv")}
    functions[args.stage](args.data.resolve())
