"""P1 frozen champion: new 11-fit cold workflow or clock-independent saved replay."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parent
RECON = CODE / "scripts/p1_champion_reconstruction_20260906_v1"
MODELS = PACKAGE / "03_model"
LOGS = PACKAGE / "04_logs"
DOCS = PACKAGE / "06_docs"
KEYS = ["station", "year", "layer", "time"]
RAW = KEYS + ["temp", "psal", "depth"]
TEST_SHA = "6d5c6522c282651b99f4261ffa803cf99950596028e996de1e7714db77408387"
SAMPLE_SHA = "e7027bcb56836587715e5cd818c6d595ebef4a8518538fa7de7bdcba2b7fc1e4"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def checked():
    manifest = read(PACKAGE / "package-manifest.json")
    for name, expected in manifest["files"].items():
        path = (PACKAGE / name).resolve()
        if not path.is_relative_to(PACKAGE) or sha(path) != expected:
            raise ValueError("package source/model changed: " + name)
    frozen = read(CODE / "frozen.json")
    if frozen["kind"] != manifest["kind"] or frozen["tree_arm"] != "union":
        raise ValueError("package policy mismatch")
    if frozen["budget"]["total_fits"] != 11:
        raise ValueError("fixed fit budget mismatch")
    for name, version in frozen["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("package version mismatch: " + name)
    return frozen


def empty_training_gate(frozen):
    if frozen["kind"] != "cold" or any(MODELS.iterdir()):
        raise RuntimeError("training requires a new cold package and empty 03_model")
    if (DOCS / "cold-start.json").exists():
        raise FileExistsError("consumed cold attempt; no restart")


def runtime(stage):
    threads = 4 if "tree" in stage else 2
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    scratch = LOGS / "runtime" / stage
    scratch.mkdir(parents=True, exist_ok=True)
    for name in ("TEMP", "TMP", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        os.environ[name] = str(scratch)
    sys.path.insert(0, str(RECON))
    sys.path.insert(0, str(CODE / "scripts"))
    return threads


def boundary(train=None, official=None):
    """Python audit boundary, not an OS air-gap claim. Install after imports."""
    allowed = {Path(p).resolve() for p in ([train] if train else [])}
    if official:
        allowed.update({Path(official).resolve() / n for n in ("test.csv", "sample_submission.csv")})
    libraries = (Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve())

    def guard(event, args):
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo"}:
            raise PermissionError("network access prohibited")
        if event != "open" or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode = args[1] or ""
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = any(c in str(mode) for c in "wax+") or bool(flags & (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if path.is_relative_to(PACKAGE):
            return
        if not writing and (path in allowed or any(path.is_relative_to(p) for p in libraries)):
            return
        raise PermissionError("outside package data boundary: " + path.name)

    sys.addaudithook(guard)


def inventory(folder):
    return {p.relative_to(PACKAGE).as_posix(): sha(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and "04_runtime" not in p.parts}


def check_files(files):
    for relative, digest in files.items():
        path = (PACKAGE / relative).resolve()
        if not path.is_relative_to(PACKAGE) or sha(path) != digest:
            raise ValueError("generated artifact changed: " + relative)


def tree_module():
    import tree
    return tree


def tree_train(frozen, data):
    import numpy as np
    from threadpoolctl import threadpool_limits

    tree = tree_module()
    boundary(train=data / "train.csv")
    cfg = frozen["tree_config"]
    frame = tree.training_source(cfg)
    masks, support = tree.split_masks(frame, cfg)
    tm, vm = masks[("2025_q4", "inner")]
    inner, valid = frame.loc[tm].reset_index(drop=True), frame.loc[vm].reset_index(drop=True)
    output = MODELS / "tree"
    output.mkdir()
    state = {"fit_cap": 8, "attempted_fits": 0, "completed_fits": 0, "fit_receipts": []}

    def progress(stage):
        tree.core.write_json(LOGS / "tree-progress.json", {**state, "stage": stage,
                             "pid": os.getpid(), "time": time.time()})

    with threadpool_limits(limits=4):
        probabilities, rules = tree.fit_four(inner, valid, cfg, output / "inner/models", state, progress)
        selector = tree.select_policy(valid, probabilities, rules, cfg)
        bits = tree.predict_policy(valid, probabilities, rules, cfg, selector)["union"]
        write(output / "selector.json", selector)
        np.savez_compressed(output / "inner-probe.npz", row_id=valid.row_id.to_numpy(),
                            **probabilities, union=bits)
        selected_metrics = tree.metric(valid.label, bits)
        del inner, valid, probabilities
        gc.collect()
        probe = frame.iloc[np.linspace(0, len(frame) - 1, 2048, dtype=int)].reset_index(drop=True)
        predictions, _rules = tree.fit_four(frame, probe, cfg, output / "full/models", state,
                                           progress, in_sample=True)
        np.savez_compressed(output / "full-probe.npz", row_id=probe.row_id.to_numpy(), **predictions)
    if state["completed_fits"] != 8 or state["attempted_fits"] != 8:
        raise ValueError("tree actual fit count mismatch")
    if sha(data / "train.csv") != cfg["train_sha256"]:
        raise ValueError("train source changed")
    write(DOCS / "tree-training.json", {"status": "COMPLETE", "pid": os.getpid(), **state,
          "support": next(s for s in support if s["fold"] == "2025_q4" and s["stage"] == "inner"),
          "selection_metric_not_holdout": selected_metrics, "files": inventory(output),
          "selector_sha256": sha(output / "selector.json"), "official_rows": 0,
          "old_model_oof_answer_reads": 0})


def tree_predictions(tree, frame, folder, cfg, *, valid=None):
    import numpy as np

    pre = tree.core.joblib.load(folder / "preprocess.joblib")
    x = pre["encoder"].transform(tree.base_features(frame, pre["stats"], cfg))
    if valid is not None:
        positions = tree.core.pd.Index(frame.row_id).get_indexer(valid.row_id)
        if (positions < 0).any():
            raise ValueError("probe IDs changed")
        x = x[positions]
    values = {"O": [], "B": []}
    for name in pre["model_files"]:
        model = tree.core.joblib.load(folder / name)
        # Do not change the serialized model parameters or threads.
        probability = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
        if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError("invalid tree probability")
        values[name[0]].append(probability)
        del model
    if [len(values[a]) for a in ("O", "B")] != [1, 3]:
        raise ValueError("O1/B3 count mismatch")
    return {arm: np.mean(value, axis=0) for arm, value in values.items()}, pre


def tree_qa(frozen, data):
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score
    from threadpoolctl import threadpool_limits

    tree = tree_module()
    boundary(train=data / "train.csv")
    training = read(DOCS / "tree-training.json")
    check_files(training["files"])
    frame = tree.training_source(frozen["tree_config"])
    masks, support = tree.split_masks(frame, frozen["tree_config"])
    tm, vm = masks[("2025_q4", "inner")]
    checks = {"fresh_pid": training["pid"] != os.getpid(), "eight_fits": training["completed_fits"] == 8,
              "train_eval_disjoint": not bool((tm & vm).any()),
              "source_hash": sha(data / "train.csv") == frozen["tree_config"]["train_sha256"]}
    output = MODELS / "tree"
    with threadpool_limits(limits=4):
        for phase in ("inner", "full"):
            with np.load(output / f"{phase}-probe.npz", allow_pickle=False) as probe:
                rows = frame.loc[frame.row_id.isin(probe["row_id"])].reset_index(drop=True)
                if not np.array_equal(rows.row_id, probe["row_id"]):
                    raise ValueError("probe ordered IDs changed")
                predictions, pre = tree_predictions(tree, rows if phase == "inner" else frame,
                    output / phase / "models", frozen["tree_config"], valid=rows if phase == "full" else None)
                for arm in ("O", "B"):
                    checks[f"{phase}_{arm}_exact"] = bool(np.array_equal(predictions[arm], probe[arm]))
                if phase == "inner":
                    rules = tree.core.rule_masks(rows, pre["stats"])
                    selected = tree.select_policy(rows, predictions, rules, frozen["tree_config"])
                    checks["selector_exact"] = selected == read(output / "selector.json")
                    bits = tree.predict_policy(rows, predictions, rules, frozen["tree_config"], selected)["union"]
                    checks["union_exact"] = bool(np.array_equal(bits, probe["union"]))
                    tn, fp, fn, tp = confusion_matrix(rows.label, bits, labels=[0, 1]).ravel()
                    metric = training["selection_metric_not_holdout"]
                    for name, number in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)):
                        if name in metric:
                            checks["independent_" + name] = int(number) == metric[name]
                    checks["independent_f1"] = abs(float(f1_score(rows.label, bits)) - metric["f1"]) < 1e-14
    checks["same_support"] = training["support"] == next(
        s for s in support if s["fold"] == "2025_q4" and s["stage"] == "inner")
    check_files(training["files"])
    write(DOCS / "tree-replay-qa.json", {"status": "PASS" if all(checks.values()) else "FAIL",
          "checks": checks, "pid": os.getpid(), "training_sha256": sha(DOCS / "tree-training.json"),
          "fits": 0, "official_rows": 0})
    if not all(checks.values()):
        raise ValueError("fresh tree replay/arithmetic failed")


def ms_train(data):
    import mstcn

    mstcn.train(CODE, data / "train.csv", MODELS / "mstcn", training_approved=True, gpu_approved=True)


def ms_qa():
    import mstcn

    # Exact unchanged replay; consumes only this NEW cold run's own probe/state.
    mstcn.replay(MODELS / "mstcn")


def training_finalise(frozen, data):
    tree = read(DOCS / "tree-training.json")
    tq = read(DOCS / "tree-replay-qa.json")
    ms = read(MODELS / "mstcn/training-result.json")
    mq = read(MODELS / "mstcn/fresh-process-replay.json")
    checks = {"tree8": tree["completed_fits"] == 8, "MS3": ms["completed_fits"] == 3,
              "tree_fresh_qa": tq["status"] == "PASS" and tq["training_sha256"] == sha(DOCS / "tree-training.json"),
              "MS_fresh_qa": mq["status"] == "PASS" and mq["training_result_sha256"] == sha(MODELS / "mstcn/training-result.json"),
              "train_immutable": sha(data / "train.csv") == frozen["tree_config"]["train_sha256"],
              "model_count": len(list(MODELS.glob("tree/*/models/[OB]_*.joblib"))) == 8 and
                             len(list(MODELS.glob("mstcn/03_model/seed_*.pt"))) == 3}
    check_files(tree["files"])
    write(DOCS / "training-qa.json", {"status": "PASS" if all(checks.values()) else "FAIL",
          "checks": checks, "pid": os.getpid(), "completed_fits": 11, "files": inventory(MODELS),
          "tree_qa_sha256": sha(DOCS / "tree-replay-qa.json"),
          "MS_qa_sha256": sha(MODELS / "mstcn/fresh-process-replay.json"),
          "cold_start_sha256": sha(DOCS / "cold-start.json"), "official_rows": 0,
          "old_models_oof_answers_as_training_inputs": 0})
    if not all(checks.values()):
        raise ValueError("training QA failed")


def inference_gate(frozen):
    if frozen["kind"] == "cold":
        qa = read(DOCS / "training-qa.json")
        if qa["status"] != "PASS" or qa["completed_fits"] != 11:
            raise ValueError("new cold training QA required before official reads")
        check_files(qa["files"])
    # Saved models and their completed original QA metadata are package-manifest pinned.


def answer_status(kind, matches_original, *, replay=False, fresh_pid=True, byte_exact=True):
    return "PASS" if ((kind != "saved" or matches_original) and
                      (not replay or (fresh_pid and byte_exact))) else "FAIL"


def official(data, sample=False):
    import composition
    import pandas as pd

    for name, expected in (("test.csv", TEST_SHA), ("sample_submission.csv", SAMPLE_SHA)):
        if sha(data / name) != expected:
            raise ValueError("official source hash mismatch")
    frame = pd.read_csv(data / "test.csv", usecols=RAW)
    if len(frame) != 169011:
        raise ValueError("official population mismatch")
    composition.canonical_keys(frame)
    if sample:
        keys = pd.read_csv(data / "sample_submission.csv", usecols=KEYS)
        if not keys.equals(frame[KEYS]):
            raise ValueError("sample ordered keys mismatch")
    if sha(data / "test.csv") != TEST_SHA or sha(data / "sample_submission.csv") != SAMPLE_SHA:
        raise ValueError("official source changed while reading")
    return frame


def tree_infer(frozen, data, directory):
    import numpy as np
    from threadpoolctl import threadpool_limits

    tree = tree_module()
    boundary(official=data)
    inference_gate(frozen)
    original = official(data)
    frame = original.assign(_order=np.arange(len(original))).sort_values(
        ["station", "layer", "time"], kind="stable").reset_index(drop=True)
    with threadpool_limits(limits=4):
        predictions, pre = tree_predictions(tree, frame, MODELS / "tree/full/models", frozen["tree_config"])
    rules = tree.core.rule_masks(frame, pre["stats"])
    bits = tree.predict_policy(frame, predictions, rules, frozen["tree_config"],
                               read(MODELS / "tree/selector.json"))["union"]
    restored = np.empty(len(frame), dtype=np.int8)
    restored[frame._order.to_numpy()] = bits
    original[KEYS].assign(tree=restored).to_parquet(directory / "tree.parquet", index=False)


def ms_predict(frame):
    """Exact numeric path of frozen mstcn.predict_frame, without its old run clock/probes."""
    import mstcn
    import numpy as np

    source, config = mstcn.load_source(CODE)
    torch, device = mstcn.configure_device(source)
    surface, _names, _dependency = mstcn.fresh_features(frame, CODE, source, config)
    models = MODELS / "mstcn/03_model"
    encoder = mstcn.encoder_from_receipt(read(models / "encoder.json"))
    encoded = mstcn.encode_with_saved(surface, encoder, source)
    windows = source._all_windows(encoded, config)
    predictions = []
    for seed in mstcn.SEEDS:
        model = mstcn.load_model(models, seed, source, config, device)
        predictions.append(source.predict_encoded(model, encoded, windows, batch_size=64, device=device))
        del model
        torch.cuda.empty_cache()
    mean = source.PredictionBundle(*(
        np.mean(np.stack([getattr(p, name) for p in predictions]), axis=0).astype(np.float32)
        for name in ("row_probability", "boundary_probability", "type_probability")))
    proposal = source.decode_long_event_segments(source._decoder_row_probability(mean, config),
        mean.boundary_probability, encoded.layout, high_threshold=0.8, low_threshold=0.4,
        snap_radius=12, minimum_rows=19, maximum_rows=None)
    return surface.keys.copy(), proposal


def ms_infer(frozen, data, directory):
    import mstcn

    # Load scientific libraries before the file guard, then verify source/model pins.
    mstcn.load_source(CODE)
    boundary(official=data)
    inference_gate(frozen)
    frame = official(data)
    keys, bits = ms_predict(frame)
    keys.assign(proposal=bits).to_parquet(directory / "mstcn.parquet", index=False)


def combine(frozen, data, directory, replay):
    import composition
    import numpy as np
    import pandas as pd

    boundary(official=data)
    inference_gate(frozen)
    parts = []
    for name in ("tree", "mstcn"):
        receipt = read(directory / (name + "-stage.json"))
        path = directory / (name + ".parquet")
        if receipt["status"] != "PASS" or receipt["sha256"] != sha(path):
            raise ValueError("component hash mismatch")
        parts.append(pd.read_parquet(path))
    answer, joined = composition.combine(*parts)
    original = official(data, sample=True)
    if not answer[KEYS].equals(original[KEYS]) or len(answer) != 169011:
        raise ValueError("combined ordered keys mismatch")
    composition.binary_column(answer, "label")
    if list(answer.columns) != KEYS + ["label"]:
        raise ValueError("answer schema mismatch")
    target = PACKAGE / "05_answer" / ("replay" if replay else "") / "P1_submission.csv"
    target.parent.mkdir(exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as stream:
        answer.to_csv(stream, index=False, lineterminator="\n")
    own = pd.read_csv(target)
    if not own[KEYS].equals(answer[KEYS]) or not np.array_equal(own.label, answer.label):
        raise ValueError("own answer serialization changed")
    digest = sha(target)
    receipt = {"status": "PASS", "pid": os.getpid(), "sha256": digest, "rows": 169011,
               "positive_rows": int(answer.label.sum()), "composition": joined,
               "expected_original_sha256": frozen["expected_answer_sha256"],
               "matches_original_b2f17": digest == frozen["expected_answer_sha256"],
               "new_fits": 0, "test_rows": 169011, "sample_key_rows": 169011,
               "sample_label_rows": 0, "hidden_rows": 0, "old_answer_values_read": 0,
               "upload": 0, "manifest_sha256": sha(PACKAGE / "package-manifest.json")}
    if replay:
        first = read(DOCS / "answer.json")
        receipt["fresh_pid"] = first["pid"] != os.getpid()
        receipt["byte_exact_to_own_first"] = first["sha256"] == digest
    receipt["status"] = answer_status(frozen["kind"], receipt["matches_original_b2f17"],
        replay=replay, fresh_pid=receipt.get("fresh_pid", True),
        byte_exact=receipt.get("byte_exact_to_own_first", True))
    if frozen["kind"] == "cold":
        receipt["cold_to_this_stage_seconds"] = time.time() - read(DOCS / "cold-start.json")["started_unix"]
        receipt["training_qa_sha256"] = sha(DOCS / "training-qa.json")
    checked()
    write(DOCS / ("answer-replay-qa.json" if replay else "answer.json"), receipt)
    if receipt["status"] != "PASS":
        raise ValueError("answer replay mismatch; preserved; no retry")


def process(stage, data, until, directory=None, replay=False):
    """Kill only our newly spawned process tree on timeout, including Windows venv child."""
    command = [sys.executable, "-I", str(__file__), "_worker", "--stage", stage,
               "--data", str(data), "--until", str(until)]
    if directory is not None:
        command += ["--stage-dir", str(directory)]
    if replay:
        command += ["--replay"]
    prefix = ("replay-" if replay else "") + stage
    with (LOGS / (prefix + ".stdout.log")).open("x", encoding="utf-8") as out, (
            LOGS / (prefix + ".stderr.log")).open("x", encoding="utf-8") as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, cwd=PACKAGE)
        write(LOGS / (prefix + "-process.json"), {"pid": child.pid, "started": time.time(), "stage": stage})
        try:
            code = child.wait(timeout=max(0.01, until - time.time()))
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                child.kill()
            child.wait()
            raise TimeoutError("whole workflow deadline reached; no restart") from None
        if code != 0:
            raise RuntimeError(f"stage {stage} failed with exit {code}; inspect its preserved log")


def worker(args):
    remaining = args.until - time.time()
    if remaining <= 0:
        raise TimeoutError("whole deadline expired")
    timer = threading.Timer(remaining, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    try:
        runtime(args.stage)
        frozen = checked()
        if args.stage in ("tree-train", "tree-qa", "ms-train", "ms-qa", "training-finalise"):
            start = read(DOCS / "cold-start.json")
            if frozen["kind"] != "cold" or args.until > start["started_unix"] + frozen["budget"]["cold_seconds"]:
                raise ValueError("cold training worker must use its own bounded approved attempt")
            if args.stage == "tree-train" and any(MODELS.iterdir()):
                raise ValueError("tree training requires empty new model folder")
            if args.stage == "ms-train":
                tq = read(DOCS / "tree-replay-qa.json")
                if tq["status"] != "PASS" or tq["training_sha256"] != sha(DOCS / "tree-training.json"):
                    raise ValueError("MS training requires new tree replay QA first")
        os.environ["P1_DATA_DIR"] = str(args.data)
        if args.stage == "tree-train":
            tree_train(frozen, args.data)
        elif args.stage == "tree-qa":
            tree_qa(frozen, args.data)
        elif args.stage == "ms-train":
            ms_train(args.data)
        elif args.stage == "ms-qa":
            ms_qa()
        elif args.stage == "training-finalise":
            training_finalise(frozen, args.data)
        elif args.stage == "tree":
            tree_infer(frozen, args.data, args.stage_dir)
        elif args.stage == "mstcn":
            ms_infer(frozen, args.data, args.stage_dir)
        elif args.stage == "combine":
            combine(frozen, args.data, args.stage_dir, args.replay)
        else:
            raise ValueError("unknown stage")
        # Unchanged MS train/replay install their own narrower audit boundary;
        # the supervisor rechecks the outer package in a fresh next process.
        if args.stage not in ("ms-train", "ms-qa"):
            checked()
        if time.time() > args.until:
            raise TimeoutError("stage completed after deadline")
        if args.stage in ("tree", "mstcn"):
            write(args.stage_dir / (args.stage + "-stage.json"), {"status": "PASS", "pid": os.getpid(),
                  "sha256": sha(args.stage_dir / (args.stage + ".parquet")), "new_fits": 0})
    finally:
        timer.cancel()


def launch(mode, data, gpu_approved):
    frozen = checked()
    if mode == "train":
        empty_training_gate(frozen)
        if not gpu_approved:
            raise PermissionError("explicit GPU slot approval required")
        write(DOCS / "cold-start.json", {"started_unix": time.time(), "pid": os.getpid(),
              "budget": frozen["budget"], "empty_model_folder": True,
              "manifest_sha256": sha(PACKAGE / "package-manifest.json")})
    if frozen["kind"] == "cold":
        until = read(DOCS / "cold-start.json")["started_unix"] + frozen["budget"]["cold_seconds"]
    else:
        until = time.time() + frozen["budget"]["saved_command_seconds"]
    if time.time() >= until:
        raise TimeoutError("cold clock has expired; use a separate saved adapter, never reset it")
    write(DOCS / (mode + "-attempt.json"), {"started_unix": time.time(), "until": until, "pid": os.getpid()})
    try:
        if mode == "train":
            for stage in ("tree-train", "tree-qa", "ms-train", "ms-qa", "training-finalise"):
                process(stage, data, until)
        else:
            inference_gate(frozen)
            directory = LOGS / ("answer-replay-stages" if mode == "verify" else "answer-stages")
            directory.mkdir(exist_ok=False)
            for stage in ("tree", "mstcn", "combine"):
                process(stage, data, until, directory, mode == "verify")
        checked()
        write(DOCS / (mode + "-terminal.json"), {"status": "COMPLETE", "mode": mode,
              "pid": os.getpid(), "completed_unix": time.time(), "until": until,
              "elapsed_seconds": time.time() - read(DOCS / (mode + "-attempt.json"))["started_unix"]})
    except BaseException as error:
        write(DOCS / (mode + "-terminal.json"), {"status": "TERMINAL_TECHNICAL_FAILURE", "mode": mode,
              "error_type": type(error).__name__, "error": str(error), "pid": os.getpid(),
              "automatic_retry_authorized": False})
        raise
    print(json.dumps({"status": "COMPLETE", "mode": mode, "pid": os.getpid()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "infer", "verify", "_worker"))
    parser.add_argument("--data", type=Path, default=os.environ.get("P1_DATA_DIR"))
    parser.add_argument("--gpu-approved", action="store_true")
    parser.add_argument("--stage")
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--until", type=float)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.data is None:
        parser.error("set P1_DATA_DIR or supply --data; no implicit dataset search")
    args.data = args.data.resolve()
    if args.mode == "_worker":
        if args.until is None or args.stage is None:
            parser.error("worker requires bounded stage")
        worker(args)
    else:
        launch(args.mode, args.data, args.gpu_approved)
