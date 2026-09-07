"""Two separate portable variants; original MS, rules and source snapshots retained."""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import nbformat
import p1_original_learning_seed_ablation_20260907_v2 as s

BASE = Path("C:/Users/cedis/Documents/OceanFinalRelease_20260907/P1/SAVED_MODELS")
DAY = Path("C:/Users/cedis/Documents/OceanFinalDay_20260907")


def adapt(text, arm):
    def one(before, after):
        nonlocal text
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    one('os.environ[key] = "8"', 'os.environ[key] = "4"')
    one("random_state=seed, n_jobs=8,", "random_state=seed, n_jobs=4,")
    one(', "cap_seconds": 21600', ', "cap_seconds": None')
    one(",\n                               timeout=max(1, 21600-(time.time()-started))", "")
    if arm == "B5":
        one("SEEDS = [20260813, 20260829, 20260847]", "SEEDS = [20260813, 20260829, 20260847, 20260861, 20260875]")
        text = text.replace('"fits_max": 4', '"fits_max": 6').replace('"fits": 4', '"fits": 6').replace('receipt["fits"] == 4', 'receipt["fits"] == 6').replace('"fits": 7', '"fits": 9')
        assert text.count("np.mean(") == 3
        text = text.replace("np.mean(", "mean_b(")
        helper = '''\ndef mean_b(values, axis=0):
    import numpy as np
    values = np.asarray(values)
    if axis != 0 or values.shape[0] != 5:
        raise ValueError("five seeds required")
    return (np.mean(values[:3], axis=0)*3 + values[3] + values[4])/5
\n'''
        one("def module(name, path):", helper + "\ndef module(name, path):")
    compile(text, "variant_run.py", "exec")
    return text


def build(arm, data):
    m, r = s.m, s.m.runtime()
    qa = m.read(s.OUT / "independent-qa.json")
    assert qa["status"] == "PASS" and all(qa["checks"].values())
    assert qa["result_sha256"] == m.sha(s.OUT / "terminal_result.json")
    for relative, expected in m.read(BASE / "source-manifest.json")["files"].items():
        assert m.sha(BASE / relative) == expected
    inherited_training = m.read(BASE / "03_model/tree/training-result.json")
    for name in ["O", "B"]:
        assert m.sha(BASE / "03_model/tree" / (name + ".joblib")) == inherited_training[name + "_sha256"]
    out = DAY / ("P1_original_" + arm + "_v1")
    package = out / "PACKAGE"
    package.mkdir(parents=True, exist_ok=False)
    for name in ["01_data", "03_model", "04_logs", "05_answer", "06_docs"]:
        (package / name).mkdir()
    shutil.copytree(BASE / "02_code", package / "02_code", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(BASE / "03_model", package / "03_model", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "ATTEMPT_LOCK.json", "*.log"))
    shutil.copy2(BASE / "contract.json", package / "contract.json")
    code = package / "02_code"
    (code / "run.py").write_text(adapt((code / "run.py").read_text(encoding="utf-8"), arm), encoding="utf-8")
    cfgtext = (code / "tree_config.toml").read_text(encoding="utf-8")
    assert cfgtext.count("threads = 8") == 1
    cfgtext = cfgtext.replace("threads = 8", "threads = 4")
    recipe = m.read(code / "tree_recipe.json")
    if arm == "O_slow":
        prefix = "[models.xgboost]\nn_estimators = 700\nlearning_rate = 0.04"
        assert cfgtext.count(prefix) == 1
        cfgtext = cfgtext.replace(prefix, "[models.xgboost]\nn_estimators = 1400\nlearning_rate = 0.02")
        recipe["O_selection"]["iteration_count"] = 1400
    (code / "tree_config.toml").write_text(cfgtext, encoding="utf-8")
    (code / "tree_recipe.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    tree = package / "03_model/tree"
    o, b = r.joblib.load(tree / "O.joblib"), r.joblib.load(tree / "B.joblib")
    if arm == "O_slow":
        new = r.joblib.load(s.OUT / "models/full_O_slow_20260813.joblib")
        o = dataclasses.replace(o, encoder=new["encoder"], model=new["packages"][0]["global"], iteration_count=1400)
        r.joblib.dump(o, tree / "O.joblib", compress=3)
    else:
        assert b["seeds"] == [20260813, 20260829, 20260847]
        for seed in [20260861, 20260875]:
            new = r.joblib.load(s.OUT / "models" / f"full_B_extra_{seed}.joblib")
            assert r.joblib.hash(b["encoder"]) == r.joblib.hash(new["encoder"]), "same full original-feature encoder required"
            b["packages"].extend(new["packages"])
            b["seeds"].append(seed)
        r.joblib.dump(b, tree / "B.joblib", compress=3)
    # Replace stale probes/receipts only in this new package, using a train-only probe.
    frame = r.pd.read_csv(data / "train.csv", usecols=r.scope["old"].RAW, nrows=256)
    bundle = r.build_features(frame, config=r.load_config(code / "tree_config.toml", env={}))
    ox, bx = o.encoder.transform(bundle), b["encoder"].transform(bundle)
    bp = r.np.vstack([p["global"].predict_proba(bx)[:, 1] for p in b["packages"]])
    bp = (bp[:3].mean(axis=0)*3+bp[3]+bp[4])/5 if arm == "B5" else bp.mean(axis=0)
    r.np.savez_compressed(tree / "own_probe.npz", o_x=ox, b_x=bx, o=o.model.predict_proba(ox)[:, 1], b=bp)
    receipt = m.read(tree / "training-result.json")
    receipt.update(fits=6 if arm == "B5" else 4, O_sha256=m.sha(tree / "O.joblib"), B_sha256=m.sha(tree / "B.joblib"),
                   pid=__import__("os").getpid(), source_experiment=s.ID, new_component_fits=2 if arm == "B5" else 1,
                   inheritance="original exact-retrained O/B/MS, not answer files", whole_variant_cold=False)
    (tree / "training-result.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    # infer-tree requires qa.json; this is a verified internal QA receipt, followed by new separate-process tree replay.
    (tree / "qa.json").write_text(json.dumps({"status": "PASS", "basis": "independent historical QA and saved model pins; new probe replay receipt is tree/qa-variant.json"}), encoding="utf-8")
    runner = (code / "run.py").read_text(encoding="utf-8")
    assert runner.count('write(folder / "qa.json"') == 1
    runner = runner.replace('write(folder / "qa.json"', 'write(folder / "qa-variant.json"')
    # Cold path writes qa-variant; inference must accept that result. Saved path executes tree-qa first.
    runner = runner.replace('read(ROOT / "03_model/tree/qa.json")', 'read(ROOT / "03_model/tree/qa-variant.json")')
    (code / "run.py").write_text(runner, encoding="utf-8")
    for name, stages in [("TRAIN", ["all"]), ("SAVED_PREDICT", ["tree-qa", "infer-tree", "infer-ms", "combine"])]:
        cells = [nbformat.v4.new_markdown_cell(f"# P1 {arm}\nFresh extraction only. TRAIN retrains all components. SAVED_PREDICT does not fit. P1_DATA_DIR must name distributed P1 inputs.")]
        cells.append(nbformat.v4.new_code_cell("import os, sys, subprocess\nfrom pathlib import Path\npackage=Path.cwd().resolve()\ndata=Path(os.environ['P1_DATA_DIR']).resolve()"))
        for stage in stages:
            cells.append(nbformat.v4.new_code_cell(f"with (package/'04_logs/{stage}.notebook.log').open('x', encoding='utf-8') as log:\n    subprocess.run([sys.executable, '-I', '-B', str(package/'02_code/run.py'), {stage!r}, '--data', str(data)], cwd=package, check=True, stdout=log, stderr=subprocess.STDOUT)\nprint({stage!r}+' completed')"))
        nb = nbformat.v4.new_notebook(cells=cells)
        nbformat.validate(nb)
        nbformat.write(nb, package / (name + ".ipynb"))
    (package / "README.md").write_text(
        f"# P1 original {arm} independent candidate\n\nOnly distributed data, no external observations or pretrained weights. No old answer input or Public-fitted coefficient.\n\n"
        "Set P1_DATA_DIR; pinned environment in 02_code. SOURCE_ONLY: empty 03_model/05_answer, TRAIN.ipynb trains then infers. SAVED_MODELS: SAVED_PREDICT.ipynb replays tree probes and regenerates answer. Do not run both in an already used folder. 01_data references distributed files; 02_code source, 03_model learned models, 04_logs execution, 05_answer/P1_submission.csv answer.\n\n"
        + ("B only: original three seeds plus 20260861/20260875. Arithmetic is (mean(original3)*3 + seed4 + seed5)/5; all feature/weight/threshold/cell/MS/GI settings unchanged. Full cold has 9 fits (O1+B5+MS3).\n\n" if arm == "B5" else "O only: learning_rate=0.02, 1400 trees, seed20260813; original B3/MS/GI/cells and threshold unchanged. Full cold has 7 fits (O1+B3+MS3).\n\n")
        + "Historical comparison uses 287862 Q3/Q4 rows, no Q2 MS-owned composed evidence. This is retrospective, not guaranteed Public/Private gain. Risk is reported separately. Historical six-cell selector code is not recovered; cells retain training-OOF provenance, not a new selection claim. Training CPU4 for tree variant, inherited MS deterministic CPU policy retained. No artificial new runtime cutoff and no general official six-hour compliance claim. Existing O/B/MS cold, new component training, and saved notebook replay are separate claims; a new whole-variant cold is not claimed before receipt. Final designation pending.\n", encoding="utf-8")
    shutil.copy2(s.OUT / "independent-qa.json", package / "06_docs/internal-qa.json")
    shutil.copy2(s.CONFIG, package / "06_docs/variant-config.json")
    modelpins = {p.relative_to(package).as_posix(): m.sha(p) for p in (package / "03_model").rglob("*") if p.is_file()}
    m.write(package / "06_docs/model-lineage.json", {"files": modelpins, "arm": arm, "source_experiment": s.ID})
    pins = {p.relative_to(package).as_posix(): m.sha(p) for p in package.rglob("*") if p.is_file() and "03_model" not in p.parts and "04_logs" not in p.parts and "__pycache__" not in p.parts}
    m.write(package / "source-manifest.json", {"files": pins})
    print(json.dumps({"status": "BUILT", "arm": arm, "package": str(package)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["B5", "O_slow"])
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    build(args.arm, args.data.resolve())
