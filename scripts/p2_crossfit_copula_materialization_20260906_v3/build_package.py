"""Freeze only fully evaluated retained arms into a new standalone CPU package."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESEARCH = REPO / "reports/p2_crossfit_copula_forward_numeric_20260906_v2"


def build(destination):
    if destination.exists():
        raise FileExistsError("new empty destination required")
    result = json.loads((RESEARCH / "result.json").read_text())
    qa = json.loads((RESEARCH / "independent-qa.json").read_text())
    replay = json.loads((RESEARCH / "replay.json").read_text())
    assert result["status"] == "COMPLETE_INTERNAL_8FOLD_CPU_ONLY" and qa["status"] == replay["status"] == "PASS"
    assert result["new_backbone_fits"] == 36 and result["new_copula_fits"] == 12
    selected = result["retained_mean_improvement_candidates"]
    if not selected:
        raise ValueError("no strict-primary-mean improvement; full materialization not warranted")
    assert all(result["natural"]["primary_B3"][arm]["rmse"] < result["natural"]["primary_B3"]["C3"]["rmse"] for arm in selected)
    spec = importlib.util.spec_from_file_location("p2_candidate_package", HERE / "02_code/run.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    assert qa["result_sha256"] == runtime.sha(RESEARCH / "result.json")
    assert qa["replay_sha256"] == runtime.sha(RESEARCH / "replay.json")
    cfg = json.loads((REPO / "scripts/portable_20260906/P2/config.json").read_text())
    cfg.update(package_id="p2_crossfit_copula_materialization_20260906_v3", device="cpu", cpu_threads=2,
        selected_arms=selected, correction_strength=1.0, training_cap_seconds=3600,
        backbone_fits=9 if "crossfit_full" in selected else 3, copula_fits=len(selected),
        prior_model_or_answer_reuse=False, new_full_training=True,
        selection_source="Completed paired 8fold historical B3 natural primary; both risk slices retained; not official feedback",
        full_inner_selection="Nonempty KST calendar months at indices floor((N-1)/3), floor(2*(N-1)/3); fixed7day purge, all other source data; no label-error selection",
        historical_result_sha256=runtime.sha(RESEARCH / "result.json"),
        historical_independent_qa_sha256=runtime.sha(RESEARCH / "independent-qa.json"))
    source = (Path(os.environ["P2_DATA_DIR"]) / "observations.csv").resolve()
    frame, _ = runtime.public_population(source, cfg, True)
    cfg["inner_plan"] = runtime.inner_plan(frame)
    cfg["training_key_sha256"] = runtime.keysha(frame)
    cfg["versions"] = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "torch", "scipy", "scikit-learn", "threadpoolctl")}
    cfg["versions"]["python"] = platform.python_version()
    for name in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
        (destination / name).mkdir(parents=True)
    for path in (HERE / "02_code").glob("*.py"):
        shutil.copyfile(path, destination / "02_code" / path.name)
    shutil.copyfile(HERE / "README.md", destination / "README.md")
    shutil.copyfile(HERE / "requirements.txt", destination / "requirements.txt")
    runtime.save(destination / "config.json", cfg)
    code_hashes = {p.relative_to(destination).as_posix(): runtime.sha(p) for p in sorted((destination / "02_code").glob("*.py"))}
    code_hashes["config.json"] = runtime.sha(destination / "config.json")
    runtime.save(destination / "06_docs/BUILD_MANIFEST.json", {"files_sha256": code_hashes,
        "source_data_copied": 0, "old_models_copied": 0, "old_predictions_copied": 0,
        "backbone_fits_planned": cfg["backbone_fits"], "copula_fits_planned": cfg["copula_fits"],
        "selected_arms": selected, "historical_result_sha256": cfg["historical_result_sha256"],
        "status": "FROZEN_EMPTY_MODEL_SOURCE_ONLY_FULL_TRAINING_PLAN"})
    print(json.dumps({"status": "SEALED_EMPTY_PACKAGE", "package": str(destination),
                     "arms": selected, "fits": cfg["backbone_fits"] + cfg["copula_fits"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.destination.resolve())
