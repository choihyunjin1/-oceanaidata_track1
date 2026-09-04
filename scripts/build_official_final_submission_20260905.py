"""Build three isolated, model-driven official-final submission packages.

The generated local tree contains organizer-data hardlinks, executable training
and inference notebooks, actual model weights, and byte-exact answer files. Raw
organizer files are deliberately excluded from every upload archive.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "final_submission_20260905.json"
DEFAULT_OUT = ROOT / "artifacts" / "official_final_submission_20260905"
MAX_UPLOAD = 50_000_000
PART_BYTES = 45_000_000
PROBLEMS = ("P1", "P2", "P3")
PACKAGED_SOURCE_MODULES = {
    # These are the complete import closures used by the authoritative
    # 02_train/ and 04_predict/ entrypoints.  Historical external-data modules
    # remain in Git as audit evidence, but must not enter the final package.
    "P1": (
        "__init__.py",
        "ms_tcn_asrf.py",
        "ms_tcn_asrf_data.py",
    ),
    "P2": (
        "__init__.py",
        "data.py",
        "features.py",
        "normalized_curvature_residual.py",
        "submission.py",
    ),
}


class BuildError(RuntimeError):
    """Raised when a final-package invariant fails."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def require_hash(path: str | Path, expected: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise BuildError(f"missing {label}: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected:
        raise BuildError(f"{label} SHA drift: {actual} != {expected}")
    return resolved


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def copy_file(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def link_file(source: Path, target: Path) -> tuple[Path, str]:
    """Create a space-efficient local hardlink, falling back to a normal copy."""

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        mode = "hardlink"
    except OSError:
        shutil.copy2(source, target)
        mode = "copy"
    return target, mode


def copy_tree(
    source: Path, target: Path, *, extra_ignore_patterns: tuple[str, ...] = ()
) -> None:
    if not source.is_dir():
        raise BuildError(f"missing source tree: {source}")
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache", *extra_ignore_patterns
        ),
    )


def copy_source_modules(problem: str, target: Path) -> None:
    """Copy only the dependency-closed source surface used by a final package."""

    names = PACKAGED_SOURCE_MODULES.get(problem)
    if names is None:
        raise BuildError(f"no packaged source allowlist for {problem}")
    source = ROOT / "src" / {"P1": "p1_qc", "P2": "p2_restore"}[problem]
    for name in names:
        copy_file(source / name, target / name)


def copy_relative(relative: str, destination_root: Path) -> Path:
    source = ROOT / relative
    if not source.is_file():
        raise BuildError(f"missing repository source: {source}")
    return copy_file(source, destination_root / relative)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def safe_temporary_target(target: Path, replace: bool) -> Path:
    resolved = target.resolve()
    artifacts = (ROOT / "artifacts").resolve()
    try:
        resolved.relative_to(artifacts)
    except ValueError as exc:
        raise BuildError("output must remain inside repository artifacts/") from exc
    temporary = resolved.with_name(resolved.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    if resolved.exists():
        if not replace:
            raise FileExistsError(resolved)
        shutil.rmtree(resolved)
    temporary.mkdir(parents=True)
    return temporary


def scaffold(problem: str, temporary: Path, base_contract: dict[str, Any]) -> Path:
    package = temporary / problem
    for name in (
        "01_data/organizer_dataset",
        "02_train",
        "03_model/weights",
        "03_model/decision_artifacts",
        "04_predict",
        "05_answer",
        "06_submission",
        "07_source",
    ):
        (package / name).mkdir(parents=True, exist_ok=True)
    runtime = ROOT / "scripts" / "final_submission_20260905"
    copy_file(runtime / "common.py", package / "common.py")
    copy_file(runtime / problem / "run_submission.py", package / "run_submission.py")
    for name in ("train_model.py",):
        copy_file(runtime / problem / name, package / "02_train" / name)
    copy_file(runtime / problem / "predict_submission.py", package / "04_predict" / "predict_submission.py")
    pipeline = runtime / problem / f"{problem.lower()}_pipeline.py"
    if pipeline.is_file():
        copy_file(pipeline, package / "02_train" / pipeline.name)
        copy_file(pipeline, package / "04_predict" / pipeline.name)
    notebook_root = ROOT / "notebooks" / "final_submission_20260905" / problem
    copy_file(notebook_root / "TRAIN.ipynb", package / "02_train" / "TRAIN.ipynb")
    copy_file(notebook_root / "PREDICT.ipynb", package / "04_predict" / "PREDICT.ipynb")
    contract = dict(base_contract)
    contract.update(
        {
            "schema_version": f"ocean.{problem.lower()}.model_driven_final.20260905.v2",
            "built_from_git_commit": git_commit(),
            "package_atomic": True,
            "answer_must_come_from_model_inference": True,
            "frozen_candidate_csv_used_as_prediction_input": False,
            "upload_per_file_limit_bytes": MAX_UPLOAD,
        }
    )
    write_json(package / "contract.json", contract)
    requirements = {
        "P1": "numpy\npandas\npyarrow\nscikit-learn\ntorch\n",
        "P2": "numpy\npandas\nscikit-learn\ntorch\n",
        "P3": "numpy\npandas\npyarrow\nscikit-learn\ncatboost==1.2.10\njoblib\n",
    }
    (package / "requirements.txt").write_text(requirements[problem], encoding="utf-8")
    (package / "RUN_INFERENCE.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "$notebookDir = Join-Path $PSScriptRoot '04_predict'\n"
        "Push-Location $notebookDir\n"
        "try { python -m jupyter nbconvert --to notebook --execute PREDICT.ipynb "
        "--output PREDICT.executed.ipynb --ExecutePreprocessor.timeout=1800 } finally { Pop-Location }\n",
        encoding="utf-8-sig",
    )
    retrain_outputs = {
        "P1": "03_model/retrained_from_scratch",
        "P2": "03_model/retrained_from_notebook",
        "P3": "03_model/retrained_base_reference",
    }
    (package / "RUN_TRAINING.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "$dataDir = Join-Path $PSScriptRoot '01_data\\organizer_dataset'\n"
        "$trainer = Join-Path $PSScriptRoot '02_train\\train_model.py'\n"
        f"$outputDir = Join-Path $PSScriptRoot '{retrain_outputs[problem].replace('/', chr(92))}'\n"
        "& python $trainer --data-dir $dataDir --package-dir $PSScriptRoot --output-dir $outputDir\n"
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n",
        encoding="utf-8-sig",
    )
    return package


def populate_official_data(
    package: Path, problem: str, data_dir: Path, contract: dict[str, Any]
) -> None:
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise BuildError(f"missing {problem} organizer data directory: {data_dir}")
    files = []
    for name, expected in contract["official_inputs"].items():
        source = require_hash(data_dir / name, expected, f"{problem} organizer input {name}")
        target, mode = link_file(source, package / "01_data" / "organizer_dataset" / name)
        record = file_record(target, package)
        record["storage"] = mode
        files.append(record)
    write_json(
        package / "01_data" / "INPUT_MANIFEST.json",
        {
            "schema_version": f"ocean.{problem.lower()}.organizer_inputs.20260905.v1",
            "source": "organizer_distributed_dataset_only",
            "redistribute_in_upload": False,
            "files": files,
        },
    )
    (package / "01_data" / "LOCAL_DATA_PATH.txt").write_text(
        str(data_dir) + "\n", encoding="utf-8"
    )
    (package / "01_data" / "README.md").write_text(
        f"# {problem} data boundary\n\n"
        "`organizer_dataset/` contains local hardlinks (or copies if hardlink creation failed) "
        "of the organizer-distributed files. They are present so the local package is runnable, "
        "but are excluded from Git and upload archives because redistribution is not allowed. "
        "`INPUT_MANIFEST.json` binds every input by filename, byte count, and SHA-256.\n",
        encoding="utf-8",
    )


def update_contract(package: Path, **fields: Any) -> dict[str, Any]:
    path = package / "contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract.update(fields)
    write_json(path, contract)
    return contract


def build_patch(anchor: Path, candidate: Path) -> dict[str, Any]:
    dtype = {"station": "string", "time": "string"}
    left = pd.read_csv(anchor, dtype=dtype)
    right = pd.read_csv(candidate, dtype=dtype)
    keys = ["station", "year", "layer", "time"]
    if not left[keys].equals(right[keys]):
        raise BuildError("P1 model anchor/final candidate key order differs")
    changed = left["label"].ne(right["label"])
    if int(changed.sum()) != 2:
        raise BuildError("P1 final candidate must differ by exactly two rows")
    if not left.loc[changed, "label"].eq(0).all() or not right.loc[changed, "label"].eq(1).all():
        raise BuildError("P1 final patch must be add-only")
    return {
        "schema_version": "p1.gi_spike2_patch.20260905.v2",
        "rows": right.loc[changed, keys].to_dict(orient="records"),
    }


def build_p1(args: argparse.Namespace, temporary: Path, master: dict[str, Any]) -> Path:
    base = dict(master["P1"])
    anchor = require_hash(args.p1_anchor, base["anchor_sha256"], "P1 trained-model anchor")
    candidate = require_hash(args.p1_candidate, base["candidate_sha256"], "P1 champion")
    router = require_hash(args.p1_router_anchor, args.p1_router_sha256, "P1 router anchor")
    package = scaffold("P1", temporary, base)
    populate_official_data(package, "P1", args.p1_data_dir, base)

    derived_sources = {
        "train_features.parquet": ROOT / "artifacts/cache/train_offline_e9fe1eb46cb7431f.parquet",
        "train_features.json": ROOT / "artifacts/cache/train_offline_e9fe1eb46cb7431f.json",
        "test_features.parquet": ROOT / "artifacts/cache/test_offline_c2a3877bdecea937.parquet",
        "test_features.json": ROOT / "artifacts/cache/test_offline_c2a3877bdecea937.json",
    }
    derived_files = []
    for name, source in derived_sources.items():
        if not source.is_file():
            raise BuildError(f"missing P1 historical derived surface: {source}")
        target, mode = link_file(source, package / "01_data" / "derived" / name)
        record = file_record(target, package)
        record["storage"] = mode
        derived_files.append(record)

    decision_dir = package / "03_model" / "decision_artifacts"
    router_target = copy_file(router, decision_dir / "router_anchor.csv")
    patch_target = decision_dir / "gi_spike2_patch.json"
    write_json(patch_target, build_patch(anchor, candidate))
    decision_files = [file_record(router_target, package), file_record(patch_target, package)]

    evidence = ROOT / "artifacts" / "p1_mstcn_e150_full_deployment_20260827_v1"
    model_files = []
    checkpoint_records = []
    for source in sorted(evidence.glob("full_width_512_seed_*_epoch_150_state.pt")):
        target, mode = link_file(source, package / "03_model" / "weights" / source.name)
        record = file_record(target, package)
        record["storage"] = mode
        model_files.append(record)
        checkpoint_records.append(record)
    if len(checkpoint_records) != 3:
        raise BuildError("P1 requires exactly three certified e150 checkpoints")
    provenance = package / "03_model" / "training_provenance"
    for pattern in ("*history.json", "*receipt.json", "terminal_result.json", "independent_qa.json"):
        for source in sorted(evidence.glob(pattern)):
            copy_file(source, provenance / source.name)
    model_manifest = package / "03_model" / "MODEL_MANIFEST.json"
    write_json(
        model_manifest,
        {
            "status": "CERTIFIED_HISTORICAL_SCRATCH_TRAINING_OUTPUT",
            "training_source": "07_source/scripts/run_p1_mstcn_e150_full_deployment_20260827_v1.py",
            "architecture_source": "07_source/scripts/run_p1_incumbent_preserving_mstcn_asrf_v2.py",
            "training_rows": 776706,
            "fit_count": 3,
            "seeds": [20260827, 20260839, 20260863],
            "epochs_per_fit": 150,
            "input_features": 165,
            "pretrained_weights_loaded": 0,
            "external_data_rows": 0,
            "checkpoints": checkpoint_records,
            "answer_is_recomputed_from_checkpoints": True,
            "expected_pre_patch_model_output_sha256": base["anchor_sha256"],
        },
    )
    model_files.append(file_record(model_manifest, package))

    copy_source_modules("P1", package / "07_source/src/p1_qc")
    copy_file(
        ROOT / "scripts/reassemble_p1_upload_20260905.py",
        package / "REASSEMBLE_UPLOAD.py",
    )
    for relative in (
        "scripts/run_p1_mstcn_e150_full_deployment_20260827_v1.py",
        "scripts/run_p1_incumbent_preserving_mstcn_asrf_v2.py",
        "configs/experiments/p1_mstcn_e150_full_deployment_20260827_v1.json",
        "configs/experiments/p1_incumbent_preserving_mstcn_asrf_v2.json",
        "00_ORGANIZER_DATA_POLICY.md",
    ):
        copy_relative(relative, package / "07_source")
    update_contract(
        package,
        router_anchor_sha256=args.p1_router_sha256,
        derived_files=derived_files,
        decision_files=decision_files,
        model_files=model_files,
        model_training_execution=(
            "historical_three_fit_scratch_training_certified_by_checkpoint_and_provenance"
        ),
        retraining_entrypoint_scope="full_three_seed_mstcn_training_to_separate_output",
    )
    write_problem_docs(package, "P1")
    return package


def import_module(path: Path, name: str):
    parent = str(path.parent)
    sys.path.insert(0, parent)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise BuildError(f"cannot import module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def build_p2(args: argparse.Namespace, temporary: Path, master: dict[str, Any]) -> Path:
    base = dict(master["P2"])
    anchor = require_hash(args.p2_anchor, base["anchor_sha256"], "P2 bin17 anchor")
    require_hash(args.p2_candidate, base["candidate_sha256"], "P2 champion reference")
    package = scaffold("P2", temporary, base)
    populate_official_data(package, "P2", args.p2_data_dir, base)
    anchor_target = copy_file(anchor, package / "03_model/decision_artifacts/bin17_anchor.csv")

    copy_source_modules("P2", package / "07_source/src/p2_restore")
    p2_sources = (
        "scripts/run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12.py",
        "scripts/run_p2_prefix_safe_domain_balanced_deepset_20260901_v13.py",
        "scripts/run_p2_public_temperature_input_gradient_regularized_deepset_20260901_v23.py",
        "scripts/run_p2_masked_third_central_moment_profile_pooling_deepset_20260901_v50.py",
        "scripts/run_p2_v52_score_priority_third_moment_input_gradient_20260901_v1.py",
        "scripts/materialize_p2_v52_score_priority_20260901_v1.py",
        "configs/experiments/p2_v52_score_priority_third_moment_input_gradient_20260901_v1.json",
        "configs/experiments/p2_v52_score_priority_deployment_20260901_v1.json",
        "00_ORGANIZER_DATA_POLICY.md",
    )
    for relative in p2_sources:
        copy_relative(relative, package / "07_source")

    trainer = import_module(package / "02_train/train_model.py", "p2_final_trainer")
    manifest = trainer.train(args.p2_data_dir, package, "03_model/weights")
    if manifest.get("fit_count") != 3:
        raise BuildError("P2 fresh training did not complete exactly three fits")
    model_files = [
        file_record(path, package)
        for path in sorted((package / "03_model/weights").glob("v52_seed_*.pt"))
    ]
    model_files.append(file_record(package / "03_model/MODEL_MANIFEST.json", package))
    update_contract(
        package,
        decision_files=[file_record(anchor_target, package)],
        model_files=model_files,
        model_training_execution="fresh_three_fit_training_during_package_build",
        retraining_entrypoint_scope="full_three_seed_v52_training",
        allow_documented_replay_variance=True,
        historical_champion_sha256=base["candidate_sha256"],
    )
    first_receipt = import_runner(package).materialize(
        package / "01_data/organizer_dataset", package
    )
    update_contract(
        package,
        model_replay_sha256=first_receipt["sha256"],
        historical_champion_hash_exact=first_receipt["historical_champion_hash_exact"],
    )
    write_problem_docs(package, "P2")
    return package


def build_p3(args: argparse.Namespace, temporary: Path, master: dict[str, Any]) -> Path:
    base = dict(master["P3"])
    require_hash(args.p3_original, base["original_component_sha256"], "P3 original reference")
    require_hash(args.p3_axis, base["axis_component_sha256"], "P3 axis reference")
    require_hash(args.p3_candidate, base["candidate_sha256"], "P3 champion reference")
    package = scaffold("P3", temporary, base)
    populate_official_data(package, "P3", args.p3_data_dir, base)
    model_files = []

    original_target = package / "03_model/weights/original"
    original_map = {
        "model.cbm": "model.cbm",
        "model_multi.cbm": "model_multi.cbm",
        "feature_columns.json": "feature_columns.json",
    }
    for source_name, target_name in original_map.items():
        source = args.p3_original_models.resolve() / source_name
        target, _mode = link_file(source, original_target / target_name)
        model_files.append(file_record(target, package))
    router_source = require_hash(
        args.p3_original_router,
        "28ee70f2092565872446e089f6d169428af1a4686a572d3f5f1e1deeaaf78117",
        "P3 original router",
    )
    target, _mode = link_file(router_source, original_target / "router.joblib")
    model_files.append(file_record(target, package))

    axis_target = package / "03_model/weights/axis"
    for name in ("single.cbm", "multi.cbm", "router.joblib", "calibrator.joblib", "feature_columns.json"):
        source = args.p3_axis_models.resolve() / name
        if not source.is_file():
            raise BuildError(f"missing P3 axis model file: {source}")
        target, _mode = link_file(source, axis_target / name)
        model_files.append(file_record(target, package))

    copy_tree(
        ROOT / "src/p3_wave",
        package / "07_source/src/p3_wave",
        extra_ignore_patterns=(
            "kma_*.py",
            "era5_*.py",
            "chronos2_transfer.py",
            "champion_matched_era5_*.py",
        ),
    )
    p3_sources = (
        "scripts/train_predict_p3_final.py",
        "scripts/run_p3_component_loss_router.py",
        "scripts/build_p3_loss_router_submission.py",
        "scripts/run_p3_corrected_repeated_forward_catboost_v1.py",
        "scripts/run_p3_corrected_repeated_forward_catboost_v2.py",
        "scripts/run_p3_corrected_fixed_long_shrink_v4.py",
        "scripts/build_p3_refined_public_optimum_20260827.py",
        "configs/experiments/p3_corrected_repeated_forward_catboost_v2.json",
        "configs/experiments/p3_corrected_fixed_long_shrink_v4.json",
        "configs/compliance/p3_clean_incumbent_20260901.json",
        "00_ORGANIZER_DATA_POLICY.md",
    )
    for relative in p3_sources:
        copy_relative(relative, package / "07_source")
    for relative in p3_sources:
        if relative.startswith(("scripts/", "configs/experiments/")):
            copy_relative(relative, package / "02_train" / "exact_lineage_source")
    copy_file(
        args.p3_original_models.resolve() / "manifest.json",
        package / "03_model/training_provenance/original_model_manifest.json",
    )
    copy_file(
        args.p3_original_router.resolve().parent / "manifest.json",
        package / "03_model/training_provenance/original_router_manifest.json",
    )
    copy_file(
        ROOT / "artifacts/p3_corrected_repeated_forward_catboost_v2/manifest.json",
        package / "03_model/training_provenance/axis_base_manifest.json",
    )
    copy_file(
        ROOT / "artifacts/p3_corrected_fixed_long_shrink_v4/manifest.json",
        package / "03_model/training_provenance/axis_calibration_manifest.json",
    )
    copy_file(
        ROOT / "artifacts/p3_corrected_fixed_long_shrink_v4/metrics.json",
        package / "03_model/training_provenance/axis_calibration_metrics.json",
    )
    model_manifest = package / "03_model/MODEL_MANIFEST.json"
    write_json(
        model_manifest,
        {
            "status": "CERTIFIED_HISTORICAL_SCRATCH_TRAINING_OUTPUT",
            "fit_lineages": 2,
            "model_files": model_files,
            "training_sources": [
                "07_source/scripts/train_predict_p3_final.py",
                "07_source/scripts/run_p3_component_loss_router.py",
                "07_source/scripts/build_p3_loss_router_submission.py",
                "07_source/scripts/run_p3_corrected_repeated_forward_catboost_v2.py",
                "07_source/scripts/run_p3_corrected_fixed_long_shrink_v4.py",
            ],
            "training_source_sha256": {
                relative: sha256_file(package / relative)
                for relative in (
                    "07_source/scripts/train_predict_p3_final.py",
                    "07_source/scripts/run_p3_component_loss_router.py",
                    "07_source/scripts/build_p3_loss_router_submission.py",
                    "07_source/scripts/run_p3_corrected_repeated_forward_catboost_v2.py",
                    "07_source/scripts/run_p3_corrected_fixed_long_shrink_v4.py",
                )
            },
            "training_provenance_files": [
                file_record(path, package)
                for path in sorted(
                    (package / "03_model/training_provenance").glob("*.json")
                )
            ],
            "pretrained_weights_loaded": 0,
            "external_data_rows": 0,
            "answer_is_recomputed_from_saved_model_chains": True,
            "expected_component_sha256": {
                "original": base["original_component_sha256"],
                "axis": base["axis_component_sha256"],
            },
        },
    )
    model_files.append(file_record(model_manifest, package))
    update_contract(
        package,
        model_files=model_files,
        decision_files=[],
        derived_files=[],
        model_training_execution=(
            "historical_two_scratch_chain_training_certified_by_model_hash_and_provenance"
        ),
        retraining_entrypoint_scope=(
            "base_model_reference_entrypoint_plus_exact_router_and_calibrator_sources"
        ),
    )
    write_problem_docs(package, "P3")
    return package


def submission_form(problem: str, contract: dict[str, Any]) -> dict[str, Any]:
    schemas = {
        "P1": [
            {"name": "station", "type": "string"},
            {"name": "year", "type": "integer"},
            {"name": "layer", "type": "integer"},
            {"name": "time", "type": "ISO-8601 string"},
            {"name": "label", "type": "integer", "allowed": [0, 1]},
        ],
        "P2": [
            {"name": "station", "type": "string"},
            {"name": "layer", "type": "integer"},
            {"name": "time", "type": "ISO-8601 string"},
            {"name": "temp", "type": "finite float", "unit": "C"},
        ],
        "P3": [
            {"name": "case_id", "type": "string"},
            {"name": "station", "type": "string"},
            {"name": "lead_h", "type": "integer", "allowed": [3, 6, 9, 12, 18, 24]},
            {"name": "hs_pred", "type": "finite float", "unit": "m"},
        ],
    }
    historical_exact = bool(contract.get("historical_champion_hash_exact", True))
    notes = {
        "P1": "운영진 배포 데이터만 사용한 scratch 학습 계보. 모델 가중치와 재현 코드를 동봉했습니다.",
        "P2": (
            "운영진 배포 데이터만 사용한 scratch 3-fit replay입니다. 역사적 공식 최고점은 "
            "별도 SHA의 답안에 귀속되며 현재 replay의 공식 점수라고 주장하지 않습니다."
        ),
        "P3": "운영진 배포 데이터만 사용한 scratch 학습 계보. 모델 가중치와 재현 코드를 동봉했습니다.",
    }
    return {
        "problem": problem,
        "candidate_id": contract["candidate_id"],
        "data_directory": "01_data/organizer_dataset",
        "training_entrypoint": "02_train/train_model.py",
        "training_notebook": "02_train/TRAIN.ipynb",
        "model_directory": "03_model",
        "model_manifest": "03_model/MODEL_MANIFEST.json",
        "prediction_entrypoint": "04_predict/predict_submission.py",
        "prediction_notebook": "04_predict/PREDICT.ipynb",
        "answer_file": f"05_answer/{problem}_submission.csv",
        "answer_sha256": contract.get("model_replay_sha256", contract["candidate_sha256"]),
        "historical_champion_sha256": contract["candidate_sha256"],
        "answer_is_byte_exact_historical_champion": historical_exact,
        "score_provenance": (
            "official_public_score_of_this_exact_answer"
            if historical_exact
            else "same_recipe_fresh_model_replay_unscored; historical score is reference only"
        ),
        "rows": contract["expected_rows"],
        "columns_in_exact_order": contract["expected_columns"],
        "column_contract": schemas[problem],
        "website_form": {
            "title": contract["title"],
            "one_line_summary": contract["summary"],
            "repository_url": "https://github.com/choihyunjin1/-oceanaidata_track1",
            "result_url": "",
            "notes": notes[problem],
        },
    }


def write_problem_docs(package: Path, problem: str) -> None:
    contract = json.loads((package / "contract.json").read_text(encoding="utf-8"))
    form = submission_form(problem, contract)
    write_json(package / "06_submission/FORM.json", form)
    columns = ", ".join(f"`{name}`" for name in contract["expected_columns"])
    historical_exact = bool(contract.get("historical_champion_hash_exact", True))
    (package / "06_submission/FORMAT.md").write_text(
        f"# {problem} 제출 양식\n\n"
        "- 데이터 위치: `01_data/organizer_dataset/` (로컬 전용)\n"
        "- 학습 진입점: `02_train/train_model.py` / `02_train/TRAIN.ipynb`\n"
        "- 모델 위치: `03_model/` / manifest `03_model/MODEL_MANIFEST.json`\n"
        "- 추론 진입점: `04_predict/predict_submission.py` / `04_predict/PREDICT.ipynb`\n"
        f"- 파일: `05_answer/{problem}_submission.csv`\n"
        f"- 정확한 열 순서: {columns}\n"
        f"- 행 수: `{contract['expected_rows']}`\n"
        f"- 현재 model replay SHA-256: `{contract.get('model_replay_sha256', contract['candidate_sha256'])}`\n"
        f"- 역사적 공식 최고점 CSV SHA-256: `{contract['candidate_sha256']}`\n"
        f"- 현재 답안이 역사적 최고점과 byte-exact인가: `{historical_exact}`\n"
        "- 키 순서는 운영진 `sample_submission.csv`와 정확히 같아야 하며 index 열을 쓰지 않습니다.\n"
        "- 홈페이지 제목·한 줄 요약·저장소 URL은 `FORM.json`을 그대로 사용합니다.\n",
        encoding="utf-8",
    )
    caveats = {
        "P1": (
            "최종값은 세 MS-TCN 체크포인트의 실제 추론을 router anchor와 union한 뒤, "
            "등록된 GI spike 2행을 추가해 생성합니다. 학습 당시의 165-feature 파생 표면은 "
            "원본 입력 해시에 묶인 로컬 전용 캐시입니다. 포털의 파일당 50 MB 제한 때문에 "
            "가중치와 파생 표면은 별도 part ZIP으로 나뉘며, core ZIP의 "
            "`REASSEMBLE_UPLOAD.py`가 manifest의 part/source SHA를 검증하며 복원합니다."
        ),
        "P2": (
            "빌드 과정에서 v52 세 seed를 실제로 새로 학습해 체크포인트를 저장했고, 최종값은 "
            "그 체크포인트 추론과 bin17 anchor의 고정 20/80 결합으로 생성합니다. 역사적 최고점 실행은 "
            "체크포인트를 저장하지 않았고 GPU 재학습의 수치 경로가 달라질 수 있으므로, 역사적 공식 점수와 "
            "SHA는 참고 근거로만 남기고 현재 model replay SHA를 별도로 고정합니다."
        ),
        "P3": (
            "최종값은 두 개의 저장된 CatBoost+router scratch 계보를 실제 추론한 뒤, 12/18/24h에만 "
            "봉인 alpha를 적용해 생성합니다. 과거 KMA/ERA5 계보는 포함하지 않습니다."
        ),
    }
    replay_statement = (
        "현재 답안은 저장 모델 추론으로 역사적 최고점 CSV와 byte-exact 재현됩니다."
        if historical_exact
        else (
            "현재 답안은 새로 학습해 저장한 모델의 재현 가능한 출력입니다. 역사적 최고점 실행은 "
            "가중치를 남기지 않아 그 공식 점수와 현재 답안의 점수 동일성을 주장하지 않습니다."
        )
    )
    training_lineage = {
        "P1": (
            "# P1 학습 계보\n\n"
            "`train_model.py`는 운영진 `train.csv`와 입력 SHA에 묶인 165-feature 표면으로 "
            "MS-TCN을 seed 20260827/20260839/20260863, 각 150 epoch scratch 학습해 별도 "
            "출력 폴더에 체크포인트를 저장합니다. 현재 certified `03_model/weights`는 이 과정을 "
            "이미 완료한 저장 산출물이며, build에서는 3개 checkpoint와 학습 provenance를 검증했습니다. "
            "답안은 이 checkpoint들을 실제 로드한 추론 뒤 등록된 add-only 2행 decision artifact를 적용합니다.\n"
        ),
        "P2": (
            "# P2 학습 계보\n\n"
            "`train_model.py`는 운영진 `observations.csv`로 v52 모델을 seed 20260901/20260902/20260903, "
            "각 60 epoch scratch 학습합니다. 이번 package build에서 이 3 fit을 실제 실행해 "
            "`03_model/weights`를 새로 만들었고, 답안은 그 checkpoint 추론과 고정 bin17 anchor를 "
            "결합해 생성했습니다. 역사적 최고점 실행의 checkpoint가 없으므로 역사적 공식 점수는 "
            "현재 모델 답안의 점수라고 주장하지 않습니다.\n"
        ),
        "P3": (
            "# P3 학습 계보\n\n"
            "`train_model.py`는 운영진 train wave/atmos로 단일·다중 CatBoost base model을 scratch "
            "재학습하는 안전한 별도-output 진입점입니다. 최고점 exact 계보의 두 full chain은 "
            "`exact_lineage_source/`와 `../07_source/`에 보존한 역사적 trainer/config가 base models, "
            "chronological router, fixed calibrator를 생성했고, 해당 실행 manifest와 model SHA는 "
            "`03_model/training_provenance`에 있습니다. 현재 답안은 그 9개 저장 산출물을 실제 로드해 "
            "추론한 뒤 봉인된 long-lead affine combination으로 생성합니다.\n"
        ),
    }[problem]
    (package / "02_train" / "TRAINING_LINEAGE.md").write_text(
        training_lineage,
        encoding="utf-8",
    )
    (package / "README.md").write_text(
        f"# {problem} 독립 공식 최종 패키지\n\n"
        "## 결론\n\n"
        f"규정 준수 계보 중 확인된 최고 공식 후보 `{contract['candidate_id']}`의 학습 recipe 패키지입니다. "
        f"공식 public {contract['official_metric']}는 `{contract['official_public_metric_value']}`, "
        f"역사적 환산 점수는 `{contract['official_points']}`점입니다. frozen 최종 CSV를 예측 입력으로 "
        f"복사하지 않습니다. {replay_statement}\n\n"
        "## 폴더 계약\n\n"
        "- `01_data/`: 운영진 배포 데이터(local only)와 입력 해시\n"
        "- `02_train/`: scratch 학습 코드와 `TRAIN.ipynb`\n"
        "- `03_model/`: 학습 산출 가중치, training manifest, 허용된 decision artifacts\n"
        "- `04_predict/`: 모델 추론 코드와 `PREDICT.ipynb`\n"
        f"- `05_answer/`: 모델이 만든 `{problem}_submission.csv`와 검증 receipt\n"
        "- `06_submission/`: 열 순서·dtype·홈페이지 입력 양식\n"
        "- `07_source/`: 정확한 역사적 학습 계보 소스와 규정 문서\n\n"
        f"{caveats[problem]}\n\n"
        "원본 배포 데이터는 로컬 실행 트리에는 존재하지만 Git과 upload ZIP에서는 제외됩니다. "
        "외부 관측·재분석·예보 자료, 실제 관측 기반 pretrained weights, hidden truth는 사용하지 않습니다.\n",
        encoding="utf-8",
    )


def import_runner(package: Path):
    return import_module(package / "run_submission.py", f"final_{package.name.lower()}_runner")


def materialize_all(packages: dict[str, Path]) -> None:
    for problem, package in packages.items():
        data_dir = package / "01_data/organizer_dataset"
        receipt = import_runner(package).materialize(data_dir, package)
        if receipt.get("status") != "READY_MODEL_INFERENCE_EXACT_NOT_UPLOADED":
            raise BuildError(f"{problem} model-driven materialization failed")


def execute_notebooks(packages: dict[str, Path]) -> None:
    import nbformat
    from nbclient import NotebookClient

    for _problem, package in packages.items():
        for relative in ("02_train/TRAIN.ipynb", "04_predict/PREDICT.ipynb"):
            path = package / relative
            notebook = nbformat.read(path, as_version=4)
            client = NotebookClient(
                notebook,
                timeout=1800,
                kernel_name="python3",
                resources={"metadata": {"path": str(path.parent)}},
            )
            client.execute()
            nbformat.write(notebook, path.with_name(path.stem + ".executed.ipynb"))


def purge_runtime_caches(packages: dict[str, Path]) -> None:
    """Remove interpreter caches from the finished, user-facing packages."""

    for package in packages.values():
        for cache in sorted(package.rglob("__pycache__"), reverse=True):
            shutil.rmtree(cache)
        for cache in sorted(package.rglob(".pytest_cache"), reverse=True):
            shutil.rmtree(cache)


def split_binary(source: Path, target_dir: Path, logical_root: Path) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    with source.open("rb") as handle:
        index = 0
        while True:
            block = handle.read(PART_BYTES)
            if not block:
                break
            safe_name = source.relative_to(logical_root).as_posix().replace("/", "__")
            path = target_dir / f"{safe_name}.part{index:02d}"
            path.write_bytes(block)
            parts.append(file_record(path, target_dir))
            index += 1
    return {
        "source_relative_path": source.relative_to(logical_root).as_posix(),
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "parts": parts,
    }


def excluded_from_core(problem: str, relative: Path) -> bool:
    posix = relative.as_posix()
    if posix.startswith("upload_parts/"):
        return True
    if posix.startswith("01_data/organizer_dataset/") or posix == "01_data/LOCAL_DATA_PATH.txt":
        return True
    if problem == "P1" and (
        posix.startswith("01_data/derived/") or posix.startswith("03_model/weights/")
    ):
        return True
    package_names = {"P1": "p1_qc", "P2": "p2_restore"}
    package_name = package_names.get(problem)
    if package_name is not None and relative.parts[:3] == (
        "07_source",
        "src",
        package_name,
    ):
        return relative.name not in PACKAGED_SOURCE_MODULES[problem]
    return False


def zip_core(package: Path, target: Path) -> None:
    problem = package.name
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(package)
            if excluded_from_core(problem, relative) or "__pycache__" in relative.parts:
                continue
            archive.write(path, Path(problem) / relative)
    if target.stat().st_size > MAX_UPLOAD:
        raise BuildError(f"core archive exceeds 50 MB: {target}")


def build_uploads(temporary: Path) -> list[dict[str, Any]]:
    upload = temporary / "upload"
    upload.mkdir()
    for problem in PROBLEMS:
        zip_core(temporary / problem, upload / f"{problem}_official_final_core.zip")
    split_root = temporary / "P1/upload_parts"
    split_manifest = []
    for directory in (
        temporary / "P1/03_model/weights",
        temporary / "P1/01_data/derived",
    ):
        for source in sorted(directory.glob("*")):
            if source.is_file():
                split_manifest.append(split_binary(source, split_root, temporary / "P1"))
    write_json(split_root / "REASSEMBLY_MANIFEST.json", {"files": split_manifest})
    for part in sorted(split_root.glob("*.part??")):
        target = upload / f"P1_{part.name}.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.write(part, part.name)
        if target.stat().st_size > MAX_UPLOAD:
            raise BuildError(f"split archive exceeds 50 MB: {target}")
    copy_file(split_root / "REASSEMBLY_MANIFEST.json", upload / "P1_REASSEMBLY_MANIFEST.json")
    return [file_record(path, upload) for path in sorted(upload.iterdir()) if path.is_file()]


def write_master(temporary: Path, uploads: list[dict[str, Any]], master: dict[str, Any]) -> None:
    receipts = {
        problem: json.loads((temporary / problem / "05_answer/receipt.json").read_text(encoding="utf-8"))
        for problem in PROBLEMS
    }
    payload = {
        "schema_version": "ocean.official_final.model_driven.20260905.v2",
        "status": "LOCAL_READY_MODEL_INFERENCE_NOT_UPLOADED",
        "git_commit_at_build": git_commit(),
        "policy": master["policy"],
        "atomic_problem_directories": True,
        "frozen_candidate_csv_used_as_prediction_input": False,
        "training_notebooks_present": True,
        "training_notebooks_executed_in_manifest_audit_mode": all(
            (temporary / problem / "02_train/TRAIN.executed.ipynb").is_file()
            for problem in PROBLEMS
        ),
        "prediction_notebooks_executed": all(
            (temporary / problem / "04_predict/PREDICT.executed.ipynb").is_file()
            for problem in PROBLEMS
        ),
        "historical_champion_hash_exact": {
            problem: receipts[problem].get("historical_champion_hash_exact", True)
            for problem in PROBLEMS
        },
        "receipts": receipts,
        "upload_files": uploads,
    }
    write_json(temporary / "MASTER_MANIFEST.json", payload)
    (temporary / "README_FIRST.md").write_text(
        "# 공식 최종 제출 로컬 패키지\n\n"
        "P1/P2/P3가 각각 `01_data`부터 `07_source`까지 독립되어 있습니다. 모든 `05_answer` CSV는 "
        "frozen CSV 복사가 아니라 `03_model`의 실제 가중치를 `04_predict`가 로드해 생성합니다. P1/P3는 "
        "기존 최고점 SHA와 정확히 일치합니다. P2는 역사적 최고점 실행이 weights를 남기지 않아 같은 recipe를 "
        "새로 3-fit한 model replay SHA를 별도 고정하며, 이 차이를 숨기지 않습니다.\n\n"
        "각 문제의 정확한 제출 양식은 `P?/06_submission/FORM.json`과 `FORMAT.md`를 확인하세요.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--execute-notebooks", action="store_true")
    parser.add_argument("--p1-data-dir", type=Path, required=True)
    parser.add_argument("--p2-data-dir", type=Path, required=True)
    parser.add_argument("--p3-data-dir", type=Path, required=True)
    parser.add_argument("--p1-router-anchor", type=Path, required=True)
    parser.add_argument(
        "--p1-router-sha256",
        default="1b04e81c18d5a5cac3115c3a256e8d5a38a9493a32478a184df81fd99f9f6e5f",
    )
    parser.add_argument("--p1-anchor", type=Path, required=True)
    parser.add_argument("--p1-candidate", type=Path, required=True)
    parser.add_argument("--p2-anchor", type=Path, required=True)
    parser.add_argument("--p2-candidate", type=Path, required=True)
    parser.add_argument("--p3-original", type=Path, required=True)
    parser.add_argument("--p3-axis", type=Path, required=True)
    parser.add_argument("--p3-candidate", type=Path, required=True)
    parser.add_argument("--p3-axis-models", type=Path, required=True)
    parser.add_argument("--p3-original-models", type=Path, required=True)
    parser.add_argument("--p3-original-router", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    master = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not master["policy"]["organizer_distributed_data_only"]:
        raise BuildError("organizer-data-only policy is not active")
    temporary = safe_temporary_target(args.output_root, args.replace)
    packages = {
        "P1": build_p1(args, temporary, master),
        "P2": build_p2(args, temporary, master),
        "P3": build_p3(args, temporary, master),
    }
    materialize_all(packages)
    if args.execute_notebooks:
        execute_notebooks(packages)
    purge_runtime_caches(packages)
    uploads = build_uploads(temporary)
    write_master(temporary, uploads, master)
    temporary.replace(args.output_root.resolve())
    print((args.output_root / "MASTER_MANIFEST.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
