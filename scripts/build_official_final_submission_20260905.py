"""Build isolated P1/P2/P3 final-submission folders and <=50 MB upload files."""

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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "final_submission_20260905.json"
DEFAULT_OUT = ROOT / "artifacts" / "official_final_submission_20260905"
MAX_UPLOAD = 50_000_000
PART_BYTES = 45_000_000


class BuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise BuildError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise BuildError(f"{label} SHA drift: {actual} != {expected}")
    return path


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        return
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def copy_matches(patterns: Iterable[str], target_root: Path) -> list[str]:
    copied: list[str] = []
    for pattern in patterns:
        for source in sorted(ROOT.glob(pattern)):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            relative = source.relative_to(ROOT)
            copy_file(source, target_root / relative)
            copied.append(relative.as_posix())
    return copied


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def safe_replace_target(target: Path, replace: bool) -> Path:
    resolved = target.resolve()
    artifacts = (ROOT / "artifacts").resolve()
    try:
        resolved.relative_to(artifacts)
    except ValueError as exc:
        raise BuildError("output must stay inside repository artifacts/") from exc
    temporary = resolved.with_name(resolved.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    if resolved.exists():
        if not replace:
            raise FileExistsError(resolved)
        shutil.rmtree(resolved)
    temporary.mkdir(parents=True)
    return temporary


def problem_scaffold(problem: str, temporary: Path, contract: dict[str, Any]) -> Path:
    package = temporary / problem
    package.mkdir()
    copy_file(ROOT / "scripts" / "final_submission_20260905" / "common.py", package / "common.py")
    copy_file(
        ROOT / "scripts" / "final_submission_20260905" / problem / "run_submission.py",
        package / "run_submission.py",
    )
    copy_file(
        ROOT
        / "notebooks"
        / "final_submission_20260905"
        / problem
        / f"{problem}_final_submission.ipynb",
        package / f"{problem}_final_submission.ipynb",
    )
    (package / "outputs").mkdir()
    (package / "assets").mkdir()
    contract = dict(contract)
    contract.update(
        {
            "schema_version": f"ocean.{problem.lower()}.final_package.20260905.v1",
            "built_from_git_commit": git_commit(),
            "upload_per_file_limit_bytes": MAX_UPLOAD,
            "package_atomic": True,
        }
    )
    (package / "contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (package / "requirements.txt").write_text(
        "numpy==2.3.5\npandas==3.0.1\nnbformat==5.11.0\nnbclient==0.11.0\n",
        encoding="utf-8",
    )
    (package / "RUN_NOTEBOOK.ps1").write_text(
        f"$ErrorActionPreference = 'Stop'\n"
        f"if (-not $env:{problem}_DATA_DIR) {{ throw '{problem}_DATA_DIR is required' }}\n"
        f"python -m jupyter nbconvert --to notebook --execute {problem}_final_submission.ipynb "
        f"--output {problem}_final_submission.executed.ipynb --ExecutePreprocessor.timeout=900\n",
        encoding="utf-8-sig",
    )
    return package


def write_readme(problem: str, package: Path, contract: dict[str, Any], exact_note: str) -> None:
    (package / "README.md").write_text(
        f"# {problem} official final package\n\n"
        f"## 결론\n\n`{contract['candidate_id']}`를 재현·검증하는 독립 패키지입니다. "
        f"확인된 공식 public {contract['official_metric']}는 `{contract['official_public_metric_value']}`이고 "
        f"문제 배점 환산은 `{contract['official_points']}`점입니다.\n\n"
        "## 가장 중요한 규정\n\n"
        "이 패키지는 운영진 배포 데이터만 사용합니다. 외부 관측·재분석·예보 자료와 실제 관측 기반 "
        "사전학습 가중치를 사용하지 않았습니다. 사용된 학습 모델은 모두 배포 데이터로 scratch 학습했습니다. "
        "원본 배포 데이터는 재배포 금지이므로 패키지에 포함하지 않습니다.\n\n"
        "## 실행\n\n"
        f"1. `{problem}_DATA_DIR`를 운영진 배포 `{problem}` 데이터 폴더로 설정합니다.\n"
        f"2. 이 디렉터리를 작업 폴더로 열고 `{problem}_final_submission.ipynb`를 위에서 아래로 실행합니다.\n"
        f"3. `outputs/{problem}_submission.csv`와 `outputs/receipt.json`의 SHA를 확인합니다.\n\n"
        f"{exact_note}\n\n"
        "노트북과 `run_submission.py`는 네트워크 요청이나 업로드를 수행하지 않습니다. `source_audit/`에는 "
        "학습·추론 계보를 검토하기 위한 소스와 설정이 들어 있습니다.\n\n"
        "## 홈페이지 입력값\n\n"
        f"- 제목: `{contract['title']}`\n"
        f"- 한 줄 요약: {contract['summary']}\n"
        "- 저장소 URL: `https://github.com/choihyunjin1/-oceanaidata_track1`\n",
        encoding="utf-8",
    )


def build_p1(args: argparse.Namespace, temporary: Path, master: dict[str, Any]) -> Path:
    contract = dict(master["P1"])
    anchor = require_hash(args.p1_anchor, contract["anchor_sha256"], "P1 anchor")
    candidate_path = require_hash(args.p1_candidate, contract["candidate_sha256"], "P1 candidate")
    package = problem_scaffold("P1", temporary, contract)
    copy_file(anchor, package / "assets" / "e150_anchor.csv")
    keys = ["station", "year", "layer", "time"]
    anchor_frame = pd.read_csv(anchor, dtype={"station": "string", "time": "string"})
    candidate = pd.read_csv(candidate_path, dtype={"station": "string", "time": "string"})
    if not anchor_frame[keys].equals(candidate[keys]):
        raise BuildError("P1 anchor/candidate keys differ")
    changed = anchor_frame["label"].ne(candidate["label"])
    if int(changed.sum()) != 2 or not (
        anchor_frame.loc[changed, "label"].eq(0).all()
        and candidate.loc[changed, "label"].eq(1).all()
    ):
        raise BuildError("P1 final candidate is not the expected two-row add-only patch")
    patch_payload = {
        "schema_version": "p1.gi_spike2_patch.20260905.v1",
        "rows": candidate.loc[changed, keys].to_dict(orient="records"),
    }
    patch_path = package / "assets" / "gi_spike2_patch.json"
    patch_path.write_text(
        json.dumps(patch_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    local_contract = json.loads((package / "contract.json").read_text(encoding="utf-8"))
    local_contract["patch_sha256"] = sha256_file(patch_path)
    (package / "contract.json").write_text(
        json.dumps(local_contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    audit = package / "source_audit"
    copy_tree(ROOT / "src" / "p1_qc", audit / "src" / "p1_qc")
    copy_matches(
        [
            "scripts/run_p1_mstcn_e150_full_deployment_20260827_v1.py",
            "scripts/run_p1_incumbent_preserving_mstcn_asrf_v2.py",
            "configs/experiments/p1_mstcn_e150_full_deployment_20260827_v1.json",
            "configs/experiments/p1_incumbent_preserving_mstcn_asrf_v2.json",
            "reports/p1_mstcn_e150_full_deployment_20260827_v1/*",
        ],
        audit,
    )
    evidence = ROOT / "artifacts" / "p1_mstcn_e150_full_deployment_20260827_v1"
    for pattern in (
        "*history.json",
        "*receipt.json",
        "*prediction.npz",
        "full_encoder.json",
        "terminal_result.json",
        "independent_qa.json",
        "postexecution_independent_qa.json",
    ):
        for source in evidence.glob(pattern):
            copy_file(source, package / "model_evidence" / source.name)
    copy_file(
        ROOT / "scripts" / "reassemble_p1_final_weights_20260905.py",
        package / "reassemble_models.py",
    )
    split_models(evidence, package / "model_parts")
    write_readme(
        "P1",
        package,
        contract,
        "Exact mode uses the frozen scratch-trained ensemble output plus the registered two-row GI spike patch. "
        "Optional checkpoints are split into sub-50 MB parts and can be reconstructed with `reassemble_models.py`.",
    )
    return package


def split_models(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True)
    models: list[dict[str, Any]] = []
    for source in sorted(source_dir.glob("*_state.pt")):
        parts: list[dict[str, Any]] = []
        with source.open("rb") as handle:
            index = 0
            while True:
                block = handle.read(PART_BYTES)
                if not block:
                    break
                name = f"{source.name}.part{index:02d}"
                path = target_dir / name
                path.write_bytes(block)
                parts.append({"filename": name, "bytes": len(block), "sha256": sha256_file(path)})
                index += 1
        models.append(
            {
                "filename": source.name,
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "parts": parts,
            }
        )
    (target_dir / "MANIFEST.json").write_text(
        json.dumps({"part_bytes": PART_BYTES, "models": models}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def build_p2(args: argparse.Namespace, temporary: Path, master: dict[str, Any]) -> Path:
    contract = dict(master["P2"])
    anchor = require_hash(args.p2_anchor, contract["anchor_sha256"], "P2 anchor")
    candidate = require_hash(args.p2_candidate, contract["candidate_sha256"], "P2 candidate")
    package = problem_scaffold("P2", temporary, contract)
    copy_file(anchor, package / "assets" / "bin17_anchor.csv")
    copy_file(candidate, package / "assets" / "frozen_v52_candidate.csv")
    audit = package / "source_audit"
    copy_tree(ROOT / "src" / "p2_restore", audit / "src" / "p2_restore")
    copy_matches(
        [
            "00_ORGANIZER_DATA_POLICY.md",
            "scripts/materialize_p2_v52_score_priority_20260901_v1.py",
            "scripts/run_p2_*20260901*.py",
            "configs/experiments/p2_*20260901*.json",
            "configs/compliance/organizer_data_policy_20260901.json",
            "reports/p2_v52_score_priority_third_moment_input_gradient_20260901_v1/*",
            "reports/p2_v52_score_priority_deployment_20260901_v1/*",
            "reports/p2_masked_third_central_moment_profile_pooling_deepset_20260901_v50/*",
            "reports/p2_public_temperature_input_gradient_regularized_deepset_20260901_v23/*",
            "reports/p2_v23_official_submission_20260901_v1/*",
        ],
        audit,
    )
    write_readme(
        "P2",
        package,
        contract,
        "Exact mode freezes the already deployed three-fit scratch ensemble output because the historical run did not persist weights. "
        "The complete v52 training/materialization source and its evidence are included under `source_audit/`; this limitation is explicit, not hidden.",
    )
    return package


def build_p3(args: argparse.Namespace, temporary: Path, master: dict[str, Any]) -> Path:
    contract = dict(master["P3"])
    original = require_hash(
        args.p3_original, contract["original_component_sha256"], "P3 original component"
    )
    axis = require_hash(args.p3_axis, contract["axis_component_sha256"], "P3 axis component")
    require_hash(args.p3_candidate, contract["candidate_sha256"], "P3 candidate")
    package = problem_scaffold("P3", temporary, contract)
    copy_file(original, package / "assets" / "original_component.csv")
    copy_file(axis, package / "assets" / "axis_component.csv")
    audit = package / "source_audit"
    copy_tree(ROOT / "src" / "p3_wave", audit / "src" / "p3_wave")
    copy_matches(
        [
            "scripts/build_p3_refined_public_optimum_20260827.py",
            "scripts/reproduce_p3_submission.py",
            "configs/experiments/p3_corrected_fixed_long_shrink_v4.json",
            "configs/compliance/p3_clean_incumbent_20260901.json",
            "reports/p3_clean_incumbent_reset_20260901_v1/*",
        ],
        audit,
    )
    copy_tree(args.p3_axis_models, package / "models" / "axis")
    copy_tree(args.p3_original_models, package / "models" / "original")
    write_readme(
        "P3",
        package,
        contract,
        "Exact mode recomputes the frozen affine long-lead blend. Axis saved-weight inference is byte-exact. "
        "The original historical saved-weight replay has a documented <=0.0048767 m difference, so its exact deployed component is bundled together with both clean scratch-model bundles.",
    )
    return package


def import_runner(package: Path):
    sys.path.insert(0, str(package))
    try:
        spec = importlib.util.spec_from_file_location(
            f"final_{package.name}", package / "run_submission.py"
        )
        if spec is None or spec.loader is None:
            raise BuildError(f"cannot import runner for {package.name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def materialize_all(packages: dict[str, Path], data_dirs: dict[str, Path]) -> None:
    for problem, package in packages.items():
        module = import_runner(package)
        receipt = module.materialize(data_dirs[problem], package)
        if receipt["status"] != "READY_EXACT_NOT_UPLOADED":
            raise BuildError(f"{problem} materialization did not finish")


def execute_notebooks(packages: dict[str, Path], data_dirs: dict[str, Path]) -> None:
    import nbformat
    from nbclient import NotebookClient

    for problem, package in packages.items():
        notebook_path = package / f"{problem}_final_submission.ipynb"
        notebook = nbformat.read(notebook_path, as_version=4)
        old = os.environ.get(f"{problem}_DATA_DIR")
        os.environ[f"{problem}_DATA_DIR"] = str(data_dirs[problem])
        try:
            client = NotebookClient(
                notebook,
                timeout=900,
                kernel_name="python3",
                resources={"metadata": {"path": str(package)}},
            )
            client.execute()
        finally:
            if old is None:
                os.environ.pop(f"{problem}_DATA_DIR", None)
            else:
                os.environ[f"{problem}_DATA_DIR"] = old
        nbformat.write(notebook, package / f"{problem}_final_submission.executed.ipynb")


def zip_tree(source: Path, target: Path, *, exclude_model_parts: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if exclude_model_parts and relative.parts[0] == "model_parts":
                continue
            archive.write(path, Path(source.name) / relative)
    if target.stat().st_size > MAX_UPLOAD:
        raise BuildError(f"upload archive exceeds 50 MB: {target}")


def build_uploads(temporary: Path) -> list[dict[str, Any]]:
    upload = temporary / "upload"
    upload.mkdir()
    for problem in ("P1", "P2", "P3"):
        zip_tree(
            temporary / problem,
            upload / f"{problem}_official_final_core.zip",
            exclude_model_parts=problem == "P1",
        )
    parts = temporary / "P1" / "model_parts"
    for part in sorted(parts.glob("*.part??")):
        target = upload / f"{part.name}.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.write(part, part.name)
        if target.stat().st_size > MAX_UPLOAD:
            raise BuildError(f"P1 model part archive exceeds 50 MB: {target.name}")
    rows: list[dict[str, Any]] = []
    for path in sorted(upload.iterdir()):
        rows.append(
            {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return rows


def write_master(temporary: Path, uploads: list[dict[str, Any]], master: dict[str, Any]) -> None:
    receipts = {
        problem: json.loads(
            (temporary / problem / "outputs" / "receipt.json").read_text(encoding="utf-8")
        )
        for problem in ("P1", "P2", "P3")
    }
    payload = {
        "schema_version": "ocean.official_final_submission.20260905.v1",
        "status": "LOCAL_READY_EXACT_NOT_UPLOADED",
        "git_commit_at_build": git_commit(),
        "policy": master["policy"],
        "atomic_problem_directories": True,
        "notebooks_executed": all(
            (temporary / p / f"{p}_final_submission.executed.ipynb").is_file() for p in receipts
        ),
        "receipts": receipts,
        "upload_files": uploads,
    }
    (temporary / "MASTER_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (temporary / "README_FIRST.md").write_text(
        "# 공식 최종 제출 로컬 패키지\n\n"
        "결론: P1/P2/P3는 서로 독립된 디렉터리와 노트북으로 준비되어 있으며, 한 문제의 실행은 다른 문제의 파일을 읽거나 수정하지 않습니다.\n\n"
        "- 각 문제: `P?/README.md` -> `P?_final_submission.executed.ipynb` -> `outputs/receipt.json`\n"
        "- 홈페이지 업로드 후보: `upload/` (모든 개별 파일 50,000,000 bytes 이하)\n"
        "- 원본 배포 데이터, 외부 데이터, 비밀정보는 포함하지 않음\n"
        "- 실제 홈페이지 최종 제출은 자동 수행하지 않음\n",
        encoding="utf-8",
    )
    (temporary / "UPLOAD_CHECKLIST.md").write_text(
        "# 홈페이지 최종 제출 체크리스트\n\n"
        "1. `MASTER_MANIFEST.json` 상태가 `LOCAL_READY_EXACT_NOT_UPLOADED`인지 확인한다.\n"
        "2. 해당 문제의 `README.md`에 있는 제목과 한 줄 요약을 그대로 사용한다.\n"
        "3. 기본 업로드는 `upload/P?_official_final_core.zip`이다.\n"
        "4. P1 체크포인트까지 요구되면 `upload/*.pt.part??.zip` 15개도 함께 올린다. 각 파일은 50 MB 미만이다.\n"
        "5. 저장소 URL은 `https://github.com/choihyunjin1/-oceanaidata_track1`이다.\n"
        "6. 최종 버튼은 이후 답안 제출을 잠그므로 세 문제 모두 검토한 다음 문제별로 실행한다.\n"
        "7. 원본 배포 데이터, 외부 데이터, `.env`, 토큰, 캐시를 추가하지 않는다.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--execute-notebooks", action="store_true")
    parser.add_argument("--p1-data-dir", type=Path, required=True)
    parser.add_argument("--p2-data-dir", type=Path, required=True)
    parser.add_argument("--p3-data-dir", type=Path, required=True)
    parser.add_argument("--p1-anchor", type=Path, required=True)
    parser.add_argument("--p1-candidate", type=Path, required=True)
    parser.add_argument("--p2-anchor", type=Path, required=True)
    parser.add_argument("--p2-candidate", type=Path, required=True)
    parser.add_argument("--p3-original", type=Path, required=True)
    parser.add_argument("--p3-axis", type=Path, required=True)
    parser.add_argument("--p3-candidate", type=Path, required=True)
    parser.add_argument("--p3-axis-models", type=Path, required=True)
    parser.add_argument("--p3-original-models", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    master = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not master["policy"]["organizer_distributed_data_only"]:
        raise BuildError("top-level clean-data policy is not active")
    temporary = safe_replace_target(args.output_root, args.replace)
    packages = {
        "P1": build_p1(args, temporary, master),
        "P2": build_p2(args, temporary, master),
        "P3": build_p3(args, temporary, master),
    }
    data_dirs = {
        "P1": args.p1_data_dir.resolve(),
        "P2": args.p2_data_dir.resolve(),
        "P3": args.p3_data_dir.resolve(),
    }
    materialize_all(packages, data_dirs)
    if args.execute_notebooks:
        execute_notebooks(packages, data_dirs)
    uploads = build_uploads(temporary)
    write_master(temporary, uploads, master)
    temporary.replace(args.output_root.resolve())
    print(
        json.dumps(
            json.loads((args.output_root / "MASTER_MANIFEST.json").read_text(encoding="utf-8")),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
