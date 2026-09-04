"""Refresh final-package source surfaces and core ZIPs without retraining."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_official_final_submission_20260905.py"
DEFAULT_PACKAGE = ROOT / "artifacts/official_final_submission_20260905"


def load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("official_final_builder", BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import builder: {BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_package(path: Path) -> Path:
    resolved = path.resolve()
    artifacts = (ROOT / "artifacts").resolve()
    try:
        resolved.relative_to(artifacts)
    except ValueError as exc:
        raise RuntimeError("package must remain inside repository artifacts/") from exc
    for required in ("MASTER_MANIFEST.json", "P1", "P2", "P3", "upload"):
        if not (resolved / required).exists():
            raise FileNotFoundError(resolved / required)
    return resolved


def require_descendant(parent: Path, child: Path, label: str) -> Path:
    resolved_parent = parent.resolve()
    resolved_child = child.resolve()
    try:
        resolved_child.relative_to(resolved_parent)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes final package: {resolved_child}") from exc
    return resolved_child


def prune_and_copy_source(builder: Any, package: Path, problem: str) -> None:
    module_name = {"P1": "p1_qc", "P2": "p2_restore"}[problem]
    target = require_descendant(
        package,
        package / problem / "07_source/src" / module_name,
        f"{problem} source directory",
    )
    target.mkdir(parents=True, exist_ok=True)
    allowed = set(builder.PACKAGED_SOURCE_MODULES[problem])
    unexpected_dirs = [item for item in target.iterdir() if item.is_dir()]
    if unexpected_dirs:
        raise RuntimeError(f"unexpected source subdirectories: {unexpected_dirs}")
    for item in target.iterdir():
        if item.is_file() and item.name not in allowed:
            item.unlink()
    builder.copy_source_modules(problem, target)
    actual = {item.name for item in target.iterdir() if item.is_file()}
    if actual != allowed:
        raise RuntimeError(f"{problem} packaged source allowlist drift: {actual}")


def refresh(package_path: str | Path) -> dict:
    package = require_package(Path(package_path))
    builder = load_builder()
    for problem in ("P1", "P2"):
        prune_and_copy_source(builder, package, problem)
    p1_root = require_descendant(package, package / "P1", "P1 package directory")
    shutil.copy2(
        ROOT / "scripts/reassemble_p1_upload_20260905.py",
        p1_root / "REASSEMBLE_UPLOAD.py",
    )
    selection = json.loads(builder.CONFIG.read_text(encoding="utf-8"))
    for problem in builder.PROBLEMS:
        contract_path = package / problem / "contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        for field in ("title", "summary"):
            contract[field] = selection[problem][field]
        builder.write_json(contract_path, contract)
        builder.write_problem_docs(package / problem, problem)
    builder.purge_runtime_caches(
        {problem: package / problem for problem in builder.PROBLEMS}
    )

    upload = require_descendant(package, package / "upload", "upload directory")
    core_records = []
    for problem in builder.PROBLEMS:
        target = upload / f"{problem}_official_final_core.zip"
        temporary = target.with_name(target.name + ".refreshing")
        if temporary.exists():
            temporary.unlink()
        builder.zip_core(package / problem, temporary)
        os.replace(temporary, target)
        with zipfile.ZipFile(target) as archive:
            entries = set(archive.namelist())
        if problem in builder.PACKAGED_SOURCE_MODULES:
            module_name = {"P1": "p1_qc", "P2": "p2_restore"}[problem]
            prefix = f"{problem}/07_source/src/{module_name}/"
            actual = {
                Path(name).name
                for name in entries
                if name.startswith(prefix) and name.endswith(".py")
            }
            expected = set(builder.PACKAGED_SOURCE_MODULES[problem])
            if actual != expected:
                raise RuntimeError(f"{problem} core source closure drift: {actual}")
        core_records.append(builder.file_record(target, upload))

    master_path = package / "MASTER_MANIFEST.json"
    master = json.loads(master_path.read_text(encoding="utf-8"))
    master["upload_files"] = [
        builder.file_record(path, upload)
        for path in sorted(upload.iterdir())
        if path.is_file()
    ]
    master["core_source_allowlist"] = {
        problem: list(names)
        for problem, names in builder.PACKAGED_SOURCE_MODULES.items()
    }
    master["core_archives_refreshed_from_git_commit"] = builder.git_commit()
    builder.write_json(master_path, master)
    return {
        "status": "FINAL_UPLOAD_CORE_REFRESH_PASS",
        "package": str(package),
        "core_archives": core_records,
        "upload_file_count": len(master["upload_files"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    print(json.dumps(refresh(args.package_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
