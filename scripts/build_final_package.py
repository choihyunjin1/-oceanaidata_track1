"""Build a deterministic, code-only P1 reproduction archive.

The organizer's redistribution rule means that source observations, candidate
submissions, trained weights, and caches must never enter this archive.  This
builder therefore uses an explicit allowlist and records hashes against logical
archive paths only.  It does not upload the resulting file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "final_package" / "p1_qc_code.zip"

ALLOWLIST_GLOBS = (
    "00_MUST_READ_FIRST.md",
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dl.txt",
    "requirements-lock.txt",
    "configs/*.toml",
    "src/p1_qc/**/*.py",
    "scripts/bootstrap_env.ps1",
    "scripts/smoke_cuda.py",
    "scripts/analyze_oof_failures.py",
    "scripts/benchmark_data_io.py",
    "scripts/validate_submission.py",
    "scripts/build_final_package.py",
    "tests/**/*.py",
    "notebooks/*.ipynb",
    "reports/P1_IMPLEMENTATION_GUIDE.md",
    "reports/EXTERNAL_DATA_APPROVAL_DRAFT.md",
    "reports/ENVIRONMENT_2026-08-13.md",
    "reports/P1_RECONNAISSANCE_2026-08-13.md",
    "reports/P1_MODEL_SELECTION_2026-08-13.md",
    "reports/P1_FAILURE_RECON_2026-08-13.md",
    "reports/P1_DATA_LOADING_BENCHMARK_2026-08-13.md",
    "reports/P1_BREAKTHROUGH_RESEARCH_2026-08-13.md",
)

REQUIRED_LOGICAL_FILES = (
    "00_MUST_READ_FIRST.md",
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "configs/p1.toml",
    "src/p1_qc/__init__.py",
    "src/p1_qc/__main__.py",
    "reports/ENVIRONMENT_2026-08-13.md",
    "scripts/analyze_oof_failures.py",
    "scripts/benchmark_data_io.py",
    "reports/P1_FAILURE_RECON_2026-08-13.md",
    "reports/P1_DATA_LOADING_BENCHMARK_2026-08-13.md",
    "reports/P1_BREAKTHROUGH_RESEARCH_2026-08-13.md",
)

FORBIDDEN_SUFFIXES = {
    ".csv",
    ".zip",
    ".parquet",
    ".joblib",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
}
FORBIDDEN_PARTS = {
    "artifacts",
    "cache",
    "caches",
    "data",
    "datasets",
    "models",
    "outputs",
    "submissions",
    "데이터셋 원본",
}
TEXT_SUFFIXES = {".py", ".toml", ".md", ".txt", ".ps1", ".json", ".ipynb"}
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class PackagePolicyError(ValueError):
    """Raised when a would-be archive member violates the package policy."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _absolute_path_offset(text: str) -> int | None:
    """Return the first likely machine-local absolute-path offset, if any."""

    separator = chr(92)
    offsets: list[int] = []

    def can_start_at(index: int) -> bool:
        return index == 0 or (not text[index - 1].isalnum() and text[index - 1] not in {":", "/"})

    for index in range(max(0, len(text) - 2)):
        previous_ok = index == 0 or not text[index - 1].isalnum()
        if (
            previous_ok
            and text[index].isascii()
            and text[index].isalpha()
            and text[index + 1] == ":"
            and text[index + 2] in {separator, "/"}
        ):
            offsets.append(index)

    unc = separator * 2
    cursor = 0
    while (index := text.find(unc, cursor)) >= 0:
        server_start = index + len(unc)
        if server_start < len(text) and text[server_start].isalnum():
            next_separator = text.find(separator, server_start)
            if next_separator > server_start:
                offsets.append(index)
                break
        cursor = server_start

    local_roots = (
        "Users",
        "home",
        "root",
        "tmp",
        "var",
        "opt",
        "etc",
        "srv",
        "workspace",
        "data",
        "abs",
        "private" + "/" + "var",
    )
    for local_root in local_roots:
        prefix = "/" + local_root + "/"
        cursor = 0
        while (index := text.find(prefix, cursor)) >= 0:
            if can_start_at(index):
                offsets.append(index)
                break
            cursor = index + 1
    mount_prefix = "/" + "mnt" + "/"
    cursor = 0
    while (index := text.find(mount_prefix, cursor)) >= 0:
        drive_offset = index + len(mount_prefix)
        if (
            can_start_at(index)
            and drive_offset + 1 < len(text)
            and text[drive_offset].isascii()
            and text[drive_offset].isalpha()
            and text[drive_offset + 1] == "/"
        ):
            offsets.append(index)
            break
        cursor = drive_offset
    cursor = 0
    home_prefix = "~" + "/"
    while (index := text.find(home_prefix, cursor)) >= 0:
        if can_start_at(index):
            offsets.append(index)
            break
        cursor = index + 1
    return min(offsets) if offsets else None


def _logical_path(project_root: Path, path: Path) -> str:
    resolved_root = project_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise PackagePolicyError(f"allowlisted file escapes project root: {path}") from exc
    logical = PurePosixPath(relative.as_posix())
    if logical.is_absolute() or ".." in logical.parts:
        raise PackagePolicyError(f"unsafe archive path: {logical}")
    try:
        logical.as_posix().encode("ascii")
    except UnicodeEncodeError as exc:
        raise PackagePolicyError(f"archive filename must be ASCII: {logical}") from exc
    return logical.as_posix()


def _validate_member(logical: str, path: Path, content: bytes) -> None:
    logical_path = PurePosixPath(logical)
    lowered_parts = {part.casefold() for part in logical_path.parts}
    forbidden_parts = {part.casefold() for part in FORBIDDEN_PARTS}
    if lowered_parts & forbidden_parts:
        raise PackagePolicyError(f"forbidden package directory: {logical}")
    if logical_path.suffix.casefold() in FORBIDDEN_SUFFIXES:
        raise PackagePolicyError(f"forbidden package file type: {logical}")
    if path.is_symlink():
        raise PackagePolicyError(f"symbolic links are not packaged: {logical}")
    if logical_path.suffix.casefold() not in TEXT_SUFFIXES:
        return
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PackagePolicyError(f"text file is not UTF-8: {logical}") from exc
    offset = _absolute_path_offset(text)
    if offset is not None:
        line = text.count("\n", 0, offset) + 1
        raise PackagePolicyError(
            f"local absolute path reference in {logical}:{line}; use P1_DATA_DIR or a relative path"
        )


def collect_allowlisted_files(
    project_root: Path,
    *,
    patterns: Sequence[str] = ALLOWLIST_GLOBS,
    required: Sequence[str] = REQUIRED_LOGICAL_FILES,
) -> list[tuple[str, Path, bytes]]:
    """Collect and validate a stable list of logical archive members."""

    project_root = project_root.resolve()
    matched: dict[str, tuple[Path, bytes]] = {}
    for pattern in patterns:
        for path in project_root.glob(pattern):
            if not path.is_file():
                continue
            logical = _logical_path(project_root, path)
            content = path.read_bytes()
            _validate_member(logical, path, content)
            matched[logical] = (path, content)

    missing = sorted(set(required) - set(matched))
    if missing:
        raise PackagePolicyError(f"required reproduction files are missing: {missing}")
    return [(logical, *matched[logical]) for logical in sorted(matched)]


def _git_metadata(project_root: Path) -> dict[str, str | bool | None]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _manifest(project_root: Path, members: Iterable[tuple[str, Path, bytes]]) -> dict[str, object]:
    files: list[dict[str, object]] = []
    tree_hasher = hashlib.sha256()
    for logical, _path, content in members:
        content_hash = _sha256(content)
        logical_hash = _sha256(logical.encode("utf-8") + b"\0" + content)
        files.append(
            {
                "logical_path": logical,
                "bytes": len(content),
                "sha256": content_hash,
                "logical_sha256": logical_hash,
            }
        )
        tree_hasher.update(logical.encode("utf-8"))
        tree_hasher.update(b"\0")
        tree_hasher.update(bytes.fromhex(content_hash))
    return {
        "format": "p1-qc-code-only-v1",
        "source_git": _git_metadata(project_root),
        "tree_sha256": tree_hasher.hexdigest(),
        "files": files,
        "excludes_source_data": True,
        "input_contract": "Set P1_DATA_DIR to the organizer-provided P1 directory at runtime.",
    }


def _zip_entry(name: str, content: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info, content


def build_package(project_root: Path, output: Path, *, force: bool = False) -> dict[str, object]:
    """Build the archive and return a local receipt containing its SHA-256."""

    project_root = project_root.expanduser().resolve()
    output = output.expanduser().resolve()
    sidecar = output.with_suffix(".manifest.json")
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing package without --force: {output}")
    if sidecar.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing manifest without --force: {sidecar}")
    members = collect_allowlisted_files(project_root)
    manifest = _manifest(project_root, members)
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip.tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for logical, _path, content in members:
                info, payload = _zip_entry(logical, content)
                archive.writestr(info, payload, compresslevel=9)
            info, payload = _zip_entry("PACKAGE_MANIFEST.json", manifest_bytes)
            archive.writestr(info, payload, compresslevel=9)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    sidecar.write_bytes(manifest_bytes)
    receipt = {
        "package": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output.read_bytes()),
        "manifest": str(sidecar),
        "manifest_sha256": _sha256(manifest_bytes),
        "tree_sha256": manifest["tree_sha256"],
        "file_count": len(members),
        "uploaded": False,
    }
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = build_package(args.project_root, args.output, force=args.force)
    except (FileExistsError, PackagePolicyError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
