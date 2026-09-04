"""Reassemble P1 large assets from the portal-safe split ZIP archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO


class ReassemblyError(RuntimeError):
    """Raised when an archive, path, size, or digest violates the manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_target(package_dir: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ReassemblyError(f"unsafe manifest path: {relative}")
    root = package_dir.resolve()
    target = (root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ReassemblyError(f"manifest path escapes package: {relative}") from exc
    return target


def safe_archive(upload_dir: Path, part_name: str) -> Path:
    pure = PurePosixPath(part_name)
    if pure.name != part_name or pure.is_absolute() or ".." in pure.parts:
        raise ReassemblyError(f"unsafe part name: {part_name}")
    root = upload_dir.resolve()
    archive = (root / f"P1_{part_name}.zip").resolve()
    try:
        archive.relative_to(root)
    except ValueError as exc:
        raise ReassemblyError(f"part archive escapes upload directory: {part_name}") from exc
    return archive


def copy_member(
    archive_path: Path,
    member_name: str,
    output: BinaryIO,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    if not archive_path.is_file():
        raise ReassemblyError(f"missing split archive: {archive_path.name}")
    digest = hashlib.sha256()
    total = 0
    with zipfile.ZipFile(archive_path) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        if len(files) != 1 or files[0].filename != member_name:
            raise ReassemblyError(f"unexpected ZIP member set: {archive_path.name}")
        if files[0].file_size != expected_bytes:
            raise ReassemblyError(f"unexpected ZIP member size: {archive_path.name}")
        with archive.open(files[0]) as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                total += len(block)
                if total > expected_bytes:
                    raise ReassemblyError(f"oversized split part: {archive_path.name}")
                digest.update(block)
                output.write(block)
    if total != expected_bytes or digest.hexdigest() != expected_sha256:
        raise ReassemblyError(f"split part integrity failure: {archive_path.name}")


def reassemble(
    upload_dir: str | Path,
    package_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    replace: bool = False,
) -> dict:
    upload = Path(upload_dir).resolve()
    package = Path(package_dir).resolve()
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else upload / "P1_REASSEMBLY_MANIFEST.json"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    records = []
    for file_spec in payload["files"]:
        target = safe_target(package, file_spec["source_relative_path"])
        if target.exists():
            if (
                target.stat().st_size == int(file_spec["source_bytes"])
                and sha256_file(target) == file_spec["source_sha256"]
            ):
                records.append(
                    {
                        "path": file_spec["source_relative_path"],
                        "status": "already_exact",
                        "sha256": file_spec["source_sha256"],
                    }
                )
                continue
            if not replace:
                raise FileExistsError(f"target exists with a different digest: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".reassembling")
        if temporary.exists():
            temporary.unlink()
        try:
            with temporary.open("xb") as output:
                for part in file_spec["parts"]:
                    archive_path = safe_archive(upload, part["path"])
                    copy_member(
                        archive_path,
                        part["path"],
                        output,
                        int(part["bytes"]),
                        part["sha256"],
                    )
            if (
                temporary.stat().st_size != int(file_spec["source_bytes"])
                or sha256_file(temporary) != file_spec["source_sha256"]
            ):
                raise ReassemblyError(
                    f"reassembled file integrity failure: {file_spec['source_relative_path']}"
                )
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        records.append(
            {
                "path": file_spec["source_relative_path"],
                "status": "reassembled",
                "sha256": file_spec["source_sha256"],
            }
        )
    return {"status": "P1_REASSEMBLY_PASS", "files": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            reassemble(
                args.upload_dir,
                args.package_dir,
                manifest_path=args.manifest,
                replace=args.replace,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
