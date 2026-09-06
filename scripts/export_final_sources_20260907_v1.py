"""Export source-only release packages to Git without runtime artifacts or weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from verify_final_candidate_bundle_20260906_v1 import cold_archive_check


def export(release, p3_zip, destination):
    if destination.exists():
        raise FileExistsError("preserve existing export")
    destination.mkdir(parents=True)
    for problem in ("P1", "P2"):
        archive = release / problem / f"{problem}_SOURCE_ONLY.zip"
        cold_archive_check(archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(destination / problem)
    cold_archive_check(p3_zip)
    with zipfile.ZipFile(p3_zip) as z:
        prefix = "P3_numeric_cold_v1/"
        for item in z.infolist():
            if not item.filename.startswith(prefix):
                raise ValueError("unexpected P3 package root")
            relative = item.filename[len(prefix):]
            if not relative:
                continue
            target = destination / "P3" / relative
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(item) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    pins = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            if path.suffix.lower() not in {".py", ".json", ".md", ".txt", ".toml", ".ipynb"}:
                raise ValueError("non-source export")
            pins[path.relative_to(destination).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (destination / "SOURCE_EXPORT_MANIFEST.json").open("x", encoding="utf-8") as output:
        json.dump({"status": "SOURCE_ONLY_EXPORT", "files": pins, "models": 0, "source_data": 0, "answers": 0}, output, indent=2)
    print(json.dumps({"files": len(pins), "status": "SOURCE_ONLY_EXPORT"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--p3-zip", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    export(args.release.resolve(), args.p3_zip.resolve(), args.destination.resolve())
