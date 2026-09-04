"""Reassemble and verify optional P1 checkpoint parts inside the P1 package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    package = args.package_dir.resolve()
    manifest = json.loads((package / "model_parts" / "MANIFEST.json").read_text(encoding="utf-8"))
    output_dir = package / "reassembled_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    for model in manifest["models"]:
        target = output_dir / model["filename"]
        with target.open("wb") as output:
            for part in model["parts"]:
                path = package / "model_parts" / part["filename"]
                if sha256_file(path) != part["sha256"]:
                    raise RuntimeError(f"part hash mismatch: {path.name}")
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1 << 20), b""):
                        output.write(block)
        actual = sha256_file(target)
        if actual != model["sha256"]:
            raise RuntimeError(f"reassembled model hash mismatch: {target.name}")
        print({"model": target.name, "bytes": target.stat().st_size, "sha256": actual})


if __name__ == "__main__":
    main()
