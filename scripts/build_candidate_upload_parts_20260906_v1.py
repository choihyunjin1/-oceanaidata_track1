"""Split an existing candidate ZIP into verified ZIP attachments; stdlib only."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

CHUNK = 40_000_000
ATTACHMENT_CAP = 45_000_000


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split(source, output, chunk_size=CHUNK):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.suffix.lower() != ".zip" or not zipfile.is_zipfile(source):
        raise ValueError("an existing valid ZIP is required")
    if not 1 <= chunk_size <= CHUNK:
        raise ValueError("invalid chunk size")
    source_sha = sha(source)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with source.open("rb") as stream:
        while block := stream.read(chunk_size):
            name = f"part{len(records) + 1:03d}.zip"
            target = output / name
            with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("payload.bin", block)
            if target.stat().st_size > ATTACHMENT_CAP:
                raise ValueError("attachment cap exceeded")
            records.append({"name": name, "bytes": target.stat().st_size,
                            "sha256": sha(target), "payload_bytes": len(block),
                            "payload_sha256": hashlib.sha256(block).hexdigest()})
    if sha(source) != source_sha:
        raise ValueError("source changed during partitioning")
    manifest = {"format": "ocean-candidate-zip-parts-v1", "original_name": source.name,
                "original_bytes": source.stat().st_size, "original_sha256": source_sha,
                "attachment_cap_bytes": ATTACHMENT_CAP,
                "portal_limit_live_verified": False, "parts": records}
    with (output / "REASSEMBLY_MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    shutil.copy2(Path(__file__), output / "reassemble.py")
    (output / "README.md").write_text(
        "# Reassembly\n\nThese attachments contain pieces of one ZIP, not standalone models. "
        "Keep all partNNN.zip files beside REASSEMBLY_MANIFEST.json and reassemble.py.\n\n"
        "Run: `python reassemble.py --reassemble REASSEMBLY_MANIFEST.json --output restored.zip`\n\n"
        "Then extract restored.zip and follow its problem-specific README. "
        "The source archive is unchanged. A 45 MB local attachment cap is conservative planning, "
        "not proof of the portal's current upload limit. No upload or final model lock is performed.\n",
        encoding="utf-8",
    )
    return manifest


def validate_parts(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    if meta.get("format") != "ocean-candidate-zip-parts-v1" or not meta.get("parts"):
        raise ValueError("invalid manifest")
    combined, total = hashlib.sha256(), 0
    resolved = []
    for i, record in enumerate(meta["parts"], 1):
        if record["name"] != f"part{i:03d}.zip":
            raise ValueError("noncanonical or missing part order")
        path = manifest_path.parent / record["name"]
        if path.is_symlink() or path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("part bytes/hash mismatch")
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != ["payload.bin"]:
                raise ValueError("part must contain exactly one payload")
            block = archive.read("payload.bin")
        if len(block) != record["payload_bytes"] or hashlib.sha256(block).hexdigest() != record["payload_sha256"]:
            raise ValueError("payload mismatch")
        combined.update(block)
        total += len(block)
        resolved.append(path)
    if total != meta["original_bytes"] or combined.hexdigest() != meta["original_sha256"]:
        raise ValueError("whole archive hash mismatch")
    return meta, resolved


def reassemble(manifest_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("new destination required")
    meta, paths = validate_parts(manifest_path)
    with output.open("xb") as stream:
        for path in paths:
            with zipfile.ZipFile(path) as archive:
                stream.write(archive.read("payload.bin"))
    if sha(output) != meta["original_sha256"] or not zipfile.is_zipfile(output):
        raise ValueError("reassembled ZIP verification failed; preserve output for diagnosis")
    return {"status": "BYTE_EXACT_ARCHIVE_REASSEMBLY_PASS", "sha256": sha(output),
            "bytes": output.stat().st_size, "model_retraining": False, "uploads": 0}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--archive", type=Path)
    group.add_argument("--reassemble", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = split(args.archive, args.output) if args.archive else reassemble(args.reassemble, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
