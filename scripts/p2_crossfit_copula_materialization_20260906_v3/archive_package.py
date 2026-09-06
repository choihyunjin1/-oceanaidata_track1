"""Archive verified local outputs without data, locks, or calibration arrays."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def archive(package, destination, extraction):
    assert package.is_dir() and not destination.exists() and not extraction.exists()
    assert not extraction.is_relative_to(package) and not package.is_relative_to(extraction)
    for name in ("training-qa.json", "inference-qa.json", "replay-qa.json"):
        assert json.loads((package / "04_logs" / name).read_text())["status"] == "PASS"
    allowed = [package / name for name in ("README.md", "requirements.txt", "config.json")]
    allowed += sorted((package / "02_code").glob("*.py"))
    allowed += sorted((package / "03_model").glob("*.pt"))
    allowed += sorted((package / "03_model").glob("*.npz"))
    allowed += sorted((package / "04_logs").glob("*.json"))
    allowed += sorted((package / "05_answer").glob("*.csv"))
    allowed += [package / "06_docs/BUILD_MANIFEST.json"]
    if (package / "06_docs/VERIFIED_RUN.md").is_file():
        allowed += [package / "06_docs/VERIFIED_RUN.md"]
    assert not any("failure" in path.name.lower() for path in allowed)
    names = {path.relative_to(package).as_posix(): sha(path) for path in allowed}
    assert len(names) == len(allowed) and all(path.is_file() for path in allowed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as out:
        for dirname in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
            out.writestr(dirname + "/", "")
        for path in allowed:
            out.write(path, path.relative_to(package).as_posix())
    extraction.mkdir(parents=True)
    with ZipFile(destination) as source:
        assert source.testzip() is None
        assert len(source.namelist()) == len(set(source.namelist()))
        for member in source.infolist():
            relative = PurePosixPath(member.filename)
            assert not relative.is_absolute() and ".." not in relative.parts
            target = extraction.joinpath(*relative.parts).resolve()
            assert target.is_relative_to(extraction)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as reader, target.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
    assert all(sha(extraction / name) == digest for name, digest in names.items())
    result = {"status": "ARCHIVE_AND_EXTRACTION_HASH_PASS_REPLAY_PENDING", "archive_sha256": sha(destination),
              "archive_bytes": destination.stat().st_size, "file_count": len(names), "files_sha256": names,
              "source_data_files": 0, "attempt_locks": 0, "calibration_or_replay_arrays": 0,
              "archive": str(destination), "extraction": str(extraction),
              "next": "Run extracted 02_code/run.py REPLAY --replay-receipt zip-extraction-replay.json in a new process; not a second training run."}
    receipt = destination.with_suffix(".manifest.json")
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "files_sha256"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("extraction", type=Path)
    args = parser.parse_args()
    archive(args.package.resolve(), args.destination.resolve(), args.extraction.resolve())
