"""Create a new independent package from this local template; no repository imports."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def build(output):
    template = Path(__file__).resolve().parent
    output = output.resolve()
    if output.exists():
        raise FileExistsError("use a new package directory, never overwrite")
    output.mkdir(parents=True)
    for name in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
        (output / name).mkdir()
    allowed = [
        "build_package.py",
        "README.md",
        "config.json",
        "requirements.txt",
        "RUN_TRAINING.ps1",
        "RUN_INFERENCE.ps1",
        "RUN_REPLAY.ps1",
        "02_code/core.py",
        "02_code/run.py",
        "01_data/README.md",
        "06_docs/SOURCE_PROVENANCE.json",
        "06_docs/SUBMISSION.md",
    ]
    for relative in allowed:
        shutil.copy2(template / relative, output / relative)
    manifest = {
        "status": "PACKAGE_BUILT_MODELS_EMPTY",
        "packaged_source_data_files": 0,
        "copied_prior_models": 0,
        "copied_prior_answers": 0,
        "files": {p: hashlib.sha256((output / p).read_bytes()).hexdigest() for p in allowed},
    }
    (output / "06_docs/BUILD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "template_files": len(allowed)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    build(parser.parse_args().output)
