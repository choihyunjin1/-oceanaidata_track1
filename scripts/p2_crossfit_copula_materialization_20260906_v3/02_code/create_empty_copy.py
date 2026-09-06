"""Copy code/config only to a new directory; never copy models, answers or logs."""

import argparse
import shutil
from pathlib import Path


def create(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise FileExistsError("new disjoint output directory required")
    for directory in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
        (destination / directory).mkdir(parents=True)
    for path in (source / "02_code").glob("*.py"):
        shutil.copyfile(path, destination / "02_code" / path.name)
    for name in ("config.json", "README.md", "requirements.txt"):
        shutil.copyfile(source / name, destination / name)
    if (source / "06_docs/BUILD_MANIFEST.json").is_file():
        shutil.copyfile(source / "06_docs/BUILD_MANIFEST.json", destination / "06_docs/BUILD_MANIFEST.json")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(create(Path(__file__).resolve().parents[1], args.destination))
