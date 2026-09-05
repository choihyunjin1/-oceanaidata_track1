"""Explicit focused checks with opt-in, exact-input PASS reuse. No training launcher."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def paths(root: Path, values: list[str], *, tests: bool = False) -> list[str]:
    result = []
    for value in values:
        path = (root / value).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Expected a repository file: {value}")
        relative = path.relative_to(root.resolve())
        if relative.parts[0] not in {"src", "scripts", "tests"} or path.suffix != ".py":
            raise ValueError("Only explicit source/test Python files accepted")
        if tests and (relative.parts[0] != "tests" or not path.name.startswith("test_")):
            raise ValueError("Select test_*.py files, not directories or pytest options")
        result.append(relative.as_posix())
    if not result:
        raise ValueError("Both tests and lint files must be selected")
    return sorted(set(result))


def environment() -> dict[str, str]:
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTEST_ADDOPTS="",
    )


def fingerprint(root: Path, selected: list[str], commands: list[list[str]]) -> str:
    files = {root / value for value in selected}
    for directory in ("src", "scripts", "configs", "tests"):
        files.update(
            p
            for p in (root / directory).rglob("*")
            if p.is_file() and p.suffix in {".py", ".pyi", ".toml", ".json", ".yaml", ".yml"}
        )
    files.update(
        root / p for p in ("pytest.ini", "setup.cfg", "tox.ini", ".ruff.toml", "conftest.py")
    )
    files.update(root.glob("requirements*.txt"))
    files.update(root.glob("*.toml"))
    files.update(root.glob("*.md"))
    files.update(root / p for p in ("AGENTS.md", "00_ORGANIZER_DATA_POLICY.md"))
    digest = hashlib.sha256()
    for path in sorted(files):
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Dependency symlink escapes repository")
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            with path.open("rb") as handle:
                digest.update(hashlib.file_digest(handle, "sha256").digest())
    env = environment()
    runtime = {
        "python": [sys.executable, sys.version],
        "commands": commands,
        "packages": sorted(
            (d.metadata.get("Name", ""), d.version) for d in importlib.metadata.distributions()
        ),
        "environment": {
            k: env.get(k)
            for k in (
                "PYTHONPATH",
                "PYTEST_ADDOPTS",
                "PYTEST_PLUGINS",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }
    digest.update(json.dumps(runtime, sort_keys=True).encode())
    return digest.hexdigest()


def verify(root: Path, tests: list[str], lint: list[str], *, reuse: bool = False) -> dict:
    tests, lint = paths(root, tests, tests=True), paths(root, lint)
    commands = [
        [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", *tests],
        [sys.executable, "-m", "ruff", "check", *lint],
    ]
    key = fingerprint(root, tests + lint, commands)
    cache = root / "artifacts/agent_verification" / f"{key}.json"
    if reuse and cache.is_file():
        try:
            previous = json.loads(cache.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            previous = {}
        if previous.get("fingerprint") == key and previous.get("status") == "PASS":
            return {**previous, "reused": True}
    checks = []
    executed_tests = 0
    with tempfile.TemporaryDirectory(prefix="ocean-check-") as temporary_dir:
        xml_path = Path(temporary_dir) / "pytest.xml"
        for index, command in enumerate(commands):
            actual = command + [f"--junitxml={xml_path}"] if index == 0 else command
            result = subprocess.run(
                actual,
                cwd=root,
                env=environment(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if index == 0 and result.returncode == 0:
                try:
                    executed_tests = sum(
                        int(s.get("tests", "0")) - int(s.get("skipped", "0"))
                        for s in ET.parse(xml_path).iter("testsuite")
                    )
                except (ET.ParseError, ValueError, OSError):
                    executed_tests = 0
            checks.append(
                {
                    "command": command[1:],
                    "exit_code": result.returncode,
                    "summary": (result.stdout + result.stderr)[-4000:],
                }
            )
            if result.returncode or (index == 0 and executed_tests <= 0):
                break
    unchanged = key == fingerprint(root, tests + lint, commands)
    passed = executed_tests > 0 and len(checks) == 2 and all(c["exit_code"] == 0 for c in checks)
    status = "PASS" if passed and unchanged else "PASS_SOURCE_CHANGED" if passed else "FAIL"
    receipt = {
        "status": status,
        "fingerprint": key,
        "reused": False,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "checks": checks,
        "executed_tests": executed_tests,
        "source_unchanged": unchanged,
        "scope": "Code checks only; not model quality, candidate QA, lineage or official scoring.",
    }
    if status == "PASS":
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=cache.parent, suffix=".tmp", delete=False
        ) as handle:
            json.dump(receipt, handle, indent=2, ensure_ascii=False)
            temporary = Path(handle.name)
        temporary.replace(cache)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="append", required=True)
    parser.add_argument("--lint", action="append", required=True)
    parser.add_argument(
        "--execute", action="store_true", help="Default: validate selection, plan only"
    )
    parser.add_argument("--reuse-pass", action="store_true", help="Opt-in unchanged PASS reuse")
    args = parser.parse_args()
    tests, lint = paths(ROOT, args.test, tests=True), paths(ROOT, args.lint)
    if not args.execute:
        print(json.dumps({"status": "PLAN_ONLY", "tests": tests, "lint": lint}, indent=2))
        return 0
    result = verify(ROOT, tests, lint, reuse=args.reuse_pass)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
