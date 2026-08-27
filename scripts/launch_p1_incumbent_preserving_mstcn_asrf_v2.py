"""Hash-pinned launcher for the P1 MS-TCN++/ASRF v2 protocol.

The scientific runner refuses accidental direct execution.  This launcher
compares the final preregistration and local implementation modules to literal,
reviewable SHA-256 pins before importing the runner.  It is an integrity guard,
not a hostile same-process Python security boundary.  Scientific execution also
requires the caller to acknowledge this launcher's final reviewed SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_p1_incumbent_preserving_mstcn_asrf_v2.py"
EXPERIMENT_ID = "p1_incumbent_preserving_mstcn_asrf_v2"

# Filled only after runner/config/model/data QA is complete.  These literals
# provide the reviewable external integrity reference used by this launcher.
PINNED_SHA256 = {
    "runner": "78df1bd3b4777560b0134bc69ad45839c8c0093da39d2c1d3d458b3a3ba9b87a",
    "config": "1f8940d29ea6b047273e4f53445f62230e7d72bf1f0b14abe9fb18476f0345f0",
    "package_init": "5a7743ab77b2f9bd6851f12ebe694313c4fb361611821ed68dfd3b7574a82088",
    "model": "57c135bfe746d06a53e5c3b83517cb96f8262a765febefbf43e1d3a3c344fd7f",
    "data": "cf5dc2dbbb3ecf05c489b661f5427ff225caaf28af5fb44d292e3289c7bb9adf",
    "current_router_anchor_builder": "6235536f4cde74394579a1732225bb2de3375e28628cb962b3c410cc6335b08d",
    "capacity_calibration_builder": "f896fedabeccbd4ef5e37a393af0a27b6e102ba8fd525ca89a8c7b47d567bec8",
}

PINNED_PATHS = {
    "runner": Path("scripts") / "run_p1_incumbent_preserving_mstcn_asrf_v2.py",
    "config": Path("configs")
    / "experiments"
    / "p1_incumbent_preserving_mstcn_asrf_v2.json",
    "package_init": Path("src") / "p1_qc" / "__init__.py",
    "model": Path("src") / "p1_qc" / "ms_tcn_asrf.py",
    "data": Path("src") / "p1_qc" / "ms_tcn_asrf_data.py",
    "current_router_anchor_builder": Path("scripts")
    / "build_p1_current_router_oof_anchor_v1.py",
    "capacity_calibration_builder": Path("scripts")
    / "benchmark_p1_mstcn_capacity_grid_v2.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_literal_pins() -> dict[str, Any]:
    if set(PINNED_SHA256) != set(PINNED_PATHS):
        raise RuntimeError("launcher pin inventory changed")
    identities: dict[str, Any] = {}
    for name, relative in PINNED_PATHS.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _sha256(path)
        expected = PINNED_SHA256[name]
        if observed != expected:
            raise RuntimeError(f"literal execution pin mismatch: {name}")
        identities[name] = {
            "path": relative.as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": observed,
        }
    return identities


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "p1_incumbent_preserving_mstcn_asrf_v2_sealed", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sealed scientific runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute-protocol", action="store_true")
    parser.add_argument("--expected-launcher-sha256")
    args = parser.parse_args(argv)

    identities = verify_literal_pins()
    launcher_sha256 = _sha256(Path(__file__))
    if args.execute_protocol and args.expected_launcher_sha256 != launcher_sha256:
        raise RuntimeError(
            "--expected-launcher-sha256 must equal the final reviewed launcher bytes"
        )
    runner = _load_runner()
    preflight = runner.check_only(root=ROOT)
    if args.check_only:
        result = {
            **preflight,
            "external_literal_pins": identities,
            "external_launcher_sha256": launcher_sha256,
        }
    else:
        attestation = {
            "verified_by": Path(__file__).name,
            "launcher_sha256": launcher_sha256,
            "externally_expected_launcher_sha256": args.expected_launcher_sha256,
            "external_launcher_hash_acknowledged": True,
            "identities": identities,
            "all_pins_matched": True,
        }
        result = runner.execute_protocol(
            root=ROOT,
            artifact_dir=ROOT / "artifacts" / EXPERIMENT_ID,
            implementation_attestation=attestation,
            launcher_capability=runner._SEALED_LAUNCHER_CAPABILITY,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
