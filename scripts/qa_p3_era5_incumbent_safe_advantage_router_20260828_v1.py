"""Independent read-only/core QA for the P3 ERA5 safe advantage router."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p3_wave.era5_safe_advantage_router import (  # noqa: E402
    EXPERIMENT_ID,
    ROUTER_FEATURES,
    attach_truth,
    evaluate_gate,
    file_pin,
    sha256_file,
    validate_router_feature_names,
    write_json_exclusive,
)

CONFIG_REL = Path(
    "configs/experiments/p3_era5_incumbent_safe_advantage_router_20260828_v1.json"
)
KEYS = ("fold", "anchor_id", "station", "lead_h")


class IndependentQAError(RuntimeError):
    """The isolated output differs from its frozen or no-op contract."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IndependentQAError(f"JSON root is not an object: {path}")
    return value


def _sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)


def _paths(root: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    config = _read_json(root / CONFIG_REL)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise IndependentQAError("config experiment ID changed")
    output = root / config["access_and_output"]["artifact_dir"]
    experts = root / config["frozen_inputs"]["sealed_expert_predictions"]["path"]
    incumbent = root / config["frozen_inputs"]["incumbent_oof"]["path"]
    return config, output, experts, incumbent


def verify_core(root: Path = ROOT) -> dict[str, Any]:
    config, output, experts_path, incumbent_path = _paths(root.resolve())
    required = {
        "README.md",
        "metrics.json",
        "result.json",
        "sealed_outer_predictions.parquet",
        "commitments/fold_00_2024_h2_storm.json",
        "commitments/fold_01_winter_transition.json",
        "commitments/fold_02_2025_h1.json",
        "commitments/predictions_complete.json",
    }
    observed = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "independent_qa.json"
    }
    missing = required - observed
    if missing:
        raise IndependentQAError(f"core output files are missing: {sorted(missing)}")

    metrics = _read_json(output / "metrics.json")
    result = _read_json(output / "result.json")
    if (
        metrics.get("experiment_id") != EXPERIMENT_ID
        or result.get("experiment_id") != EXPERIMENT_ID
        or result.get("official_access") is not False
        or result.get("candidate_or_submission_created") is not False
        or result.get("upload_count") != 0
    ):
        raise IndependentQAError("result research-only or official-access semantics changed")
    if metrics.get("fits", {}).get("catboost") != 0 or metrics.get("fits", {}).get(
        "ridge"
    ) not in {0, 1}:
        raise IndependentQAError("fit-count contract changed")

    sealed = _sort(pd.read_parquet(output / "sealed_outer_predictions.parquet"))
    experts = _sort(pd.read_parquet(experts_path))
    if len(sealed) != 1086 or len(experts) != 1086:
        raise IndependentQAError("sealed outer row count changed")
    if not sealed.loc[:, KEYS].equals(experts.loc[:, KEYS]):
        raise IndependentQAError("sealed outer keys differ from frozen experts")
    incumbent = experts["incumbent_prediction"].to_numpy(dtype=np.float64)
    transfer = experts["transfer_prediction"].to_numpy(dtype=np.float64)
    candidate = sealed["candidate_prediction"].to_numpy(dtype=np.float64)
    active = sealed["router_active"].to_numpy(dtype=bool)
    if candidate[~active].tobytes() != incumbent[~active].tobytes():
        raise IndependentQAError("inactive prediction bytes differ from incumbent")
    expected_active = incumbent[active] + 0.20 * (transfer[active] - incumbent[active])
    if not np.array_equal(candidate[active], expected_active):
        raise IndependentQAError("active prediction differs from fixed 0.20 blend")
    if sealed.loc[sealed["fold"].isin(["2024_h2_storm", "winter_transition"]), "router_active"].any():
        raise IndependentQAError("unsupported first two outer folds are not exact no-op")

    truth = pd.read_parquet(
        incumbent_path,
        columns=["prefix_fraction", *KEYS, "target_hs"],
    )
    truth = _sort(truth.loc[truth["prefix_fraction"].eq(1.0)].drop(columns="prefix_fraction"))
    evaluated = attach_truth(sealed, truth)
    recomputed = evaluate_gate(evaluated, require_fold_consistency=True)
    recorded = metrics.get("outer_gate")
    if json.dumps(recomputed, sort_keys=True) != json.dumps(recorded, sort_keys=True):
        raise IndependentQAError("independently recomputed outer gate differs")

    model_path = output / "router_model.json"
    if metrics.get("fits", {}).get("ridge") == 1:
        if not model_path.is_file():
            raise IndependentQAError("router model receipt is missing")
        model = _read_json(model_path)
        names = tuple(model.get("model", {}).get("feature_names", ()))
        validate_router_feature_names(names)
        if names != ROUTER_FEATURES or model.get("blend_strength") != 0.20:
            raise IndependentQAError("router model contract changed")

    commitments = []
    for index, fold in enumerate(config["outer_contract"]["fold_order"]):
        path = output / "commitments" / f"fold_{index:02d}_{fold}.json"
        commitment = _read_json(path)
        if (
            commitment.get("current_fold_truth_decodes_before_commitment") != 0
            or commitment.get("inactive_rows_bit_exact_incumbent") is not True
        ):
            raise IndependentQAError("fold blind-commitment ordering changed")
        blind = output / commitment["blind_prediction"]["path"]
        values = np.load(blind, allow_pickle=False)
        fold_values = sealed.loc[sealed["fold"].eq(fold), "candidate_prediction"].to_numpy(
            dtype=np.float64
        )
        if not np.array_equal(values, fold_values) or sha256_file(blind) != commitment[
            "blind_prediction"
        ]["sha256"]:
            raise IndependentQAError("fold blind prediction differs from commitment")
        commitments.append(file_pin(path, root=output))
    complete = _read_json(output / "commitments" / "predictions_complete.json")
    if complete.get("fold_commitments") != commitments:
        raise IndependentQAError("predictions-complete commitment list changed")

    return {
        "schema_version": "p3_era5_incumbent_safe_advantage_router.independent_qa.v1",
        "experiment_id": EXPERIMENT_ID,
        "mode": "core",
        "verdict": "PASS",
        "sealed_rows": int(len(sealed)),
        "active_rows": int(active.sum()),
        "inactive_rows_bit_exact_incumbent": True,
        "active_rows_exact_fixed_blend": True,
        "outer_gate_recomputed": True,
        "fold_commitments_verified": len(commitments),
        "fits": metrics["fits"],
        "official_access": False,
    }


def verify_final(root: Path = ROOT) -> dict[str, Any]:
    core = verify_core(root)
    _config, output, _experts, _incumbent = _paths(root.resolve())
    receipt = _read_json(output / "independent_qa.json")
    if receipt.get("verdict") != "PASS":
        raise IndependentQAError("written independent core QA is not PASS")
    manifest = _read_json(output / "manifest.json")
    seal = _read_json(output / "seal.json")
    for relative, pin in manifest.get("core_files", {}).items():
        path = output / relative
        if not path.is_file() or file_pin(path, root=output) != pin:
            raise IndependentQAError(f"manifest core pin changed: {relative}")
    if seal.get("manifest") != file_pin(output / "manifest.json", root=output):
        raise IndependentQAError("seal no longer binds manifest")
    if seal.get("blind_predictions") != file_pin(
        output / "sealed_outer_predictions.parquet", root=output
    ):
        raise IndependentQAError("seal no longer binds blind predictions")
    files = sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file())
    return {
        **core,
        "mode": "final",
        "manifest_verified": True,
        "seal_verified": True,
        "file_count": len(files),
        "output_tree_sha256": sha256_file(output / "seal.json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--core", action="store_true")
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--write-receipt", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify_final(args.root) if args.final else verify_core(args.root)
    if args.write_receipt:
        if not args.core:
            raise IndependentQAError("only core QA may publish the bound receipt")
        config, output, _experts, _incumbent = _paths(args.root.resolve())
        del config
        write_json_exclusive(output / "independent_qa.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
