from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import catboost
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from p3_wave.catboost_valid_hpo_v2 import (  # noqa: E402
    EXPERIMENT_ID,
    HPOContractError,
    control_candidate,
    materialize_grid,
    sha256_file,
    validate_schedule,
)

DEFAULT_CONFIG = ROOT / "configs/experiments/p3_catboost_valid_hpo_20260829_v2.json"
BASE_CONFIG = ROOT / "configs/experiments/p3_catboost_ordered_hpo_20260829_v1.json"
ENGINE_PATH = ROOT / "scripts/run_p3_catboost_ordered_hpo_20260829_v1.py"
TEST_PATH = ROOT / "tests/test_p3_catboost_valid_hpo_20260829_v2.py"
CONTRACT_PATH = ROOT / "src/p3_wave/catboost_valid_hpo_v2.py"

SPEC = importlib.util.spec_from_file_location("p3_catboost_hpo_v1_engine", ENGINE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load frozen P3 HPO engine")
ENGINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENGINE)
_ORIGINAL_STATIC_PREFLIGHT = ENGINE.static_preflight


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: Path) -> dict[str, Any]:
    if path.resolve() != DEFAULT_CONFIG.resolve():
        raise HPOContractError("only the canonical v2 config is allowed")
    override = json.loads(path.read_text(encoding="utf-8"))
    if override.get("extends") != str(BASE_CONFIG.relative_to(ROOT)).replace("\\", "/"):
        raise HPOContractError("v2 base config path changed")
    if sha256_file(BASE_CONFIG) != override.get("extends_sha256"):
        raise HPOContractError("v2 base config hash changed")
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config = _deep_merge(base, override)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise HPOContractError("v2 experiment id changed")
    return config


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _synthetic_grid_smoke(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    smoke = config["synthetic_smoke"]
    if catboost.__version__ != smoke["catboost_version"]:
        raise HPOContractError(
            f"CatBoost version {catboost.__version__} differs from {smoke['catboost_version']}"
        )
    grid_spec = json.loads(paths["grid"].read_text(encoding="utf-8"))
    challengers = materialize_grid(grid_spec)
    candidates = [control_candidate(grid_spec), *challengers]
    if len(candidates) != int(smoke["candidate_plus_control_fit_count"]):
        raise HPOContractError("synthetic smoke candidate count changed")

    rng = np.random.default_rng(int(smoke["seed"]))
    rows = int(smoke["rows"])
    features = int(smoke["features"])
    x = rng.normal(size=(rows, features)).astype(np.float32)
    y = (0.35 * x[:, 0] - 0.2 * x[:, 1] + rng.normal(scale=0.05, size=rows)).astype(
        np.float32
    )
    receipts: list[dict[str, Any]] = []
    for candidate in candidates:
        parameters = dict(candidate["parameters"])
        model = ENGINE.CatBoostRegressor(iterations=int(smoke["iterations"]), **parameters)
        try:
            model.fit(x, y)
            prediction = np.asarray(model.predict(x), dtype=np.float64)
        except Exception as exc:
            raise HPOContractError(
                f"synthetic one-tree smoke failed for {candidate['candidate_id']}: {exc}"
            ) from exc
        if prediction.shape != (rows,) or not np.isfinite(prediction).all():
            raise HPOContractError(
                f"synthetic prediction invalid for {candidate['candidate_id']}"
            )
        actual = model.get_all_params()
        for key in ("boosting_type", "grow_policy", "depth"):
            if str(actual[key]) != str(parameters[key]):
                raise HPOContractError(
                    f"synthetic actual parameter mismatch for {candidate['candidate_id']}: {key}"
                )
        receipts.append(
            {
                "candidate_id": candidate["candidate_id"],
                "boosting_type": actual["boosting_type"],
                "grow_policy": actual["grow_policy"],
                "depth": int(actual["depth"]),
                "actual_params_sha256": _json_sha256(actual),
                "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
            }
        )
    return {
        "challenger_count": len(challengers),
        "control_count": 1,
        "catboost_version": catboost.__version__,
        "synthetic_shape": [rows, features],
        "synthetic_smoke_fit_count": len(receipts),
        "historical_fit_count": 0,
        "smoke_receipts_sha256": _json_sha256(receipts),
        "receipts": receipts,
        "maximum_authorized_future_fit_count": validate_schedule(config),
    }


ENGINE.EXPERIMENT_ID = EXPERIMENT_ID
ENGINE.DEFAULT_CONFIG = DEFAULT_CONFIG
ENGINE.load_config = load_config
ENGINE.materialize_grid = materialize_grid
ENGINE.control_candidate = control_candidate
ENGINE.validate_schedule = validate_schedule
ENGINE._grid_checks = _synthetic_grid_smoke


def static_preflight(config_path: Path, data_dir: Path | None = None) -> dict[str, Any]:
    receipt = _ORIGINAL_STATIC_PREFLIGHT(config_path, data_dir)
    receipt["experiment_id"] = EXPERIMENT_ID
    receipt["implementation_hashes"] = {
        "runner": sha256_file(Path(__file__)),
        "contract_module": sha256_file(CONTRACT_PATH),
        "tests": sha256_file(TEST_PATH),
        "frozen_v1_engine": sha256_file(ENGINE_PATH),
    }
    receipt["execution_boundary"]["synthetic_smoke_fit_count"] = receipt["grid"][
        "synthetic_smoke_fit_count"
    ]
    receipt["execution_boundary"]["historical_fit_count"] = 0
    return receipt


ENGINE.static_preflight = static_preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("static-preflight", "execute"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--authorization-token")
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.mode == "execute":
        if args.data_dir is None:
            parser.error("--data-dir is required for execute")
        result = ENGINE.execute_hpo(config_path, args.data_dir, args.authorization_token)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    receipt = static_preflight(config_path, args.data_dir)
    if args.write_receipt:
        config = load_config(config_path)
        receipt_path = ENGINE._contained(config["outputs"]["static_preflight_receipt"])
        ENGINE._write_receipt(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
