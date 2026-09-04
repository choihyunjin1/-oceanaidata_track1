"""Exactly-once causal pairwise-ranking P1 falsification."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier as PointwiseSGD

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v23_causal_bipartite_pairwise_ranking_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V16 = ROOT / "scripts/run_p1_v16_causal_delay_embedding_persistence_addonly_20260901_v1.py"
_DATA_DIR: Path | None = None


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v23_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


shared = _module(V16)
core, base = shared.core, shared.base
_ORIGINAL_WRITE = shared._ORIGINAL_WRITE


def causal_context_features(
    frame: pd.DataFrame, train_boundary_ns: int, representation: dict[str, Any]
) -> np.ndarray:
    """Return a fixed context basis with categories learned from prefix only."""

    del representation
    shared._set_transport_context(frame, train_boundary_ns)
    times = pd.DatetimeIndex(frame["_time"])
    times_ns = core._time_ns(times)
    prefix = times_ns <= train_boundary_ns
    stations = sorted(frame.loc[prefix, "station"].astype(str).unique())
    layers = sorted(frame.loc[prefix, "layer"].astype(str).unique())
    station = frame["station"].astype(str).to_numpy()
    layer = frame["layer"].astype(str).to_numpy()
    station_basis = np.column_stack([station == value for value in stations])
    layer_basis = np.column_stack([layer == value for value in layers])
    minute = times.hour.to_numpy(np.float64) * 60.0 + times.minute.to_numpy(np.float64)
    day = times.dayofyear.to_numpy(np.float64)
    calendar = np.column_stack(
        [
            np.sin(2.0 * np.pi * minute / 1440.0),
            np.cos(2.0 * np.pi * minute / 1440.0),
            np.sin(2.0 * np.pi * day / 365.25),
            np.cos(2.0 * np.pi * day / 365.25),
        ]
    )
    output = np.column_stack([station_basis, layer_basis, calendar]).astype(np.float32)
    if not np.isfinite(output).all() or output.shape[1] < 6:
        raise RuntimeError("causal context basis is invalid")
    return output


class PairwiseLogisticRanker:
    """Sklearn-shaped linear scorer trained on symmetric class-difference pairs."""

    def __init__(
        self,
        *,
        loss: str,
        penalty: str,
        alpha: float,
        max_iter: int,
        tol: None,
        class_weight: dict[int, float],
        shuffle: bool,
        random_state: int,
    ) -> None:
        if loss != "log_loss" or penalty != "l2" or tol is not None or not shuffle:
            raise ValueError("pairwise scorer contract drifted")
        del class_weight
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> PairwiseLogisticRanker:
        labels = np.asarray(labels, dtype=np.int8)
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        if not len(positive) or not len(negative):
            raise RuntimeError("pairwise objective requires both classes")
        pairs = int(json.loads(CONFIG.read_text(encoding="utf-8"))["objective"]["pairs_per_fit"])
        rng = np.random.default_rng(self.random_state)
        pos = rng.choice(positive, size=pairs, replace=True)
        neg = rng.choice(negative, size=pairs, replace=True)
        difference = np.asarray(features[pos] - features[neg], dtype=np.float64)
        pair_features = np.concatenate([difference, -difference], axis=0)
        pair_labels = np.concatenate(
            [np.ones(pairs, dtype=np.int8), np.zeros(pairs, dtype=np.int8)]
        )
        order = rng.permutation(len(pair_labels))
        self._model = PointwiseSGD(
            loss="log_loss",
            penalty="l2",
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=None,
            class_weight=None,
            shuffle=True,
            random_state=self.random_state,
        ).fit(pair_features[order], pair_labels[order])
        self.coef_ = self._model.coef_.copy()
        self.intercept_ = np.zeros(1, dtype=np.float64)
        self.classes_ = np.array([0, 1], dtype=np.int8)
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float64) @ self.coef_[0]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        score = np.clip(self.decision_function(features), -40.0, 40.0)
        positive = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - positive, positive])


def _write_v23(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        if _DATA_DIR is None:
            raise RuntimeError("data directory unavailable")
        payload = dict(value)
        payload["schema_version"] = json.loads(CONFIG.read_text(encoding="utf-8"))[
            "result_schema_version"
        ]
        payload["long_event_boundary"] = shared.boundary_recall_from_artifacts(_DATA_DIR)
        payload["objective"] = {
            "kind": "symmetric_pairwise_logistic_bipartite_ranking",
            "pairs_per_fit": 60000,
            "outer_labels_in_pairs": 0,
        }
    _ORIGINAL_WRITE(path, payload)


def _configure() -> None:
    shared.CONFIG, shared.ARTIFACT, shared.LOCK = CONFIG, ARTIFACT, LOCK
    core.EXPERIMENT_ID, core.CONFIG, core.ARTIFACT, core.LOCK = (
        EXPERIMENT_ID,
        CONFIG,
        ARTIFACT,
        LOCK,
    )
    core.__file__ = str(Path(__file__).resolve())
    core.recurrence_laminar_features = causal_context_features
    core.SGDClassifier = PairwiseLogisticRanker
    core._write = _write_v23
    base._select = shared.shared._select_transport


def preflight(data_dir: Path) -> dict[str, Any]:
    _configure()
    return core.preflight(data_dir)


def qa(data_dir: Path) -> dict[str, Any]:
    _configure()
    result_path = ARTIFACT / "result.json"
    if not result_path.exists():
        return core.qa(data_dir)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {
        "terminal_result": result["experiment_id"] == EXPERIMENT_ID,
        "fits9": result["counters"]["fits"] == 9,
        "add_only": result["counters"]["anchor_removals"] == 0,
        "outer_isolation": result["counters"]["outer_target_reads_before_all_seals"] == 0,
        "access0": result["counters"]["official"] == result["counters"]["csv"] == result["counters"]["uploads"] == 0,
        "config_hash": result["hashes"]["config"] == core._sha(CONFIG),
        "runner_hash": result["hashes"]["runner"] == core._sha(Path(__file__)),
        "lock_hash": result["hashes"]["lock"] == core._sha(LOCK),
        "completion_hash": result["hashes"]["completion"] == core._sha(ARTIFACT / "predictions_complete.json"),
        "seals": all(
            core._sha(ARTIFACT / f"{fold}_sealed.npz")
            == json.loads((ARTIFACT / f"{fold}_seal.json").read_text(encoding="utf-8"))["sha256"]
            for fold in config["parts"]
        ),
        "pairwise_objective": result["objective"]["outer_labels_in_pairs"] == 0,
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "phase": "POST_TERMINAL_IMMUTABLE_REVALIDATION",
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "result_sha256": core._sha(result_path),
    }


def execute(data_dir: Path) -> dict[str, Any]:
    global _DATA_DIR
    _DATA_DIR = data_dir.resolve(strict=True)
    shared._DATA_DIR = _DATA_DIR
    _configure()
    return core.execute(data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--qa", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = (
        preflight(args.data_dir)
        if args.preflight
        else qa(args.data_dir)
        if args.qa
        else execute(args.data_dir)
    )
    print(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False), end="")


if __name__ == "__main__":
    main()
