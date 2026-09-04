"""Shared exact surface construction for P1 MS-TCN scratch training/inference."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEYS = ["station", "year", "layer", "time"]


def load_source(package_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    package = Path(package_dir).resolve()
    source_root = package / "07_source"
    for item in (source_root / "src", source_root / "scripts"):
        if str(item) not in sys.path:
            sys.path.insert(0, str(item))
    path = source_root / "scripts" / "run_p1_incumbent_preserving_mstcn_asrf_v2.py"
    spec = importlib.util.spec_from_file_location("p1_final_training_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load P1 training source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = json.loads(
        (
            source_root
            / "configs"
            / "experiments"
            / "p1_incumbent_preserving_mstcn_asrf_v2.json"
        ).read_text(encoding="utf-8")
    )
    return module, config


def keys_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if len(left) != len(right):
        return False
    return pd.MultiIndex.from_frame(left[KEYS].astype(str)).equals(
        pd.MultiIndex.from_frame(right[KEYS].astype(str))
    )


def load_surfaces(
    package_dir: str | Path, data_dir: str | Path
) -> tuple[Any, dict[str, Any], Any, Any, Any, pd.DataFrame]:
    package = Path(package_dir).resolve()
    data = Path(data_dir).resolve()
    source, source_config = load_source(package)
    derived = package / "01_data" / "derived"
    train_meta = json.loads((derived / "train_features.json").read_text(encoding="utf-8"))
    test_meta = json.loads((derived / "test_features.json").read_text(encoding="utf-8"))
    if train_meta["feature_columns"] != test_meta["feature_columns"]:
        raise RuntimeError("P1 train/test feature schemas differ")
    numeric_names, projected, _dependency = source._feature_dependency_audit(
        train_meta, source_config
    )
    train_features = pd.read_parquet(derived / "train_features.parquet", columns=projected)
    test_features = pd.read_parquet(derived / "test_features.parquet", columns=projected)
    train_raw = pd.read_csv(data / "train.csv")
    test_keys = pd.read_csv(data / "test.csv", usecols=KEYS)
    sample = pd.read_csv(data / "sample_submission.csv")
    if train_raw[KEYS].duplicated().any() or test_keys[KEYS].duplicated().any():
        raise RuntimeError("P1 official keys are not unique")
    if not keys_equal(test_keys, sample):
        raise RuntimeError("P1 official ordered-key contract failed")
    if len(train_raw) != len(train_features) or len(test_keys) != len(test_features):
        raise RuntimeError("P1 derived feature row-count drift")
    if not np.array_equal(
        train_features["station"].astype(str).to_numpy(),
        train_raw["station"].astype(str).to_numpy(),
    ) or not np.array_equal(
        test_features["station"].astype(str).to_numpy(),
        test_keys["station"].astype(str).to_numpy(),
    ):
        raise RuntimeError("P1 raw/derived row alignment failed")
    labels = pd.to_numeric(train_raw["label"], errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise RuntimeError("P1 training labels are not binary")

    def make_surface(features: pd.DataFrame, keys: pd.DataFrame, **targets: Any) -> Any:
        return source.RowSurface(
            keys=keys[KEYS].reset_index(drop=True),
            numeric=features.loc[:, numeric_names].to_numpy(dtype=np.float32),
            station=features["station"].astype(str).to_numpy(),
            layer_category=features["layer_category"].astype(str).to_numpy(),
            depth_regime=None,
            depth=features["depth_raw"].to_numpy(dtype=np.float32),
            **targets,
        )

    train = make_surface(
        train_features,
        train_raw,
        labels=labels,
        anomaly_type=train_raw["anomaly_type"].fillna("").astype(str).to_numpy(),
    )
    test = make_surface(
        test_features,
        test_keys,
        anchor=np.zeros(len(test_keys), dtype=np.int8),
    )
    encoder, encoded = source._fit_encoder_and_transform(
        train,
        [test],
        fit_ids=np.arange(train.rows, dtype=np.int64),
        forbidden_ids=np.asarray([], dtype=np.int64),
        numeric_names=tuple(numeric_names),
    )
    training, holdout = encoded
    if training.features.shape[1] != 165 or holdout.features.shape[1] != 165:
        raise RuntimeError("P1 encoded feature width drift")
    return source, source_config, encoder, training, holdout, sample
