"""Reproduce the frozen P2 deep-stack CSV from saved local weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.deep_data import build_panel
from p2_restore.deep_training import predict_full_checkpoint
from p2_restore.submission import build_submission, validate_submission


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _test_rows(index: pd.DataFrame, times: pd.DatetimeIndex, panel: np.ndarray) -> np.ndarray:
    positions = times.get_indexer(pd.to_datetime(index["time"], utc=True))
    if (positions < 0).any():
        raise ValueError("test_index time is absent from the deep panel")
    return panel[positions, index["layer"].to_numpy(int) - 2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--result", type=Path, default=Path("artifacts/p2_deep_finalists_v1/result.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("submissions/p2/P2_DEEP_STACK_V1_REPRODUCED.csv")
    )
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    data = load_p2_data(resolve_data_dir(args.data_dir))
    panel = build_panel(data.observations)
    predictions: dict[str, np.ndarray] = {}
    for model_name, entries in result["full_models"].items():
        current = [
            _test_rows(
                data.test_index,
                panel.times,
                predict_full_checkpoint(Path(entry["checkpoint"]), panel),
            )
            for entry in entries
        ]
        predictions[model_name] = np.mean(current, axis=0)
    incumbent_path = Path("submissions/p2/P2_SCORE_ROUTER_ROUND400.csv")
    frame = pd.DataFrame(
        {
            "layer": data.test_index["layer"].to_numpy(int),
            "router_400": pd.read_csv(incumbent_path)["temp"].to_numpy(float),
            **predictions,
        }
    )
    columns = list(result["weights_by_layer"]["2"])
    final = np.empty(len(frame), dtype=np.float64)
    for layer in (2, 3, 4):
        selected = frame["layer"].to_numpy(int) == layer
        weights = np.array([result["weights_by_layer"][str(layer)][name] for name in columns])
        final[selected] = frame.loc[selected, columns].to_numpy(float) @ weights
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_submission(data.test_index, final).to_csv(
        args.output, index=False, encoding="utf-8", lineterminator="\n"
    )
    validation = validate_submission(args.output, data.test_index)
    expected = result["submission"]["sha256"]
    actual = _sha256(args.output)
    if actual != expected:
        raise RuntimeError(f"reproduced submission hash differs: {actual} != {expected}")
    print(json.dumps({"sha256": actual, **validation}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
