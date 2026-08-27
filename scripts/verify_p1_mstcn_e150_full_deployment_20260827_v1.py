"""Independent read-only QA for the P1 MS-TCN e150 deployment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT_ID = "p1_mstcn_e150_full_deployment_20260827_v1"
KEY_COLUMNS = ("station", "year", "layer", "time")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def keys(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame.loc[:, KEY_COLUMNS].astype(str))


def verify(
    *, artifact_dir: Path, delivery_set: Path, test_path: Path, current_router_path: Path
) -> dict[str, Any]:
    terminal = json.loads((artifact_dir / "terminal_result.json").read_text(encoding="utf-8"))
    qa = json.loads((artifact_dir / "independent_qa.json").read_text(encoding="utf-8"))
    manifest = json.loads((delivery_set / "SET_MANIFEST.json").read_text(encoding="utf-8"))
    if terminal["experiment_id"] != EXPERIMENT_ID or terminal["status"] != "BUILD_AND_QA_PASS_NOT_UPLOADED":
        raise AssertionError("terminal state changed")
    if terminal["upload_performed"] or manifest["upload_performed"]:
        raise AssertionError("bundle unexpectedly attests an upload")
    test = pd.read_csv(test_path, usecols=list(KEY_COLUMNS))
    router = pd.read_csv(current_router_path, usecols=[*KEY_COLUMNS, "label"])
    if len(test) != 169011 or not keys(test).equals(keys(router)):
        raise AssertionError("test/current-Router key identity changed")
    anchor = router["label"].to_numpy(dtype=np.int8)

    candidates: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for entry in manifest["submission_order"]:
        path = delivery_set / entry["path"]
        if sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise AssertionError(f"manifest identity changed: {entry['directory']}")
        frame = pd.read_csv(path)
        if len(frame) != 169011 or not keys(test).equals(keys(frame)):
            raise AssertionError(f"candidate keys changed: {entry['directory']}")
        labels = frame["label"].to_numpy()
        if not np.isin(labels, [0, 1]).all():
            raise AssertionError(f"candidate is not binary: {entry['directory']}")
        frames.append(frame)
        candidates.append(
            {
                "directory": entry["directory"],
                "rows": len(frame),
                "positive_rows": int(labels.sum()),
                "bytes": int(path.stat().st_size),
                "sha256": sha256(path),
            }
        )
    all_bits = frames[0]["label"].to_numpy(dtype=np.int8)
    gs_bits = frames[1]["label"].to_numpy(dtype=np.int8)
    station = test["station"].astype(str).to_numpy()
    if int(np.sum((anchor == 1) & (all_bits == 0))) != 0:
        raise AssertionError("all-station candidate removed Router positives")
    if int(np.sum((anchor == 1) & (gs_bits == 0))) != 0:
        raise AssertionError("GS candidate removed Router positives")
    if np.any(gs_bits[station == "I-ORS"] != anchor[station == "I-ORS"]):
        raise AssertionError("GS candidate changed I-ORS")
    if np.any(all_bits[station != "I-ORS"] != gs_bits[station != "I-ORS"]):
        raise AssertionError("all and GS candidates differ outside I-ORS")
    if int(np.sum((all_bits == 1) & (gs_bits == 0))) != 80:
        raise AssertionError("expected 80 isolated I-ORS additions")

    seed_checks: list[dict[str, Any]] = []
    for seed in (20260827, 20260839, 20260863):
        stem = f"full_width_512_seed_{seed}_epoch_150"
        receipt = json.loads((artifact_dir / f"{stem}_receipt.json").read_text(encoding="utf-8"))
        history_path = artifact_dir / f"{stem}_history.json"
        prediction_path = artifact_dir / f"{stem}_test_prediction.npz"
        checkpoint_path = artifact_dir / f"{stem}_state.pt"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if [row["epoch"] for row in history] != list(range(1, 151)):
            raise AssertionError(f"history epoch sequence changed: {seed}")
        if sum(int(row["nonfinite_count"]) for row in history) != 0:
            raise AssertionError(f"nonfinite training event found: {seed}")
        for name, path in (
            ("history", history_path),
            ("prediction", prediction_path),
            ("checkpoint", checkpoint_path),
        ):
            if sha256(path) != receipt[f"{name}_artifact"]["sha256"]:
                raise AssertionError(f"seed artifact hash changed: {seed}.{name}")
        with np.load(prediction_path, allow_pickle=False) as archive:
            shapes = {name: list(archive[name].shape) for name in archive.files}
            finite = all(np.isfinite(archive[name]).all() for name in archive.files)
        if shapes != {
            "row_probability": [169011],
            "boundary_probability": [169011, 2],
            "type_probability": [169011, 5],
        } or not finite:
            raise AssertionError(f"seed predictions changed: {seed}")
        seed_checks.append(
            {
                "seed": seed,
                "epochs": 150,
                "nonfinite_count": 0,
                "prediction_shapes": shapes,
                "checkpoint_sha256": sha256(checkpoint_path),
            }
        )

    return {
        "schema_version": "p1.mstcn_e150_full_deployment.independent_postexecution_qa.v1",
        "experiment_id": EXPERIMENT_ID,
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "decision": "PASS",
        "test_rows": len(test),
        "test_target_columns_read": 0,
        "candidates": candidates,
        "all_anchor_positive_removed_rows": int(np.sum((anchor == 1) & (all_bits == 0))),
        "gs_anchor_positive_removed_rows": int(np.sum((anchor == 1) & (gs_bits == 0))),
        "isolated_iors_additions": int(np.sum((all_bits == 1) & (gs_bits == 0))),
        "seed_checks": seed_checks,
        "upload_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--delivery-set", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--current-router", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(
        artifact_dir=args.artifact_dir,
        delivery_set=args.delivery_set,
        test_path=args.test,
        current_router_path=args.current_router,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
