"""Independently verify the frozen P2 checkpoint-0.85 deployment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from p2_restore.data import KEYS, load_p2_data, resolve_data_dir
from p2_restore.submission import validate_submission


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, test_index: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"station": "string", "time": "string"})
    validate_submission(frame, test_index)
    if not frame[KEYS].equals(test_index[KEYS]):
        raise AssertionError(f"key/order mismatch: {path}")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()

    bundle = args.bundle.resolve(strict=True)
    manifest_path = (bundle / "manifest.json").resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = load_p2_data(resolve_data_dir(args.data_dir))
    test_index = data.test_index
    layer4 = test_index["layer"].to_numpy(int) == 4

    incumbent_path = Path(manifest["incumbent"]["path"]).resolve(strict=True)
    axis_path = Path(manifest["axis_anchor"]["path"]).resolve(strict=True)
    incumbent = _load(incumbent_path, test_index)
    axis = _load(axis_path, test_index)
    checks: dict[str, object] = {
        "manifest_status": manifest["status"]
        == "QUARANTINED_PROTOCOL_VIOLATION_NOT_SUBMISSION_ELIGIBLE",
        "official_upload_performed_false": manifest["official_upload_performed"] is False,
        "hidden_target_values_read_false": manifest["hidden_target_values_read"] is False,
        "incumbent_hash_match": _sha256(incumbent_path) == manifest["incumbent"]["sha256"],
        "axis_anchor_hash_match": _sha256(axis_path) == manifest["axis_anchor"]["sha256"],
        "row_count": int(len(test_index)),
        "layer4_rows": int(layer4.sum()),
    }

    checkpoint_checks = []
    for pin in manifest["checkpoint_pins"]:
        path = Path(pin["path"]).resolve(strict=True)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint_checks.append(
            {
                "seed": int(pin["seed"]),
                "hash_match": _sha256(path) == pin["sha256"],
                "bytes_match": path.stat().st_size == int(pin["bytes"]),
                "payload_seed_match": int(payload["seed"]) == int(pin["seed"]),
                "selected_epoch_97": int(payload["selected_epoch"]) == 97,
            }
        )
    checks["checkpoint_checks"] = checkpoint_checks

    candidate_by_name = {}
    candidate_checks = []
    for item in manifest["candidates"]:
        path = Path(item["path"]).resolve(strict=True)
        frame = _load(path, test_index)
        values = frame["temp"].to_numpy(float)
        candidate_by_name[item["name"]] = values
        candidate_checks.append(
            {
                "name": item["name"],
                "hash_match": _sha256(path) == item["sha256"],
                "bytes_match": path.stat().st_size == int(item["bytes"]),
                "rows_26061": len(frame) == 26061,
                "finite": bool(np.isfinite(values).all()),
                "physical_range": bool(((values >= -5.0) & (values <= 45.0)).all()),
            }
        )
    checks["candidate_checks"] = candidate_checks

    incumbent_values = incumbent["temp"].to_numpy(float)
    axis_values = axis["temp"].to_numpy(float)
    blend = candidate_by_name["P2_1_CHECKPOINT85_L4_BLEND50"]
    full = candidate_by_name["P2_2_CHECKPOINT85_L4_FULL"]
    combined = candidate_by_name["P2_3_AXIS_U_PLUS_CHECKPOINT85_L4"]
    checks["layer23_preservation"] = {
        "blend_vs_incumbent": bool(np.array_equal(blend[~layer4], incumbent_values[~layer4])),
        "full_vs_incumbent": bool(np.array_equal(full[~layer4], incumbent_values[~layer4])),
        "combined_vs_axis": bool(np.array_equal(combined[~layer4], axis_values[~layer4])),
    }
    checks["layer4_relations"] = {
        "blend_is_exact_midpoint": bool(
            np.allclose(
                blend[layer4],
                0.5 * (incumbent_values[layer4] + full[layer4]),
                rtol=0.0,
                atol=2e-12,
            )
        ),
        "combined_matches_full": bool(
            np.array_equal(combined[layer4], full[layer4])
        ),
    }

    def _all_true(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            return all(_all_true(item) for item in value.values())
        if isinstance(value, list):
            return all(_all_true(item) for item in value)
        return True

    result = {
        "schema_version": "p2.checkpoint85_layer4_deployment.independent_qa.v1",
        "status": "TECHNICAL_INTEGRITY_PASS_QUARANTINED"
        if _all_true(checks)
        else "TECHNICAL_INTEGRITY_FAIL_QUARANTINED",
        "submission_eligibility": "FAIL_SOURCE_PROTOCOL_BOUNDARY",
        "checks": checks,
        "manifest_sha256": _sha256(manifest_path),
        "official_upload_performed": False,
        "hidden_target_values_read": False,
    }
    output = bundle / "INDEPENDENT_QA.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "TECHNICAL_INTEGRITY_PASS_QUARANTINED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
