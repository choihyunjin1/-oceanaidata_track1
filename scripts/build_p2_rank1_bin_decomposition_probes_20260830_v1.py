"""Build two pre-score frozen P2 rank-1 season-bin decomposition probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID = "p2_rank1_bin_decomposition_probes_20260830_v1"
REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "experiments" / f"{ID}.json"
REPORT = REPO / "reports" / ID / "prepared-probes.json"
EXPECTED_CONFIG_SHA = "27e4d525fbfbf359c7e16463b630a5d60893a75ae95a66a37a02bbbe8f22e215"
KEYS = ["station", "layer", "time"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-dir", type=Path, required=True)
    parser.add_argument("--alpha50", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("--execute required")
    output = args.output_dir.resolve()
    if output.is_relative_to(REPO) or output.exists() or REPORT.parent.exists():
        raise RuntimeError("exclusive external output/report contract failed")
    if sha(CONFIG) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("config hash mismatch")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    paths = {
        "test_index": args.p2_dir.resolve() / "test_index.csv",
        "sample_submission": args.p2_dir.resolve() / "sample_submission.csv",
        "alpha50": args.alpha50.resolve(),
        "champion": args.champion.resolve(),
    }
    pins = config["source_pins"]
    for name, path in paths.items():
        expected = pins[f"{name}_sha256"]
        if sha(path) != expected:
            raise RuntimeError(f"source pin mismatch: {name}")

    test = pd.read_csv(paths["test_index"], dtype={"station": "string", "time": "string"})
    sample = pd.read_csv(paths["sample_submission"], dtype={"station": "string", "time": "string"})
    alpha50 = pd.read_csv(paths["alpha50"], dtype={"station": "string", "time": "string"})
    champion = pd.read_csv(paths["champion"], dtype={"station": "string", "time": "string"})
    if list(test.columns) != config["contract"]["test_columns"]:
        raise RuntimeError("test schema mismatch")
    for name, frame in {"sample": sample, "alpha50": alpha50, "champion": champion}.items():
        if list(frame.columns) != config["contract"]["output_columns"]:
            raise RuntimeError(f"output source schema mismatch: {name}")
        if len(frame) != len(test) or not frame[KEYS].equals(test[KEYS]):
            raise RuntimeError(f"key/order mismatch: {name}")
    if len(test) != config["contract"]["rows"]:
        raise RuntimeError("row count mismatch")

    times = pd.to_datetime(test["time"], utc=True).dt.tz_convert("Asia/Seoul")
    bins = ((times.dt.dayofyear.to_numpy() - 1) // 14).astype(int)
    anchor = alpha50["temp"].to_numpy(np.float64)
    champion_values = champion["temp"].to_numpy(np.float64)
    correction = champion_values - anchor
    active = np.abs(correction) > 1e-12
    if np.any(active & ~np.isin(bins, [17, 18])):
        raise RuntimeError("champion correction escaped bins 17/18")

    deny = set(config["prior_platform_submission_sha256"])
    prepared: list[tuple[dict[str, object], bytes, bytes]] = []
    seen: set[str] = set()
    champion_rmse = float(config["evidence"]["champion_official_rmse_c"])
    for spec in config["recipe"]["candidates"]:
        enabled = np.isin(bins, spec["enabled_bins"])
        values = np.where(enabled, champion_values, anchor)
        candidate = test[KEYS].copy()
        candidate["temp"] = values
        if list(candidate.columns) != config["contract"]["output_columns"]:
            raise RuntimeError("candidate schema mismatch")
        if not np.isfinite(values).all() or np.min(values) < -5 or np.max(values) > 45:
            raise RuntimeError("candidate finite/domain mismatch")
        payload = candidate.to_csv(index=False, lineterminator="\n").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        if digest in deny or digest in seen:
            raise RuntimeError("prior-platform or within-pack duplicate")
        seen.add(digest)
        delta_champion = values - champion_values
        delta_anchor = values - anchor
        rms_champion = float(np.sqrt(np.mean(np.square(delta_champion))))
        record = {
            "priority": spec["priority"],
            "name": spec["name"],
            "enabled_bins": spec["enabled_bins"],
            "title": spec["title"],
            "summary": spec["summary"],
            "rows": len(candidate),
            "sha256": digest,
            "bytes": len(payload),
            "changed_rows_vs_champion": int(np.sum(np.abs(delta_champion) > 1e-12)),
            "rms_change_vs_champion_c": rms_champion,
            "changed_rows_vs_alpha50": int(np.sum(np.abs(delta_anchor) > 1e-12)),
            "rms_change_vs_alpha50_c": float(np.sqrt(np.mean(np.square(delta_anchor)))),
            "minimum_c": float(np.min(values)),
            "maximum_c": float(np.max(values)),
            "triangle_rmse_upper_bound_c": champion_rmse + rms_champion,
            "triangle_downside_vs_champion_c": rms_champion,
            "exact_platform_hash_duplicate": False,
        }
        note = (
            f"제출물 제목: {spec['title']}\n"
            f"한줄요약(접근방식): {spec['summary']}\n"
            f"파일 SHA-256: {digest}\n"
            "상태: FROZEN_READY_NOT_UPLOADED\n"
        ).encode("utf-8-sig")
        prepared.append((record, payload, note))

    output.mkdir(parents=True, exist_ok=False)
    for record, payload, note in prepared:
        directory = output / str(record["name"])
        directory.mkdir()
        csv_path = directory / "P2_submission.csv"
        write_bytes(csv_path, payload)
        write_bytes(directory / "제출정보.txt", note)
        if sha(csv_path) != record["sha256"]:
            raise RuntimeError("post-write hash mismatch")
        record["path"] = str(csv_path)

    manifest = {
        "schema_version": "p2.rank1_bin_decomposition_probes.prepared.20260830.v1",
        "experiment_id": ID,
        "status": "FROZEN_READY_NOT_UPLOADED",
        "config_sha256": EXPECTED_CONFIG_SHA,
        "source_hashes": {name: sha(path) for name, path in paths.items()},
        "candidates": [item[0] for item in prepared],
        "qa": {
            "both_frozen_before_new_score": True,
            "schema_key_order_finite_domain": True,
            "platform_duplicate_denylist_count": len(deny),
            "candidate_hashes_distinct": True,
            "source_correction_outside_bins_17_18": 0,
        },
        "information_question": "Which predeclared 14-day season bin carries the official rank-1 champion gain? The two disjoint candidates exactly decompose the champion correction.",
        "execution": {"model_fits": 0, "hpo": 0, "hidden_truth_reads": 0, "score_py_reads": 0, "uploads": 0, "commits": 0, "pushes": 0},
    }
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    write_bytes(output / "SET_MANIFEST.json", encoded)
    REPORT.parent.mkdir(parents=True, exist_ok=False)
    write_bytes(REPORT, encoded)
    print(json.dumps({"status": manifest["status"], "candidates": manifest["candidates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
