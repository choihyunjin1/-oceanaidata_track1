"""Train and persist the exact three-seed P2 v52 scratch ensemble."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p2_pipeline import (  # noqa: E402
    activate_source,
    build_arrays,
    domain_balanced_weights,
    train_seed,
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def train(
    data_dir: str | Path,
    package_dir: str | Path,
    output_dir: str | Path = "03_model/weights",
    *,
    replace: bool = False,
) -> dict:
    started = time.perf_counter()
    package = Path(package_dir).resolve()
    data = Path(data_dir).resolve()
    activate_source(package)
    from p2_restore.features import build_training_features
    from p2_restore.normalized_curvature_residual import build_normalized_curvature_design

    contract = json.loads((package / "contract.json").read_text(encoding="utf-8"))
    observations_path = data / "observations.csv"
    if _sha256(observations_path) != contract["official_inputs"]["observations.csv"]:
        raise RuntimeError("P2 observations.csv hash drift")
    observations = pd.read_csv(
        observations_path, dtype={"station": "string", "time": "string"}
    )
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    table = build_training_features(observations)
    design = build_normalized_curvature_design(table.frame)
    tokens, mask, context = build_arrays(table.frame)
    local = design.keys["time"].dt.tz_convert("Asia/Seoul")
    selected = local >= pd.Timestamp("2024-05-01T00:00:00+09:00")
    weights, weight_receipt = domain_balanced_weights(
        design.keys.loc[selected, "layer"].to_numpy(int), local[selected]
    )
    weights_dir = Path(output_dir)
    if not weights_dir.is_absolute():
        weights_dir = package / weights_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    for seed in (20260901, 20260902, 20260903):
        target = weights_dir / f"v52_seed_{seed}.pt"
        if target.exists() and not replace:
            raise FileExistsError(f"P2 checkpoint exists; use --replace deliberately: {target}")
        model, receipt = train_seed(
            tokens[selected],
            mask[selected],
            context[selected],
            design.normalized_target[selected],
            weights,
            seed=seed,
        )
        temporary = target.with_suffix(".pt.tmp")
        torch.save(
            {
                "schema_version": "p2.v52.final.checkpoint.20260905.v1",
                "seed": seed,
                "architecture": "masked_third_central_moment_deepset",
                "training_rows": int(selected.sum()),
                "state_dict": model.state_dict(),
                "receipt": receipt,
            },
            temporary,
        )
        os.replace(temporary, target)
        checkpoints.append(
            {
                "filename": target.name,
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
                "receipt": receipt,
            }
        )
    manifest = {
        "schema_version": "p2.v52.final.training.20260905.v1",
        "status": "TRAINED_FROM_ORGANIZER_DATA_SCRATCH",
        "candidate_id": contract["candidate_id"],
        "observations_sha256": _sha256(observations_path),
        "training_rows": int(selected.sum()),
        "fit_count": 3,
        "seeds": [20260901, 20260902, 20260903],
        "epochs_per_fit": 60,
        "pretrained_weights_loaded": 0,
        "external_data_rows": 0,
        "weighting": weight_receipt,
        "checkpoints": checkpoints,
        "runtime_seconds": time.perf_counter() - started,
    }
    path = weights_dir.parent / "MODEL_MANIFEST.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=HERE.parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("03_model/weights"))
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.data_dir,
                args.package_dir,
                args.output_dir,
                replace=args.replace,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
