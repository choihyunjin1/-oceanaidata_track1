"""Scratch-train and persist the frozen P1 three-seed MS-TCN e150 ensemble."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p1_pipeline import load_surfaces  # noqa: E402


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
    output_dir: str | Path = "03_model/retrained_from_scratch",
    *,
    replace: bool = False,
) -> dict:
    started = time.perf_counter()
    package = Path(package_dir).resolve()
    source, source_config, encoder, training, holdout, _sample = load_surfaces(
        package, data_dir
    )
    _np, _pd, scientific_torch, _model_api, _data_api = source._load_scientific()
    if not scientific_torch.cuda.is_available():
        raise RuntimeError("P1 full scratch training requires CUDA")
    device = scientific_torch.device("cuda")
    weights_dir = Path(output_dir)
    if not weights_dir.is_absolute():
        weights_dir = package / weights_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    for seed in (20260827, 20260839, 20260863):
        target = weights_dir / f"full_width_512_seed_{seed}_epoch_150_state.pt"
        if target.exists() and not replace:
            raise FileExistsError(f"P1 checkpoint exists; use --replace deliberately: {target}")
        capacity = source._config_for_capacity(source_config, width=512, seed=seed)
        scientific_torch.manual_seed(seed)
        scientific_torch.cuda.manual_seed_all(seed)
        model = source._new_model(training.features.shape[1], capacity, device)
        optimizer = scientific_torch.optim.AdamW(
            model.parameters(),
            lr=float(capacity["training"]["learning_rate"]),
            weight_decay=float(capacity["training"]["weight_decay"]),
        )
        windows = source._selected_windows(training, capacity)
        positive_weight = source._positive_weight(training.surface.labels)
        _steps, total_steps, _warmup = source._schedule_geometry(
            capacity, window_count=len(windows)
        )
        global_step = 0
        final_record = None
        seed_started = time.perf_counter()
        for epoch in range(1, 151):
            telemetry, global_step, learning_rate = source._train_epoch(
                model,
                optimizer,
                training,
                windows,
                config=capacity,
                positive_weight=positive_weight,
                device=device,
                epoch=epoch,
                global_step=global_step,
                total_steps=total_steps,
            )
            final_record = source._history_record(
                epoch=epoch,
                telemetry=telemetry,
                global_step=global_step,
                learning_rate=learning_rate,
                elapsed_seconds=time.perf_counter() - seed_started,
            )
            if epoch == 1 or epoch % 10 == 0:
                print(json.dumps({"seed": seed, "epoch": epoch, "epochs": 150}), flush=True)
        temporary = target.with_suffix(".pt.tmp")
        scientific_torch.save(
            {
                "schema_version": "p1.mstcn_e150_full_deployment.state.v1",
                "seed": seed,
                "width": 512,
                "epoch": 150,
                "input_features": 165,
                "state_dict": {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                },
            },
            temporary,
        )
        os.replace(temporary, target)
        checkpoints.append(
            {
                "filename": target.name,
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
                "seed": seed,
                "epochs": 150,
                "optimizer_steps": global_step,
                "final_epoch": final_record,
            }
        )
        del optimizer, model
        torch.cuda.empty_cache()
    manifest = {
        "schema_version": "p1.mstcn_e150.final.training.20260905.v1",
        "status": "TRAINED_FROM_ORGANIZER_DATA_SCRATCH",
        "training_rows": int(training.surface.rows),
        "test_rows_used_without_labels": int(holdout.surface.rows),
        "fit_count": 3,
        "seeds": [20260827, 20260839, 20260863],
        "epochs_per_fit": 150,
        "pretrained_weights_loaded": 0,
        "external_data_rows": 0,
        "encoder": source._encoder_receipt(encoder),
        "checkpoints": checkpoints,
        "runtime_seconds": time.perf_counter() - started,
    }
    path = weights_dir / "TRAINING_MANIFEST.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=HERE.parents[1])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("03_model/retrained_from_scratch")
    )
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
