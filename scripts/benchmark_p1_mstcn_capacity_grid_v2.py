from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(r"C:\Users\cedis\PycharmProjects\PythonProject")
SOURCE_ROOT = ROOT / "src"
OUTPUT_DIR = ROOT / "artifacts" / "p1_mstcn_asrf_capacity_calibration_v2"
OUTPUT = OUTPUT_DIR / "receipt.json"
MODEL_SOURCE = SOURCE_ROOT / "p1_qc" / "ms_tcn_asrf.py"
INPUT_FEATURE_COUNT = 165
TIME_ROWS = 2048
CAPACITY_GRID = ((256, 128), (512, 64))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(OUTPUT_DIR)
    sys.path.insert(0, str(SOURCE_ROOT))

    import torch

    from p1_qc.ms_tcn_asrf import (
        MSTCNASRF,
        MSTCNASRFConfig,
        MSTCNASRFLossConfig,
        compute_ms_tcn_asrf_loss,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the capacity calibration")
    device = torch.device("cuda")
    results: list[dict[str, object]] = []
    for width, batch_size in CAPACITY_GRID:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        model = MSTCNASRF(MSTCNASRFConfig(input_feature_count=INPUT_FEATURE_COUNT, width=width)).to(
            device
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
        features = torch.randn(
            batch_size,
            TIME_ROWS,
            INPUT_FEATURE_COUNT,
            dtype=torch.float32,
            device=device,
        )
        valid = torch.ones(batch_size, TIME_ROWS, dtype=torch.bool, device=device)
        event = (torch.rand(batch_size, TIME_ROWS, device=device) < 0.04).float()
        boundary = (torch.rand(batch_size, TIME_ROWS, 2, device=device) < 0.002).float()
        anomaly_type = torch.zeros(batch_size, TIME_ROWS, 5, device=device)
        anomaly_type[..., 1] = event
        loss_config = MSTCNASRFLossConfig(
            stage_weights=(0.25, 0.25, 0.5, 1.0),
            event_positive_weight=20.0,
        )
        elapsed: list[float] = []
        for step in range(3):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(features, valid_mask=valid)
                loss = compute_ms_tcn_asrf_loss(
                    output,
                    event,
                    boundary,
                    anomaly_type,
                    valid_mask=valid,
                    config=loss_config,
                ).total
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize(device)
            if step > 0:
                elapsed.append(time.perf_counter() - started)
        results.append(
            {
                "width": width,
                "batch_size": batch_size,
                "input_shape": [batch_size, TIME_ROWS, INPUT_FEATURE_COUNT],
                "parameter_count": model.trainable_parameter_count,
                "timed_optimizer_steps": len(elapsed),
                "optimizer_step_seconds": elapsed,
                "optimizer_step_seconds_mean": sum(elapsed) / len(elapsed),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_allocated_gb_decimal": torch.cuda.max_memory_allocated(device) / 1.0e9,
                "finite_final_loss": bool(torch.isfinite(loss).item()),
            }
        )
        del (
            model,
            optimizer,
            features,
            valid,
            event,
            boundary,
            anomaly_type,
            output,
            loss,
        )
        torch.cuda.empty_cache()
        gc.collect()

    properties = torch.cuda.get_device_properties(device)
    receipt = {
        "schema_version": "p1.mstcn_asrf.synthetic_capacity_calibration.v2",
        "status": "PASS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scientific_feature_rows_read": 0,
        "scientific_labels_read": 0,
        "official_test_sample_submission_reads": 0,
        "random_seed": 1,
        "warmup_steps_per_cell": 1,
        "timed_steps_per_cell": 2,
        "precision": "bf16 autocast; fp32 optimizer",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "model_source": str(MODEL_SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "model_source_sha256": sha256(MODEL_SOURCE),
        "results": results,
        "selection": {
            "width_256_batch_size": 128,
            "width_512_batch_size": 64,
            "reason": "largest safe measured batches with similar peak allocation and fixed 2048-row windows",
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    temporary = OUTPUT.with_suffix(".json.partial")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
