"""Run the complete local-only P2 deep-model tournament with a GUI status feed."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import torch

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.deep_data import build_panel
from p2_restore.deep_models import MODEL_SPECS
from p2_restore.deep_training import (
    DEV_BLOCK,
    TrainingConfig,
    blend_grid,
    save_checkpoint,
    train_fold,
)
from p2_restore.model import VALIDATION_BLOCKS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class Progress:
    def __init__(self, path: Path, total_units: float) -> None:
        self.path = path
        self.total_units = total_units
        self.completed = 0.0
        self.current_base = 0.0
        self.current_weight = 0.0
        self.started = time.perf_counter()

    def start_job(self, weight: float) -> None:
        self.current_base = self.completed
        self.current_weight = weight

    def finish_job(self) -> None:
        self.completed += self.current_weight
        self.current_base = self.completed
        self.current_weight = 0.0

    def update(self, phase: str, detail: str, fraction: float, *, status: str = "running") -> None:
        units = self.current_base + self.current_weight * min(max(fraction, 0.0), 1.0)
        progress = 5.0 + 92.0 * units / max(self.total_units, 1.0)
        elapsed = time.perf_counter() - self.started
        eta_seconds = elapsed * (self.total_units - units) / units if units > 0 else 0
        eta = (
            datetime.now().astimezone() + timedelta(seconds=max(eta_seconds, 0))
            if units > 0
            else None
        )
        _write_json(
            self.path,
            {
                "title": "P2 전체 모델 토너먼트",
                "status": status,
                "progress": 100.0 if status == "complete" else progress,
                "phase": phase,
                "detail": detail,
                "eta": eta.strftime("%Y-%m-%d %H:%M:%S KST") if eta else "초기 속도 측정 중",
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )


def _validate_contract(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = [spec.name for spec in MODEL_SPECS]
    if value.get("experiment_id") != "p2_deep_model_tournament_v1":
        raise ValueError("unexpected P2 deep tournament id")
    if value.get("status") != "authorized_local_model_tournament":
        raise ValueError("P2 deep tournament is not authorized")
    if value.get("upload_allowed") is not False or value.get("research_only") is not True:
        raise ValueError("P2 deep tournament must stay local-only")
    if value.get("families") != expected:
        raise ValueError("P2 deep tournament families changed")
    if value["input_contract"]["external_pretrained_weights"] is not False:
        raise ValueError("external pretrained weights are forbidden in this run")
    return value


def _screen_path(root: Path, model: str, learning_rate: float) -> Path:
    return root / "screen" / f"{model}_lr{learning_rate:.0e}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/experiments/p2_deep_model_tournament_v1.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p2_deep_model_tournament_v1")
    )
    parser.add_argument(
        "--status-file", type=Path, default=Path("artifacts/status/p2_model_tournament.json")
    )
    parser.add_argument("--screen-epochs", type=int, default=12)
    parser.add_argument("--screen-patience", type=int, default=4)
    parser.add_argument("--full-epoch-cap", type=int, default=0)
    args = parser.parse_args()

    contract = _validate_contract(args.preregistration)
    if args.screen_epochs != int(contract["optimizer_selection"]["screen_max_epochs"]):
        raise ValueError("screen epoch count differs from the frozen contract")
    if args.screen_patience != int(contract["optimizer_selection"]["screen_patience"]):
        raise ValueError("screen patience differs from the frozen contract")
    data_dir = resolve_data_dir(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    screen_units = sum(len(spec.learning_rates) * args.screen_epochs for spec in MODEL_SPECS)
    full_units = sum(
        min(spec.max_epochs, args.full_epoch_cap) if args.full_epoch_cap else spec.max_epochs
        for spec in MODEL_SPECS
    ) * len(VALIDATION_BLOCKS)
    progress = Progress(args.status_file, float(screen_units + full_units))
    progress.update("data", "원본 해시·계약 검사와 dense public-layer panel 생성", 0.0)
    started = time.perf_counter()
    data = load_p2_data(data_dir)
    panel = build_panel(data.observations)
    incumbent_oof = pd.read_parquet(contract["incumbent"]["oof_path"])

    screen_results: dict[str, list[dict[str, object]]] = {}
    selected_learning_rates: dict[str, float] = {}
    for spec in MODEL_SPECS:
        rows: list[dict[str, object]] = []
        for learning_rate in spec.learning_rates:
            progress.start_job(args.screen_epochs)
            target = _screen_path(args.output_dir, spec.name, learning_rate)
            if target.exists():
                row = json.loads(target.read_text(encoding="utf-8"))
                rows.append(row)
                progress.finish_job()
                progress.update(
                    "optimizer screen",
                    f"재사용 · {spec.name} · lr={learning_rate:g}",
                    0.0,
                )
                continue
            config = TrainingConfig(
                model=spec.name,
                learning_rate=learning_rate,
                weight_decay=spec.weight_decay,
                max_epochs=args.screen_epochs,
                patience=args.screen_patience,
                seed=20260816,
                evaluation_interval=2,
                diffusion_samples=1,
            )

            def screen_callback(
                state: dict[str, object], model_name: str = spec.name, lr: float = learning_rate
            ) -> None:
                epoch = int(state["epoch"])
                progress.update(
                    "optimizer screen",
                    f"{model_name} · lr={lr:g} · epoch {epoch}/{args.screen_epochs} · RMSE {float(state['rmse']):.5f}",
                    epoch / args.screen_epochs,
                )

            result = train_fold(
                panel,
                block="development_2024_jul_aug",
                start=DEV_BLOCK[0],
                stop=DEV_BLOCK[1],
                config=config,
                progress=screen_callback,
            )
            row = result.summary()
            checkpoint = (
                args.output_dir / "checkpoints" / f"screen_{spec.name}_{learning_rate:.0e}.pt"
            )
            row["checkpoint_sha256"] = save_checkpoint(checkpoint, result)
            _write_json(target, row)
            rows.append(row)
            progress.finish_job()
            del result
            gc.collect()
            torch.cuda.empty_cache()
        screen_results[spec.name] = rows
        selected_learning_rates[spec.name] = float(
            min(
                rows,
                key=lambda row: (float(row["best_rmse"]), float(row["config"]["learning_rate"])),
            )["config"]["learning_rate"]
        )

    model_results: dict[str, dict[str, object]] = {}
    for spec in MODEL_SPECS:
        model_root = args.output_dir / "models" / spec.name
        model_root.mkdir(parents=True, exist_ok=True)
        fold_rows: list[dict[str, object]] = []
        fold_oof: list[pd.DataFrame] = []
        max_epochs = (
            min(spec.max_epochs, args.full_epoch_cap) if args.full_epoch_cap else spec.max_epochs
        )
        for number, (block, (start, stop)) in enumerate(VALIDATION_BLOCKS.items()):
            progress.start_job(max_epochs)
            summary_path = model_root / f"{block}.json"
            oof_path = model_root / f"{block}_oof.parquet"
            if summary_path.exists() and oof_path.exists():
                fold_rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
                fold_oof.append(pd.read_parquet(oof_path))
                progress.finish_job()
                progress.update("outer model comparison", f"재사용 · {spec.name} · {block}", 0.0)
                continue
            config = TrainingConfig(
                model=spec.name,
                learning_rate=selected_learning_rates[spec.name],
                weight_decay=spec.weight_decay,
                max_epochs=max_epochs,
                patience=min(spec.patience, max_epochs),
                seed=20260816 + number,
                evaluation_interval=2,
                diffusion_samples=4,
            )

            def fold_callback(
                state: dict[str, object],
                model_name: str = spec.name,
                block_name: str = block,
                epoch_limit: int = max_epochs,
            ) -> None:
                epoch = int(state["epoch"])
                progress.update(
                    "outer model comparison",
                    f"{model_name} · {block_name} · epoch {epoch}/{epoch_limit} · RMSE {float(state['rmse']):.5f}",
                    epoch / epoch_limit,
                )

            result = train_fold(
                panel,
                block=block,
                start=start,
                stop=stop,
                config=config,
                progress=fold_callback,
            )
            summary = result.summary()
            summary["checkpoint_sha256"] = save_checkpoint(model_root / f"{block}.pt", result)
            result.oof.to_parquet(oof_path, index=False, compression="zstd")
            _write_json(summary_path, summary)
            fold_rows.append(summary)
            fold_oof.append(result.oof)
            progress.finish_job()
            del result
            gc.collect()
            torch.cuda.empty_cache()
        oof = pd.concat(fold_oof, ignore_index=True)
        oof_path = model_root / "oof.parquet"
        oof.to_parquet(oof_path, index=False, compression="zstd")
        truth = oof["truth"].to_numpy(float)
        prediction = oof["prediction"].to_numpy(float)
        blend = blend_grid(oof, incumbent_oof)
        model_results[spec.name] = {
            "model": spec.name,
            "selected_learning_rate": selected_learning_rates[spec.name],
            "rows": len(oof),
            "rmse": float(((truth - prediction) ** 2).mean() ** 0.5),
            "parameter_count": fold_rows[0]["parameter_count"],
            "best_epochs": {row["block"]: row["best_epoch"] for row in fold_rows},
            "folds": {row["block"]: row for row in fold_rows},
            "blend": blend,
            "oof_path": oof_path.as_posix(),
            "oof_sha256": _sha256(oof_path),
        }
        _write_json(model_root / "result.json", model_results[spec.name])

    ranking = sorted(
        (
            {
                "model": name,
                "rmse": values["rmse"],
                "blend_rmse": values["blend"]["selected"]["rmse"],
                "deep_weight": values["blend"]["selected"]["deep_weight"],
                "parameter_count": values["parameter_count"],
            }
            for name, values in model_results.items()
        ),
        key=lambda row: (min(row["rmse"], row["blend_rmse"]), row["model"]),
    )
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": contract["experiment_id"],
        "research_only": True,
        "uploaded": False,
        "elapsed_seconds": time.perf_counter() - started,
        "preregistration_sha256": _sha256(args.preregistration),
        "source_sha256": {
            name: _sha256(data_dir / name)
            for name in ("observations.csv", "test_index.csv", "baseline_interp.csv")
        },
        "panel": {
            "rows": len(panel.times),
            "input_channels": panel.inputs.shape[1],
            "segments": int(panel.segment_ids.max()) + 1,
            "target_rows": int(panel.target_mask.sum()),
        },
        "screen": screen_results,
        "selected_learning_rates": selected_learning_rates,
        "incumbent_rmse": contract["incumbent"]["rmse"],
        "models": model_results,
        "ranking": ranking,
        "decision": "MODEL_TOURNAMENT_COMPLETE_NO_UPLOAD",
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "result_sha256": _sha256(result_path),
            "preregistration_sha256": _sha256(args.preregistration),
            "uploaded": False,
        },
    )
    progress.start_job(0)
    progress.update(
        "complete",
        f"8개 계열 완료 · 1위 {ranking[0]['model']} · RMSE {min(ranking[0]['rmse'], ranking[0]['blend_rmse']):.6f} · 업로드 없음",
        1.0,
        status="complete",
    )
    print(
        json.dumps(
            {"ranking": ranking, "elapsed_seconds": result["elapsed_seconds"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
