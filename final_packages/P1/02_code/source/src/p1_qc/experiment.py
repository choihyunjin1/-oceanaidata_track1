"""Reproducible experiment manifests and artifact bookkeeping."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_sha(root: Path = PROJECT_ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def environment_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_sha": git_sha(),
    }
    try:
        import torch

        summary["torch"] = torch.__version__
        summary["cuda_runtime"] = torch.version.cuda
        summary["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            summary["cuda_device"] = torch.cuda.get_device_name(0)
            summary["cuda_capability"] = list(torch.cuda.get_device_capability(0))
            summary["cuda_arch_list"] = torch.cuda.get_arch_list()
    except ImportError:
        summary["torch"] = None
    return summary


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


class RunRecorder:
    """Create an ignored run directory and record traceable artifacts."""

    def __init__(
        self,
        command: str,
        config: Any,
        *,
        root: str | Path = PROJECT_ROOT / "artifacts" / "runs",
        run_id: str | None = None,
        seed: int = 20260813,
    ) -> None:
        config_value = asdict(config) if is_dataclass(config) else config
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        self.run_id = run_id or f"{timestamp}_{command}_{stable_hash(config_value)[:8]}"
        self.path = Path(root) / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.command = command
        self.seed = seed
        seed_everything(seed)
        self.manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "command": command,
            "created_at": datetime.now().astimezone().isoformat(),
            "seed": seed,
            "config_hash": stable_hash(config_value),
            "config": config_value,
            "environment": environment_summary(),
            "status": "running",
        }
        write_json(self.path / "manifest.json", self.manifest)

    def add_inputs(self, **paths: str | Path) -> None:
        inputs = self.manifest.setdefault("inputs", {})
        for name, raw_path in paths.items():
            path = Path(raw_path).resolve()
            inputs[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        write_json(self.path / "manifest.json", self.manifest)

    def copy_config(self, config_path: str | Path) -> None:
        source = Path(config_path)
        if source.is_file():
            shutil.copy2(source, self.path / "config.toml")

    def record_json(self, name: str, value: Any) -> Path:
        path = write_json(self.path / name, value)
        self.manifest.setdefault("artifacts", {})[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        write_json(self.path / "manifest.json", self.manifest)
        return path

    def record_file(self, path: str | Path, name: str | None = None) -> dict[str, Any]:
        path = Path(path)
        record = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        self.manifest.setdefault("artifacts", {})[name or path.name] = record
        write_json(self.path / "manifest.json", self.manifest)
        return record

    def finish(self, status: str = "complete", **summary: Any) -> None:
        self.manifest["status"] = status
        self.manifest["finished_at"] = datetime.now().astimezone().isoformat()
        self.manifest.update(summary)
        write_json(self.path / "manifest.json", self.manifest)
