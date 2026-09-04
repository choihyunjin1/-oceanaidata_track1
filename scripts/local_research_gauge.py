"""Display a local-only progress gauge backed by an aggregate JSON status file."""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--expected-cv-seconds", type=float, default=665.0)
    return parser.parse_args()


class ResearchGauge:
    def __init__(self, root: tk.Tk, status_file: Path, expected_cv_seconds: float) -> None:
        self.root = root
        self.status_file = status_file.resolve()
        self.expected_cv_seconds = max(float(expected_cv_seconds), 1.0)
        root.title("P1 개선 연구 진행률")
        root.geometry("680x285")
        root.resizable(False, False)
        root.configure(padx=24, pady=20)

        self.title = ttk.Label(root, text="P1 개선 연구", font=("Malgun Gothic", 16, "bold"))
        self.title.pack(anchor="w")
        self.phase = ttk.Label(root, text="상태 파일을 확인하는 중", font=("Malgun Gothic", 11))
        self.phase.pack(anchor="w", pady=(14, 5))
        self.bar = ttk.Progressbar(root, length=630, mode="determinate", maximum=100)
        self.bar.pack(anchor="w", pady=(0, 9))
        self.percent = ttk.Label(root, text="0.0%", font=("Consolas", 12, "bold"))
        self.percent.pack(anchor="w")
        self.eta = ttk.Label(root, text="예상 종료 계산 중", font=("Malgun Gothic", 10))
        self.eta.pack(anchor="w", pady=(4, 0))
        self.detail = ttk.Label(
            root,
            text="로컬 aggregate 상태만 읽습니다. 네트워크·원본 데이터 사용 없음.",
            font=("Malgun Gothic", 9),
            foreground="#555555",
            wraplength=625,
            justify="left",
        )
        self.detail.pack(anchor="w", pady=(12, 0))
        self.updated = ttk.Label(root, text="", font=("Malgun Gothic", 8), foreground="#777777")
        self.updated.pack(anchor="w", pady=(8, 0))
        root.after(250, self.refresh)

    def _read(self) -> dict[str, Any]:
        value = json.loads(self.status_file.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("status JSON must be an object")
        return value

    def refresh(self) -> None:
        try:
            value = self._read()
            progress = min(max(float(value.get("progress", 0.0)), 0.0), 100.0)
            title = str(value.get("title", "P1 개선 연구"))
            phase = str(value.get("phase", "진행 중"))
            detail = str(value.get("detail", ""))
            eta = str(value.get("eta", "확인 중"))
            status = str(value.get("status", "running"))
            if status == "running" and "expected_seconds" in value:
                stamp = datetime.fromisoformat(str(value["updated_at"]))
                elapsed = max((datetime.now().astimezone() - stamp).total_seconds(), 0.0)
                expected = max(float(value["expected_seconds"]), 1.0)
                start = float(value.get("progress_start", progress))
                end = float(value.get("progress_end", 90.0))
                progress = start + (end - start) * min(elapsed / expected, 0.97)
                remaining = max(expected - elapsed, 10.0)
                eta_time = datetime.now().astimezone().timestamp() + remaining
                eta = datetime.fromtimestamp(eta_time).astimezone().strftime("%H:%M:%S KST")
                detail = f"{detail} · 경과 {elapsed / 60:.1f}분"
            if status == "running" and phase == "feature" and progress == 35.0:
                stamp = datetime.fromisoformat(str(value["updated_at"]))
                elapsed = max((datetime.now().astimezone() - stamp).total_seconds(), 0.0)
                progress = 35.0 + 34.0 * min(elapsed / self.expected_cv_seconds, 0.97)
                remaining = max(self.expected_cv_seconds - elapsed, 20.0)
                eta_time = datetime.now().astimezone().timestamp() + remaining
                eta = datetime.fromtimestamp(eta_time).astimezone().strftime("%H:%M:%S KST")
                phase = "3개 outer fold XGBoost 학습·inner 후처리 선택"
                detail = f"CV 경과 {elapsed / 60:.1f}분 · 설정 탐색 없이 fixed 24h 특징 4개만 평가"
            self.root.title(f"P1 개선 연구 — {status}")
            self.title.configure(text=title)
            self.phase.configure(text=phase)
            self.bar["value"] = progress
            self.percent.configure(text=f"{progress:5.1f}%")
            self.eta.configure(text=f"예상 종료: {eta}")
            self.detail.configure(text=detail)
            stamp = datetime.fromtimestamp(self.status_file.stat().st_mtime)
            self.updated.configure(text=f"마지막 갱신: {stamp:%Y-%m-%d %H:%M:%S} KST")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.phase.configure(text="진행 상태 재확인 중")
            self.detail.configure(text=str(exc))
        self.root.after(1000, self.refresh)


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    ttk.Style(root).theme_use("vista")
    ResearchGauge(root, Path(args.status_file), args.expected_cv_seconds)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
