"""Small local-only progress gauge for the current P1 R1 experiment.

The gauge watches aggregate artifact files and process state.  It never opens
the competition CSV files and it does not send telemetry or network traffic.
"""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import ttk

import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--expected-model-seconds", type=float, default=665.0)
    return parser.parse_args()


class ProgressGauge:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.run_dir = Path(args.run_dir).resolve()
        self.report_dir = Path(args.report_dir).resolve()
        self.expected_model_seconds = max(float(args.expected_model_seconds), 1.0)
        self.started_at = datetime.fromtimestamp(self.run_dir.stat().st_ctime)
        self.pid = self._load_pid()

        root.title("P1 R1 테스트 진행률")
        root.geometry("620x250")
        root.resizable(False, False)
        root.configure(padx=24, pady=20)

        self.title = ttk.Label(
            root, text="P1 수온 이상탐지 — R1 검증", font=("Malgun Gothic", 16, "bold")
        )
        self.title.pack(anchor="w")
        self.phase = ttk.Label(root, text="준비 중", font=("Malgun Gothic", 11))
        self.phase.pack(anchor="w", pady=(14, 4))
        self.bar = ttk.Progressbar(root, length=570, mode="determinate", maximum=100)
        self.bar.pack(anchor="w", pady=(0, 10))
        self.percent = ttk.Label(root, text="0%", font=("Consolas", 12, "bold"))
        self.percent.pack(anchor="w")
        self.eta = ttk.Label(root, text="예상 종료 계산 중", font=("Malgun Gothic", 10))
        self.eta.pack(anchor="w", pady=(4, 0))
        self.detail = ttk.Label(
            root,
            text="로컬 artifact만 감시합니다. 원본 데이터·네트워크 사용 없음.",
            font=("Malgun Gothic", 9),
            foreground="#555555",
        )
        self.detail.pack(anchor="w", pady=(12, 0))
        self.root.after(250, self.refresh)

    def _load_pid(self) -> int | None:
        path = self.run_dir / "launcher_pid.txt"
        try:
            return int(path.read_text(encoding="ascii").strip())
        except (FileNotFoundError, ValueError):
            return None

    def _process_running(self) -> bool:
        if self.pid is None:
            return True
        return psutil.pid_exists(self.pid)

    def _state(self) -> tuple[float, str, datetime | None, str]:
        if (self.run_dir / "INVALIDATED.json").exists() or (self.run_dir / "ABORTED.json").exists():
            return 100.0, "실행 중단됨", None, "중단 사유 파일을 확인하세요."
        report_html = self.report_dir / "report.html"
        receipt = self.report_dir / "report.receipt.json"
        artifact = self.report_dir / "artifact.json"
        validation = self.run_dir / "independent_validation.json"
        manifest = self.run_dir / "manifest.json"
        if report_html.exists() and (receipt.exists() or artifact.exists()):
            return 100.0, "완료 — 보고서 생성됨", datetime.now(), str(report_html)
        if artifact.exists():
            return (
                94.0,
                "기술 보고서 렌더링·시각 검증",
                datetime.now() + timedelta(minutes=2),
                str(artifact),
            )
        if validation.exists():
            return (
                84.0,
                "독립 지표 재계산·2,000회 부트스트랩 완료",
                datetime.now() + timedelta(minutes=4),
                str(validation),
            )
        if manifest.exists():
            return (
                75.0,
                "모델 완료 — 독립 OOF 검증",
                datetime.now() + timedelta(minutes=7),
                str(manifest),
            )

        elapsed = max((datetime.now() - self.started_at).total_seconds(), 0.0)
        model_fraction = min(elapsed / self.expected_model_seconds, 0.97)
        percent = 3.0 + 69.0 * model_fraction
        remaining_model = max(self.expected_model_seconds - elapsed, 20.0)
        eta = datetime.now() + timedelta(seconds=remaining_model, minutes=10)
        detail = f"PID {self.pid or '-'} · 경과 {elapsed / 60:.1f}분"
        if not self._process_running():
            stderr = self.run_dir / "stderr.log"
            if not manifest.exists():
                return percent, "모델 프로세스 종료 — 결과 확인 필요", None, str(stderr)
        return percent, "3개 outer fold 학습 + inner 37개 경계 후보 선택", eta, detail

    def refresh(self) -> None:
        try:
            percent, phase, eta, detail = self._state()
            self.bar["value"] = percent
            self.phase.configure(text=phase)
            self.percent.configure(text=f"{percent:5.1f}%")
            eta_text = "예상 종료: 확인 필요" if eta is None else f"예상 종료: {eta:%H:%M:%S} KST"
            self.eta.configure(text=eta_text)
            self.detail.configure(text=detail)
            if percent >= 100 and phase.startswith("완료"):
                self.root.title("P1 R1 테스트 완료")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.phase.configure(text="진행 정보 재확인 중")
            self.detail.configure(text=str(exc))
        self.root.after(1000, self.refresh)


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    ttk.Style(root).theme_use("vista")
    ProgressGauge(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
