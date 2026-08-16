"""Display a local-only progress gauge for a P2 aggregate status JSON."""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-file", required=True, type=Path)
    args = parser.parse_args()
    root = tk.Tk()
    root.title("P2 최대라운드 수렴 실험")
    root.geometry("700x270")
    root.resizable(False, False)
    root.configure(padx=24, pady=20)
    title = ttk.Label(root, text="P2 최대라운드 수렴 실험", font=("Malgun Gothic", 16, "bold"))
    title.pack(anchor="w")
    phase = ttk.Label(root, text="상태 확인 중", font=("Malgun Gothic", 11))
    phase.pack(anchor="w", pady=(14, 5))
    bar = ttk.Progressbar(root, length=650, mode="determinate", maximum=100)
    bar.pack(anchor="w", pady=(0, 9))
    percent = ttk.Label(root, text="0.0%", font=("Consolas", 12, "bold"))
    percent.pack(anchor="w")
    eta = ttk.Label(root, text="예상 종료 계산 중", font=("Malgun Gothic", 10))
    eta.pack(anchor="w", pady=(4, 0))
    detail = ttk.Label(root, text="로컬 상태 파일만 읽습니다.", wraplength=650)
    detail.pack(anchor="w", pady=(12, 0))

    def refresh() -> None:
        try:
            value = json.loads(args.status_file.read_text(encoding="utf-8"))
            value_progress = min(max(float(value.get("progress", 0)), 0), 100)
            bar["value"] = value_progress
            percent.configure(text=f"{value_progress:5.1f}%")
            title.configure(text=str(value.get("title", "P2 최대라운드 수렴 실험")))
            phase.configure(text=str(value.get("phase", "진행 중")))
            detail.configure(text=str(value.get("detail", "")))
            eta.configure(text=f"예상 종료: {value.get('eta', '자동 계산 중')}")
            if value.get("status") == "complete":
                root.title("P2 최대라운드 수렴 실험 — 완료")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            detail.configure(text=f"상태 재확인 중: {exc}")
        root.after(1000, refresh)

    ttk.Style(root).theme_use("vista")
    root.after(250, refresh)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
