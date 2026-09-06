"""Restore empty runtime directories omitted by Git; never remove existing files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for problem in ("P1", "P2", "P3"):
    for name in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
        (ROOT / problem / name).mkdir(exist_ok=True)
print("Directories ready. Use a fresh COPY for training; existing models/locks were not modified.")
