"""Read-only scoped Git publication audit. It never stages or commits files."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT_FILES = {".gitignore", ".gitattributes", "README.md", "AI_HANDOFF.md"}
TEXT_EXTENSIONS = {".py", ".md", ".json", ".toml", ".txt", ".ipynb"}
SECRET = re.compile(r"(?:sk-[A-Za-z0-9_-]{32,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def permitted(name):
    path = Path(name)
    parts = path.parts
    if name in ROOT_FILES:
        return True
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return False
    if any(p in {"artifacts", "submissions", "03_model", "04_logs", "05_answer", "__pycache__", ".git"} for p in parts):
        return False
    if "LOCK" in path.name or "credential" in name.lower():
        return False
    if parts[0] == "final_packages":
        return "01_data" not in parts or path.name == "README.md"
    if parts[0] == "docs":
        return path.suffix == ".md"
    if parts[0] in {"scripts", "configs", "tests", "reports"} and re.search(r"2026090[67]", name):
        return parts[0] != "reports" or path.suffix in {".md", ".json"}
    return False


def inspect(path):
    payload = path.read_bytes()
    text = payload.decode("utf-8-sig")
    if "\x00" in text or SECRET.search(text):
        raise ValueError("binary/possible secret: " + str(path))
    if path.suffix == ".ipynb":
        notebook = json.loads(text)
        if any(cell.get("outputs") for cell in notebook["cells"]):
            raise ValueError("executed notebook outputs not publishable by this allowlist")
    if path.suffix == ".json":
        def visit(value):
            if isinstance(value, dict):
                if any(k.lower() in {"password", "access_token", "refresh_token", "private_key", "y_true", "y_pred", "raw_rows"} for k in value):
                    raise ValueError("sensitive/row-level JSON needs manual review: " + str(path))
                for member in value.values():
                    visit(member)
            elif isinstance(value, list):
                for member in value:
                    visit(member)
        visit(json.loads(text))
    return {"path": path.as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def audit(root, output):
    names = set()
    for args in (("ls-files", "--modified"), ("ls-files", "--others", "--exclude-standard")):
        names.update(subprocess.check_output(["git", *args], cwd=root, text=True).splitlines())
    selected, excluded = [], []
    for name in sorted(names):
        path = root / name
        if permitted(name) and path.is_file() and not path.is_symlink():
            record = inspect(path)
            record["path"] = name
            selected.append(record)
        else:
            excluded.append(name)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"status": "SCOPED_TEXT_AUDIT_PASS_NOT_STAGED", "selected": selected,
                   "excluded": excluded, "stage_count": 0,
                   "limits": "Pattern/structure audit plus reviewed package provenance; not a general secret-proof claim."}, stream, indent=2)
    print(json.dumps({"selected": len(selected), "bytes": sum(x["bytes"] for x in selected), "excluded": excluded}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(Path.cwd(), args.output)
