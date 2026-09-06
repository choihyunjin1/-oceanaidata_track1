"""Create thin training/inference notebook entrypoints before package sealing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat


def notebook(problem, role, commands):
    if problem not in {"P1", "P2", "P3"} or role not in {"TRAIN", "PREDICT"}:
        raise ValueError("unknown problem or notebook role")
    if not commands or not all(isinstance(c, list) and c and all(isinstance(x, str) for x in c) for c in commands):
        raise ValueError("explicit Python argument lists required")
    for cmd in commands:
        for arg in cmd:
            if "\n" in arg or "\r" in arg:
                raise ValueError("multiline argument forbidden")
        if Path(cmd[0]).is_absolute() or ".." in Path(cmd[0]).parts or not cmd[0].endswith(".py"):
            raise ValueError("local relative Python entrypoint required")
    intro = (
        f"# {problem} — {role}\n\n## Goal\n"
        "Run the same sealed CLI as the package README, with a fresh Python process per stage. "
        "This notebook is a launcher, not an alternative implementation.\n\n"
        "## Setup\nUse the package root as the notebook working directory. Select its documented "
        f"numerical Python environment, and set `{problem}_DATA_DIR` to the organizer dataset. "
        "Do not put raw data into upload ZIPs.\n\n"
        "## Key assumptions\nTraining must start in a new empty output path. Do not erase models, "
        "logs or consumed locks to rerun. Inference requires the package's successful internal QA. "
        "No upload or final model lock occurs here. Execution logs stay local.\n\n"
        "Validation status: generated launcher; structural/synthetic checks are separate from actual "
        "whole-cold numerical execution. Consult the package receipt for the latter."
    )
    setup = (
        "import os\nimport subprocess\nimport sys\nfrom datetime import UTC, datetime\nfrom pathlib import Path\n\n"
        "package_root = Path.cwd().resolve()\n"
        "assert (package_root / 'README.md').is_file(), 'Open from the extracted package root'\n"
        f"assert os.environ.get('{problem}_DATA_DIR'), 'Set {problem}_DATA_DIR first'\n"
        f"assert Path(os.environ['{problem}_DATA_DIR']).is_dir(), 'Dataset directory missing'\n"
        "print({'python': sys.version.split()[0], 'role': " + repr(role) + "})\n"
    )
    run = (
        f"commands = {commands!r}\n"
        "log_dir = package_root / '06_docs' / 'notebook_launches'\n"
        "log_dir.mkdir(parents=True, exist_ok=True)\n"
        "for index, args in enumerate(commands):\n"
        "    entry = (package_root / args[0]).resolve()\n"
        "    assert entry.is_relative_to(package_root) and entry.is_file(), 'Missing local entrypoint'\n"
        "    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')\n"
        f"    log = log_dir / ('{role.lower()}_' + stamp + '_' + str(index) + '.log')\n"
        "    with log.open('x', encoding='utf-8') as output:\n"
        "        result = subprocess.run([sys.executable, '-I', '-B', *args], cwd=package_root, "
        "stdout=output, stderr=subprocess.STDOUT, check=False)\n"
        "    print({'stage': index + 1, 'returncode': result.returncode, 'log': log.name})\n"
        "    if result.returncode != 0:\n"
        "        raise RuntimeError('Stage failed; preserve its outputs and inspect its log. Do not restart automatically.')\n"
    )
    nb = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(intro),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_markdown_cell("## Steps\nThe next cell executes real work. Do not rerun a consumed training attempt."),
        nbformat.v4.new_code_cell(run),
        nbformat.v4.new_markdown_cell(
            "## Checks and next steps\nA zero exit code is not a leaderboard score. Check the package's "
            "training/QA/answer receipt, row count, SHA and runtime. Preserve fallback files. "
            "Use only the exact validated CSV on its matching problem page; final model designation is separate."
        ),
    ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, f"{role}.ipynb", "exec")
    return nb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    destinations = [args.package / f"{role}.ipynb" for role in ("TRAIN", "PREDICT")]
    if any(path.exists() for path in destinations):
        raise FileExistsError("new notebook targets required; do not mutate a sealed package")
    for path, role in zip(destinations, ("TRAIN", "PREDICT"), strict=True):
        with path.open("x", encoding="utf-8") as stream:
            nbformat.write(notebook(spec["problem"], role, spec[role]), stream)
    print(json.dumps({"status": "GENERATED_STRUCTURE_COMPILE_PASS_NOT_REAL_EXECUTION",
                      "paths": [str(p) for p in destinations]}))


if __name__ == "__main__":
    main()
