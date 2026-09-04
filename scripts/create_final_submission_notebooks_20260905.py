"""Generate the three small, independently executable final notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "final_submission_20260905"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def notebook(problem: str, metric: str, caveat: str) -> nbf.NotebookNode:
    env_name = f"{problem}_DATA_DIR"
    filename = f"{problem}_submission.csv"
    cells = [
        markdown(
            f"# {problem} official final submission\n\n"
            "## Goal\n\n"
            "This notebook is a self-contained, top-to-bottom materializer and validator "
            f"for the frozen clean-lineage {problem} candidate. It reads only organizer-"
            "distributed inputs from the configured data directory and package-local model "
            "or frozen inference assets. It never opens hidden truth and never uploads.\n\n"
            f"Official metric: `{metric}`. {caveat}"
        ),
        markdown(
            "## Setup\n\nRun this notebook with the working directory set to this problem package."
        ),
        code(
            "from pathlib import Path\n"
            "import os\n"
            "import sys\n\n"
            "PACKAGE_DIR = Path.cwd().resolve()\n"
            "if not (PACKAGE_DIR / 'contract.json').is_file():\n"
            "    raise RuntimeError('Run from the packaged problem directory containing contract.json')\n"
            f"raw_data_dir = os.environ.get('{env_name}')\n"
            f"if not raw_data_dir:\n    raise RuntimeError('{env_name} must point to the organizer-distributed {problem} directory')\n"
            "DATA_DIR = Path(raw_data_dir).expanduser().resolve()\n"
            "sys.path.insert(0, str(PACKAGE_DIR))\n"
            "from common import bounded_receipt\n"
            "import run_submission\n\n"
            "print({'package': PACKAGE_DIR.name, 'data_dir_present': DATA_DIR.is_dir()})"
        ),
        markdown(
            "## Step 1 — hash-bound preflight\n\nOnly bounded metadata is displayed; no prediction row is printed."
        ),
        code(
            "preflight = run_submission.preflight(DATA_DIR, PACKAGE_DIR)\n"
            "bounded_receipt(preflight)"
        ),
        markdown("## Step 2 — exact local materialization"),
        code(
            f"output_path = PACKAGE_DIR / 'outputs' / '{filename}'\n"
            "receipt = run_submission.materialize(DATA_DIR, PACKAGE_DIR, output_path)\n"
            "bounded_receipt(receipt)"
        ),
        markdown(
            "## Checks\n\nThe schema, row count, key order, finite/domain rules and frozen SHA-256 are fail-closed."
        ),
        code(
            "assert receipt['status'] == 'READY_EXACT_NOT_UPLOADED'\n"
            "assert receipt['candidate_hash_exact']\n"
            "assert receipt['key_order_exact']\n"
            "assert receipt['package_atomic']\n"
            "assert output_path.is_file()\n"
            "print({'status': receipt['status'], 'rows': receipt['rows'], 'sha256': receipt['sha256']})"
        ),
        markdown(
            "## Next steps\n\n"
            "Keep the generated CSV beside its receipt. Before the website action, compare "
            "the title and one-line summary in `contract.json`, then upload deliberately. "
            "This notebook performs no network action."
        ),
    ]
    value = nbf.v4.new_notebook(cells=cells)
    value.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    value.metadata.language_info = {"name": "python", "version": "3.12"}
    return value


def main() -> None:
    specs = {
        "P1": (
            "F1",
            "Exact mode applies the two-row GI spike add-only patch to the frozen three-seed MS-TCN e150 output.",
        ),
        "P2": (
            "pooled RMSE (C)",
            "Exact mode uses the frozen output of the scratch-trained three-fit v52 ensemble; full training source is bundled for audit.",
        ),
        "P3": (
            "pooled RMSE (m)",
            "Exact mode rebuilds the registered long-lead affine combination from two clean scratch-model components.",
        ),
    }
    for problem, (metric, caveat) in specs.items():
        target = OUT / problem / f"{problem}_final_submission.ipynb"
        target.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(notebook(problem, metric, caveat), target)
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
