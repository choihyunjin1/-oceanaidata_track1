"""Independently recompute local OOF metrics; never train or read official inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {problem: f"{problem.lower()}_score_repair_20260905_v1" for problem in ("P1", "P2", "P3")}


def metric(truth: np.ndarray, prediction: np.ndarray, problem: str) -> dict:
    if problem == "P1":
        if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
            raise ValueError("P1 labels must be binary")
        tp = int(np.count_nonzero((truth == 1) & (prediction == 1)))
        fp = int(np.count_nonzero((truth == 0) & (prediction == 1)))
        fn = int(np.count_nonzero((truth == 1) & (prediction == 0)))
        denominator = 2 * tp + fp + fn
        return {"f1": 2 * tp / denominator if denominator else 0.0, "tp": tp, "fp": fp, "fn": fn}
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    sse = float(np.dot(error, error))
    return {
        "rmse": float(np.sqrt(sse / len(error))),
        "sse": sse,
        "rows": len(error),
        "bias": float(error.mean()),
    }


def audit_arrays(arrays: dict[str, np.ndarray], problem: str) -> dict:
    if problem not in RUNS:
        raise ValueError("Unsupported problem")
    required = {"key", "fold", "truth", "reference", "prediction"}
    if required - arrays.keys():
        raise ValueError(f"Missing OOF fields: {sorted(required - arrays.keys())}")
    values = {name: np.asarray(arrays[name]) for name in required}
    if any(value.ndim != 1 for value in values.values()):
        raise ValueError("All OOF fields must be 1-dimensional")
    n = len(values["truth"])
    if n == 0 or any(len(value) != n for value in values.values()):
        raise ValueError("Empty or misaligned OOF arrays")
    for name in ("key", "fold"):
        if values[name].dtype.kind not in {"U", "S"}:
            raise ValueError("Keys and folds must be pickle-free strings")
        values[name] = values[name].astype(str)
        if np.any(np.char.str_len(values[name]) == 0):
            raise ValueError("Empty key or fold")
    if np.unique(values["key"]).size != n:
        raise ValueError("Duplicate OOF keys")
    for name in ("truth", "reference", "prediction"):
        if values[name].dtype.kind not in {"i", "u", "f", "b"}:
            raise ValueError("Numeric OOF fields required")
        if not np.isfinite(values[name]).all():
            raise ValueError("Non-finite OOF values")
    truth, reference, prediction = (values[name] for name in ("truth", "reference", "prediction"))
    reference_metric = metric(truth, reference, problem)
    candidate_metric = metric(truth, prediction, problem)
    name = "f1" if problem == "P1" else "rmse"
    delta = candidate_metric[name] - reference_metric[name]
    folds = {}
    for fold in np.unique(values["fold"]):
        selected = values["fold"] == fold
        folds[str(fold)] = {
            "rows": int(selected.sum()),
            "reference": metric(truth[selected], reference[selected], problem),
            "candidate": metric(truth[selected], prediction[selected], problem),
        }
    result = {
        "status": "NUMERICAL_QA_PASS",
        "rows": n,
        "unique_keys": n,
        "reference": reference_metric,
        "candidate": candidate_metric,
        "candidate_minus_reference": delta,
        "improves_reference": delta > 0 if problem == "P1" else delta < 0,
        "changed_rows": int(np.count_nonzero(reference != prediction)),
        "folds": folds,
        "claim_limit": "Retrospective internal comparison; not fresh confirmation or official score.",
    }
    if problem == "P1":
        added = (reference == 0) & (prediction == 1)
        removed = (reference == 1) & (prediction == 0)
        result["changes"] = {
            "added_tp": int(np.count_nonzero(added & (truth == 1))),
            "added_fp": int(np.count_nonzero(added & (truth == 0))),
            "removed_tp": int(np.count_nonzero(removed & (truth == 1))),
            "removed_fp": int(np.count_nonzero(removed & (truth == 0))),
        }
    return result


def audit_run(problem: str, repo: Path = ROOT, *, run_id: str | None = None) -> dict:
    permitted = {RUNS[problem]}
    if problem == "P1":
        permitted.add("p1_score_repair_decoder_20260905_v1")
    selected_run = run_id or RUNS[problem]
    if selected_run not in permitted:
        raise ValueError("Unknown experiment; explicit local OOF allowlist required")
    relative = Path("artifacts") / selected_run / "qa_oof.npz"
    path = repo / relative
    if not path.is_file():
        return {
            "status": "PENDING",
            "artifact": relative.as_posix(),
            "reason": "QA archive not yet available",
        }
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: archive[name] for name in ("key", "fold", "truth", "reference", "prediction")
        }
    result = audit_arrays(arrays, problem)
    result["artifact"] = relative.as_posix()
    with path.open("rb") as handle:
        result["archive_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result


def make_notebook(output: Path) -> None:
    import nbformat

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "## tl;dr\nP1/P2/P3의 로컬 OOF를 독립 재계산합니다. 모델을 학습하거나 공식 입력을 읽지 않습니다. PENDING은 완료가 아닙니다."
        ),
        nbformat.v4.new_markdown_cell(
            "## Context & Methods\nF1은 TP/FP/FN 합계, RMSE는 전체 SSE/행수로 계산합니다. 원본 반복 노출 검증이므로 fresh 또는 공식 점수로 해석하지 않습니다.\n\n### Key Assumptions\n러너가 기록한 key/truth 대응과 split/provenance는 별도 코드 QA가 필요합니다. 이 노트북은 수치 검산을 담당합니다."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport sys\nrepo = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'AGENTS.md').is_file()), None)\nif repo is None:\n    raise RuntimeError('Run inside the research repository')\nsys.path.insert(0, str(repo / 'scripts'))\nfrom qa_parallel_score_repair_20260905_v1 import audit_run\n"
        ),
        nbformat.v4.new_markdown_cell(
            "## Data\n입력은 ignored `artifacts/p?_score_repair_20260905_v1/qa_oof.npz` 세 파일뿐입니다. 관측/정답 행은 출력하지 않습니다."
        ),
        nbformat.v4.new_code_cell(
            "audits = {problem: audit_run(problem, repo) for problem in ('P1', 'P2', 'P3')}\n[{ 'problem': p, 'status': a['status'], 'rows': a.get('rows'), 'archive_sha256': a.get('archive_sha256') } for p, a in audits.items()]"
        ),
        nbformat.v4.new_markdown_cell(
            "## Results\n후보−기준: F1은 양수, RMSE는 음수일 때 개선입니다. 동일 후보의 순수 수치 검산이며 제출 자격 전체를 자동 인증하지 않습니다."
        ),
        nbformat.v4.new_code_cell(
            "[{ 'problem': p, 'status': a['status'], 'reference': a.get('reference'), 'candidate': a.get('candidate'), 'delta': a.get('candidate_minus_reference'), 'changed_rows': a.get('changed_rows') } for p, a in audits.items()]"
        ),
        nbformat.v4.new_markdown_cell(
            "## Takeaways\n미완료 결과를 임의로 채우지 않습니다. 공식 점수와 재현성/출처 QA는 별도 영수증을 따릅니다."
        ),
        nbformat.v4.new_code_cell(
            "pending = [p for p, a in audits.items() if a['status'] != 'NUMERICAL_QA_PASS']\nprint('Numerical verification complete' if not pending else 'Incomplete: ' + ', '.join(pending))"
        ),
    ]
    nbformat.validate(notebook)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook", action="store_true", help="Generate a bounded numerical-QA companion"
    )
    args = parser.parse_args()
    if args.notebook:
        target = ROOT / "notebooks/parallel_score_repair_20260905_v1.ipynb"
        if target.exists():
            raise FileExistsError("Companion already exists; preserve it")
        make_notebook(target)
        print(target.relative_to(ROOT).as_posix())
        return
    report = {
        "scope": "Independent numerical OOF audit; no training, official data reads, or upload",
        "problems": {problem: audit_run(problem) for problem in RUNS},
        "conditional_followups": {
            "P1_binary_decoder": audit_run(
                "P1", run_id="p1_score_repair_decoder_20260905_v1"
            )
        },
    }
    destination = ROOT / "reports/parallel_score_repair_20260905_v1/independent-qa.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
