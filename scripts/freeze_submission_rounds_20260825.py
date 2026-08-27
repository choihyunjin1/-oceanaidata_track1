"""Freeze and independently validate two immutable three-problem CSV rounds.

This command creates local packages only.  It never opens a browser and never
uploads a file.  Existing package directories are treated as immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_qc.submission import KEY_COLUMNS as P1_KEYS  # noqa: E402
from p1_qc.submission import validate_submission as validate_p1  # noqa: E402
from p2_restore.data import KEYS as P2_KEYS  # noqa: E402
from p2_restore.submission import validate_submission as validate_p2  # noqa: E402
from p3_wave.submission import validate_submission as validate_p3  # noqa: E402

KST = ZoneInfo("Asia/Seoul")
PROBLEM_ORDER = ("P1", "P2", "P3")
OUTPUT_NAMES = {problem: f"{problem}_submission.csv" for problem in PROBLEM_ORDER}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_keys(path: Path, keys: list[str], *, string_keys: bool = True) -> pd.DataFrame:
    if not string_keys:
        return pd.read_csv(path, usecols=keys).loc[:, keys]
    dtype = {name: "string" for name in keys if name in {"station", "time", "case_id"}}
    return pd.read_csv(path, usecols=keys, dtype=dtype).loc[:, keys]


def validate_candidate(problem: str, candidate: Path, index_path: Path) -> dict[str, object]:
    raw = candidate.read_bytes()
    raw.decode("utf-8")
    if not raw.endswith(b"\n"):
        raise ValueError(f"{problem}: CSV does not end with a newline")

    if problem == "P1":
        index = read_keys(index_path, P1_KEYS, string_keys=False)
        report = validate_p1(candidate, index)
        frame = pd.read_csv(candidate, keep_default_na=False)
        keys = P1_KEYS
        value_report = {
            "positive_labels": int(pd.to_numeric(frame["label"]).sum()),
            "positive_rate": float(pd.to_numeric(frame["label"]).mean()),
        }
    elif problem == "P2":
        index = read_keys(index_path, P2_KEYS)
        report = validate_p2(candidate, index)
        frame = pd.read_csv(candidate, dtype={"station": "string", "time": "string"})
        keys = P2_KEYS
        values = pd.to_numeric(frame["temp"], errors="coerce").to_numpy(float)
        value_report = {
            "minimum_temp_c": float(values.min()),
            "maximum_temp_c": float(values.max()),
        }
    elif problem == "P3":
        keys = ["case_id", "station", "lead_h"]
        index = read_keys(index_path, keys)
        frame = pd.read_csv(candidate, dtype={"case_id": "string", "station": "string"})
        validate_p3(frame, index)
        values = pd.to_numeric(frame["hs_pred"], errors="coerce").to_numpy(float)
        report = {"rows": len(frame)}
        value_report = {
            "minimum_hs_m": float(values.min()),
            "maximum_hs_m": float(values.max()),
        }
    else:
        raise ValueError(f"unsupported problem: {problem}")

    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{problem}: missing or duplicate keys")
    if len(frame) != len(index) or not frame[keys].equals(index[keys]):
        raise ValueError(f"{problem}: key order differs from organizer index")

    expected_columns = {
        "P1": P1_KEYS + ["label", "anomaly_type"],
        "P2": P2_KEYS + ["temp"],
        "P3": ["case_id", "station", "lead_h", "hs_pred"],
    }[problem]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"{problem}: unexpected column order {list(frame.columns)}")
    if not np.isfinite(pd.to_numeric(frame.iloc[:, -2 if problem == "P1" else -1], errors="coerce")).all():
        raise ValueError(f"{problem}: non-finite target values")

    return {
        "problem": problem,
        "status": "PASS",
        "data_rows": int(report["rows"]),
        "unique_keys": int(frame.drop_duplicates(keys).shape[0]),
        "columns": list(frame.columns),
        "utf8": True,
        "newline_terminated": True,
        "organizer_key_order_exact": True,
        **value_report,
    }


def memo_text(round_id: str, label: str, problem: str, item: dict[str, object], sha: str) -> str:
    caveat = item["local_evidence"]["confidence"]
    primary_no_go = item.get("primary_no_go")
    lines = [
        f"제출 세트: Round {round_id} — {label}",
        f"문제: {problem}",
        f"제출물 제목: {item['title']}",
        f"한줄요약(접근방식): {item['one_line_summary']}",
        f"CSV 파일: {OUTPUT_NAMES[problem]}",
        f"SHA-256: {sha}",
        "상태: 동결 완료 / 미업로드",
        f"로컬 증거 등급: {caveat}",
    ]
    if primary_no_go:
        lines.append(f"본안 전환 사유: {primary_no_go}")
    lines.append("주의: 공식 점수와 순위 영향은 업로드 후 별도 원장에 기록합니다.")
    return "\n".join(lines) + "\n"


def freeze_round(
    *,
    round_id: str,
    round_spec: dict[str, object],
    staging: Path,
    indexes: dict[str, Path],
    package_root: Path,
    policy_sha256: str,
    batch_spec: dict[str, object],
) -> dict[str, object]:
    round_dir = staging / str(round_spec["directory"])
    round_dir.mkdir()
    qa_results: list[dict[str, object]] = []
    manifest_files: list[dict[str, object]] = []
    sums: list[str] = []

    problems = round_spec["problems"]
    for problem in PROBLEM_ORDER:
        item = problems[problem]
        source = (PROJECT_ROOT / str(item["source"])).resolve()
        expected_sha = str(item["expected_sha256"])
        actual_source_sha = sha256_file(source)
        if actual_source_sha != expected_sha:
            raise ValueError(
                f"{round_id}/{problem}: source SHA mismatch: {actual_source_sha} != {expected_sha}"
            )
        destination = round_dir / OUTPUT_NAMES[problem]
        shutil.copyfile(source, destination)
        destination_sha = sha256_file(destination)
        if destination_sha != actual_source_sha or source.read_bytes() != destination.read_bytes():
            raise ValueError(f"{round_id}/{problem}: copied CSV is not byte-identical")

        qa = validate_candidate(problem, destination, indexes[problem])
        qa.update(
            {
                "file": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": destination_sha,
                "source_destination_byte_identical": True,
            }
        )
        qa_results.append(qa)
        manifest_files.append(
            {
                "problem": problem,
                "file": destination.name,
                "candidate_id": item["candidate_id"],
                "source": item["source"],
                "source_absolute": str(source),
                "frozen_absolute": str(
                    package_root / str(round_spec["directory"]) / destination.name
                ),
                "bytes": destination.stat().st_size,
                "sha256": destination_sha,
                "title": item["title"],
                "one_line_summary": item["one_line_summary"],
                "local_evidence": item["local_evidence"],
                "primary_no_go": item.get("primary_no_go"),
                "risk_flags": item.get("risk_flags", []),
                "evidence_files": [
                    {
                        "path": evidence,
                        "sha256": sha256_file((PROJECT_ROOT / evidence).resolve()),
                    }
                    for evidence in item.get("evidence_files", [])
                ],
                "uploaded": False,
            }
        )
        memo = round_dir / f"{problem}_제출정보.txt"
        memo.write_text(
            memo_text(round_id, str(round_spec["label"]), problem, item, destination_sha),
            encoding="utf-8",
            newline="\n",
        )
        sums.append(f"{destination_sha}  {destination.name}")

    created_at = datetime.now(KST).isoformat()
    set_id = f"{batch_spec['batch_id']}_round_{round_id}"
    manifest = {
        "schema_version": "ocean_hackathon.submission_set.v2",
        "set_id": set_id,
        "round": round_id,
        "round_label": round_spec["label"],
        "team": batch_spec["team"],
        "created_at_kst": created_at,
        "status": "FROZEN_NOT_UPLOADED",
        "immutable_before_official_feedback": True,
        "planned_batch_upload_order": [
            "A/P1", "A/P2", "A/P3", "B/P1", "B/P2", "B/P3"
        ],
        "this_set_upload_order": list(PROBLEM_ORDER),
        "policy_authorization_sha256": policy_sha256,
        "access_audit": {
            "hidden_test_target_reads": 0,
            "official_uploads": 0,
            "candidate_changes_after_intermediate_feedback": 0,
            "final_model_submission_actions": 0
        },
        "current_official": batch_spec["current_official"],
        "files": manifest_files,
    }
    qa_payload = {
        "schema_version": "ocean_hackathon.submission_set_qa.v2",
        "set_id": set_id,
        "validated_at_kst": created_at,
        "status": "PASS",
        "checks": [
            "source expected SHA-256 match",
            "source and frozen destination byte identity",
            "UTF-8 CSV parse",
            "exact header and column order",
            "exact organizer key order",
            "unique nonblank keys",
            "problem-specific finite value domain",
            "newline-terminated file",
        ],
        "results": qa_results,
        "upload_performed": False,
    }
    calibration = {
        "schema_version": "ocean_hackathon.local_official_calibration.v1",
        "set_id": set_id,
        "status": "AWAITING_OFFICIAL_SCORES",
        "policy": batch_spec["calibration_policy"],
        "current_official": batch_spec["current_official"],
        "candidate_local_evidence": {
            item["problem"]: next(
                candidate["local_evidence"]
                for candidate in manifest_files
                if candidate["problem"] == item["problem"]
            )
            for item in qa_results
        },
        "official_receipts": {
            problem: {
                "candidate_raw_score": None,
                "official_raw_delta_signed_gain": None,
                "rank_before": None,
                "rank_after": None,
                "team_total_before": None,
                "team_total_after": None,
                "rank_or_total_flip": None,
                "practical_significance": None,
                "statistical_certainty": next(
                    candidate["local_evidence"]["confidence"]
                    for candidate in manifest_files
                    if candidate["problem"] == problem
                ),
                "receipt_captured_at_kst": None
            }
            for problem in PROBLEM_ORDER
        },
        "descriptive_comparison_after_scoring": {
            "pairwise_direction_agreement": None,
            "pairwise_order_agreement": None,
            "spearman_rank_correlation": None,
            "kendall_rank_correlation": None,
            "rank_or_team_total_reversal": None,
        },
    }
    write_json(round_dir / "SET_MANIFEST.json", manifest)

    if round_id == "B":
        current_path = (
            PROJECT_ROOT / batch_spec["current_official"]["P2"]["candidate_source"]
        ).resolve()
        round_a_path = (
            PROJECT_ROOT / batch_spec["rounds"]["A"]["problems"]["P2"]["source"]
        ).resolve()
        fallback_path = round_dir / OUTPUT_NAMES["P2"]
        current = pd.read_csv(current_path)
        round_a = pd.read_csv(round_a_path)
        fallback = pd.read_csv(fallback_path)
        expected = 0.5 * pd.to_numeric(current["temp"]).to_numpy(float) + 0.5 * pd.to_numeric(
            round_a["temp"]
        ).to_numpy(float)
        actual = pd.to_numeric(fallback["temp"]).to_numpy(float)
        formula_max_abs_error = float(np.max(np.abs(actual - expected)))
        if formula_max_abs_error > 1e-12:
            raise ValueError(
                f"B/P2: fallback formula error {formula_max_abs_error} exceeds 1e-12"
            )
        next(item for item in qa_results if item["problem"] == "P2").update(
            {
                "fallback_formula": "0.50 * current + 0.50 * round_A",
                "fallback_formula_max_abs_error_c": formula_max_abs_error,
                "fallback_formula_tolerance_c": 1e-12,
                "fallback_formula_status": "PASS"
            }
        )

    write_json(round_dir / "SET_QA.json", qa_payload)
    write_json(round_dir / "CALIBRATION_PLAN.json", calibration)
    (round_dir / "SHA256SUMS.txt").write_text(
        "\n".join(sums) + "\n", encoding="ascii", newline="\n"
    )
    return {"set_id": set_id, "directory": round_spec["directory"], "files": manifest_files}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiments" / "submission_rounds_20260825.json",
    )
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--p1-index", type=Path, required=True)
    parser.add_argument("--p2-index", type=Path, required=True)
    parser.add_argument("--p3-index", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    package_root = args.package_root.expanduser().resolve()
    package_root.mkdir(parents=True, exist_ok=True)
    final_dirs = {
        round_id: package_root / round_spec["directory"]
        for round_id, round_spec in spec["rounds"].items()
    }
    existing = [str(path) for path in final_dirs.values() if path.exists()]
    batch_manifest_path = package_root / "20260825_TWO_ROUND_BATCH_MANIFEST.json"
    calibration_ledger_path = package_root / "20260825_LOCAL_OFFICIAL_CALIBRATION_LEDGER.json"
    approval_checklist_path = package_root / "20260825_UPLOAD_APPROVAL_CHECKLIST.txt"
    existing.extend(
        str(path)
        for path in (batch_manifest_path, calibration_ledger_path, approval_checklist_path)
        if path.exists()
    )
    if existing:
        raise FileExistsError(f"refusing to overwrite immutable round directories: {existing}")

    policy_path = (PROJECT_ROOT / spec["policy_authorization"]).resolve()
    policy_sha256 = sha256_file(policy_path)
    indexes = {
        "P1": args.p1_index.expanduser().resolve(),
        "P2": args.p2_index.expanduser().resolve(),
        "P3": args.p3_index.expanduser().resolve(),
    }
    receipts: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix=".freeze_rounds_", dir=package_root) as temporary:
        staging = Path(temporary)
        for round_id in ("A", "B"):
            receipts.append(
                freeze_round(
                    round_id=round_id,
                    round_spec=spec["rounds"][round_id],
                    staging=staging,
                    indexes=indexes,
                    package_root=package_root,
                    policy_sha256=policy_sha256,
                    batch_spec=spec,
                )
            )
        for receipt in receipts:
            source = staging / str(receipt["directory"])
            source.replace(package_root / str(receipt["directory"]))

    batch_manifest = {
        "schema_version": "ocean_hackathon.two_round_batch.v1",
        "batch_id": spec["batch_id"],
        "team": spec["team"],
        "status": "SIX_FILES_FROZEN_NOT_UPLOADED",
        "policy_authorization_sha256": policy_sha256,
        "planned_upload_order": ["A/P1", "A/P2", "A/P3", "B/P1", "B/P2", "B/P3"],
        "exact_uploads": [
            {
                "round": round_id,
                "problem": item["problem"],
                "absolute_path": str(
                    package_root
                    / spec["rounds"][round_id]["directory"]
                    / item["file"]
                ),
                "sha256": item["sha256"],
                "authorized": False,
                "uploaded": False
            }
            for round_id, receipt in zip(("A", "B"), receipts, strict=True)
            for item in receipt["files"]
        ],
        "candidate_b_frozen_before_candidate_a_feedback": True,
        "final_model_submission_forbidden": True
    }
    write_json(batch_manifest_path, batch_manifest)
    calibration_ledger = {
        "schema_version": "ocean_hackathon.local_official_calibration_ledger.v1",
        "batch_id": spec["batch_id"],
        "status": "AWAITING_SIX_OFFICIAL_SCORES",
        "team": spec["team"],
        "policy": spec["calibration_policy"],
        "current_official": spec["current_official"],
        "rounds": {
            round_id: {
                problem: {
                    "candidate_id": spec["rounds"][round_id]["problems"][problem][
                        "candidate_id"
                    ],
                    "local_evidence": spec["rounds"][round_id]["problems"][problem][
                        "local_evidence"
                    ],
                    "official_raw_score": None,
                    "official_raw_delta_signed_gain": None,
                    "rank_before": None,
                    "rank_after": None,
                    "team_total_before": None,
                    "team_total_after": None,
                    "rank_or_total_flip": None,
                    "practical_significance": None
                }
                for problem in PROBLEM_ORDER
            }
            for round_id in ("A", "B")
        },
        "post_score_diagnostics": {
            "per_problem_pairwise_direction_agreement": None,
            "per_problem_local_official_order_agreement": None,
            "descriptive_spearman": None,
            "descriptive_kendall": None,
            "scalar_mapping_fit": False
        }
    }
    write_json(calibration_ledger_path, calibration_ledger)
    checklist_lines = [
        "업로드 승인 전용 체크리스트 — 아직 승인·업로드되지 않음",
        "고정 순서: A/P1 → A/P2 → A/P3 → B/P1 → B/P2 → B/P3",
        ""
    ]
    checklist_lines.extend(
        f"{item['round']}/{item['problem']} | {item['absolute_path']} | SHA-256 {item['sha256']}"
        for item in batch_manifest["exact_uploads"]
    )
    checklist_lines.extend(
        [
            "",
            "사용자에게 위 여섯 절대경로와 SHA-256을 제시한 뒤 명시적 일괄 승인을 받아야 합니다.",
            "최종 모델 제출 버튼은 누르지 않습니다."
        ]
    )
    approval_checklist_path.write_text(
        "\n".join(checklist_lines) + "\n", encoding="utf-8", newline="\n"
    )

    output = {
        "status": "PASS_FROZEN_NOT_UPLOADED",
        "package_root": str(package_root),
        "policy_authorization_sha256": policy_sha256,
        "rounds": receipts,
        "upload_performed": False,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
