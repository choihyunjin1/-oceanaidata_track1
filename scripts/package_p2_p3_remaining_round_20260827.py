"""Package the frozen, not-uploaded Round-E P2/P3 candidates without changing CSV bytes."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SOURCE = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260827_round_E_preregistered_P1x3_P2x3_P3x3"
)
OUTPUT = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260827_round_E_remaining_P2x3_P3x3_READY"
)
SOURCE_QA = Path(
    r"C:\Users\cedis\PycharmProjects\PythonProject\reports"
    r"\next_day_breakthrough_deep_research_20260827_v1\independent_bundle_qa.json"
)
CANDIDATES = (
    "P2_1_EXPLOIT_LAYERWISE_QUADRATIC",
    "P2_2_PROBE_ENDPOINT_ENVELOPE",
    "P2_3_PROBE_FULL_PAVA_ENVELOPE",
    "P3_1_EXPLOIT_LONG_NEG2",
    "P3_2_PROBE_LONG_NEG4",
    "P3_3_PROBE_LEAD18_24_NEG4",
)
QUARANTINED_P2_HASHES = {
    "a4482f37cbeb45c306a496ad149f68cc53435dcaf74206691d8b2f3cb3cf6473",
    "5ef474790ebb126b86a6be0ac7265f3846f9d594e117bca93f72c87944a3005b",
    "0f0b7d14643bed9f678805ebb878cf8f408056a62bed8f405e2b42c4e72fdcd3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"append-only output already exists: {OUTPUT}")
    source_manifest_path = SOURCE / "SET_MANIFEST.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_candidates = {item["name"]: item for item in source_manifest["candidates"]}
    if set(CANDIDATES) - set(source_candidates):
        raise KeyError("one or more frozen candidates are absent from the Round-E manifest")

    OUTPUT.mkdir(parents=True)
    frozen = []
    hash_lines = []
    for name in CANDIDATES:
        item = source_candidates[name]
        source_dir = SOURCE / name
        destination_dir = OUTPUT / name
        shutil.copytree(source_dir, destination_dir, copy_function=shutil.copy2)
        csv_name = f"{item['problem']}_submission.csv"
        source_csv = source_dir / csv_name
        destination_csv = destination_dir / csv_name
        source_hash = sha256(source_csv)
        destination_hash = sha256(destination_csv)
        if source_hash != item["sha256"] or destination_hash != source_hash:
            raise AssertionError(f"CSV byte identity failed: {name}")
        if destination_hash in QUARANTINED_P2_HASHES:
            raise AssertionError(f"quarantined P2 surface entered package: {name}")
        memo = destination_dir / f"{item['problem']}_제출정보.txt"
        if not memo.is_file():
            raise FileNotFoundError(memo)
        copied = dict(item)
        copied["source_path"] = item["path"]
        copied["path"] = str(destination_csv)
        copied["copied_byte_identically"] = True
        frozen.append(copied)
        hash_lines.append(f"{destination_hash}  {name}/{csv_name}")
        hash_lines.append(f"{sha256(memo)}  {name}/{memo.name}")

    shutil.copy2(source_manifest_path, OUTPUT / "SOURCE_ROUND_E_SET_MANIFEST.json")
    shutil.copy2(SOURCE_QA, OUTPUT / "SOURCE_ROUND_E_INDEPENDENT_QA.json")
    now = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
    manifest = {
        "schema_version": "ocean_hackathon.round_e_remaining_p2x3_p3x3.v1",
        "status": "FROZEN_READY_NOT_UPLOADED",
        "team": "분당독고다이",
        "created_at_kst": now,
        "source_round": source_manifest["schema_version"],
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_independent_qa_sha256": sha256(SOURCE_QA),
        "official_submissions_performed_by_packager": 0,
        "fresh_explicit_upload_approval_required": True,
        "submit_exact_order": list(CANDIDATES),
        "candidates": frozen,
        "exclusions": {
            "P1": "Round F submissions already handled separately; not included.",
            "P2_checkpoint85": "QUARANTINED_PROTOCOL_VIOLATION_NOT_SUBMISSION_ELIGIBLE",
            "P3_ERA5": "Experiment incomplete at packaging time; not included or modified.",
        },
    }
    manifest_path = OUTPUT / "P2_P3_SET_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    plan = """2026-08-27 P2×3·P3×3 남은 제출 세트

결론: 기존 Round E 독립 QA PASS 파일을 바이트 변경 없이 문제별로 정리했습니다.
업로드는 수행하지 않았으며, 실제 업로드 직전 사용자의 새 명시 승인이 필요합니다.

P2 제출 순서
1. P2_1_EXPLOIT_LAYERWISE_QUADRATIC
2. P2_2_PROBE_ENDPOINT_ENVELOPE
3. P2_3_PROBE_FULL_PAVA_ENVELOPE

P3 제출 순서
1. P3_1_EXPLOIT_LONG_NEG2
2. P3_2_PROBE_LONG_NEG4
3. P3_3_PROBE_LEAD18_24_NEG4

각 폴더에서 CSV만 홈페이지에 올리고, 같은 폴더의 제출정보.txt에서 제목과 한줄요약을 복사합니다.
중간 점수를 보고 나머지 파일을 교체하지 않는 동결 배치입니다.
P2 checkpoint-0.85 격리 파일과 미완료 P3 ERA5 결과는 포함하지 않았습니다.
"""
    (OUTPUT / "READY_SUBMISSION_PLAN.txt").write_text(plan, encoding="utf-8")
    hash_lines.extend(
        [
            f"{sha256(OUTPUT / 'SOURCE_ROUND_E_SET_MANIFEST.json')}  SOURCE_ROUND_E_SET_MANIFEST.json",
            f"{sha256(OUTPUT / 'SOURCE_ROUND_E_INDEPENDENT_QA.json')}  SOURCE_ROUND_E_INDEPENDENT_QA.json",
            f"{sha256(manifest_path)}  P2_P3_SET_MANIFEST.json",
            f"{sha256(OUTPUT / 'READY_SUBMISSION_PLAN.txt')}  READY_SUBMISSION_PLAN.txt",
        ]
    )
    (OUTPUT / "SHA256SUMS.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    qa = {
        "schema_version": "ocean_hackathon.round_e_remaining_p2x3_p3x3.qa.v1",
        "status": "PASS_READY_NOT_UPLOADED",
        "candidate_count": len(frozen),
        "candidate_hashes_match_source_manifest": True,
        "candidate_copies_byte_identical": True,
        "submission_memos_present": True,
        "quarantined_p2_hashes_present": False,
        "p3_era5_included": False,
        "official_upload_performed": False,
        "hidden_target_values_read": False,
        "manifest_sha256": sha256(manifest_path),
    }
    qa_path = OUTPUT / "INDEPENDENT_COPY_QA.json"
    qa_path.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), **qa}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
