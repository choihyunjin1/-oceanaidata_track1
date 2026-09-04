from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_JSON = ROOT / "configs/compliance/organizer_data_policy_20260901.json"
INCUMBENT_JSON = ROOT / "configs/compliance/p3_clean_incumbent_20260901.json"
QA_JSON = ROOT / "reports/p3_clean_incumbent_reset_20260901_v1/independent-qa.json"
EXPECTED_SHA = "ea65370a5c9291868769ad9e54a54707035dc93a01ffa4772d9fd26342f357aa"


def _load_strict(path: Path) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    assert isinstance(value, dict)
    return value


def test_active_policy_is_fail_closed_and_highest_precedence() -> None:
    policy = _load_strict(POLICY_JSON)
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert policy["status"] == "ACTIVE_HIGHEST_PRECEDENCE"
    assert policy["distributed_data_only"] is True
    assert "00_ORGANIZER_DATA_POLICY.md" in agents
    assert "latest written organizer rule" in agents


def test_all_four_synthetic_only_exception_conditions_are_required() -> None:
    policy = _load_strict(POLICY_JSON)
    pretrained = policy["pretrained_weights"]
    assert isinstance(pretrained, dict)
    exception = pretrained["synthetic_only_exception"]
    assert isinstance(exception, dict)

    assert pretrained["real_observation_trained"] == "FORBIDDEN"
    assert pretrained["default_when_provenance_is_unclear"] == "FORBIDDEN"
    assert exception["allowed_only_if_all_conditions_hold"] is True
    assert len(exception["conditions"]) == 4


def test_p3_clean_incumbent_registry_and_independent_qa_agree() -> None:
    policy = _load_strict(POLICY_JSON)
    incumbent = _load_strict(INCUMBENT_JSON)
    qa = _load_strict(QA_JSON)

    policy_p3 = policy["p3"]
    assert isinstance(policy_p3, dict)
    policy_incumbent = policy_p3["provisional_clean_incumbent"]
    assert isinstance(policy_incumbent, dict)
    candidate = incumbent["candidate"]
    assert isinstance(candidate, dict)

    assert incumbent["status"] == "ACTIVE_CLEAN_INCUMBENT"
    assert candidate["csv_sha256"] == EXPECTED_SHA
    assert policy_incumbent["csv_sha256"] == EXPECTED_SHA
    assert qa["status"] == "PASS_ACTIVE_CLEAN_INCUMBENT"
    assert qa["candidate_id"] == candidate["id"]
    assert qa["hashes"]["candidate_csv"] == EXPECTED_SHA


def test_current_must_read_docs_do_not_repeat_superseded_permission() -> None:
    current_docs = "\n".join(
        (ROOT / name).read_text(encoding="utf-8")
        for name in (
            "AGENTS.md",
            "00_MUST_READ_FIRST.md",
            "01_P2_MUST_READ_FIRST.md",
            "02_P3_MUST_READ_FIRST.md",
            "README.md",
        )
    )

    assert "외부 공개 데이터는 사용할 수 있다" not in current_docs
    assert "The official public FAQ API (id 9) allows public external data" not in current_docs
    registry = ROOT / "configs/compliance/p3_clean_incumbent_20260901.json"
    assert "ACTIVE_CLEAN_INCUMBENT" in registry.read_text(encoding="utf-8")
