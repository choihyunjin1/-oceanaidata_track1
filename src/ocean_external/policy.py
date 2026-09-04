"""Historical external-data metadata and a fail-closed execution gate.

The 2026-09-01 organizer notice supersedes the older public FAQ permission.
Catalog and receipt readers remain available only to preserve audit evidence;
``preflight_external_use`` rejects every competition use before touching a
manifest or candidate data file. Synthetic-only pretrained weights, if ever
used, require a separate four-condition gate and are not external observations.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class PolicyError(PermissionError):
    """Raised when an external-data use fails permission or provenance gates."""


_ALLOWED_PROBLEMS = frozenset({"P1", "P2", "P3"})
_ALLOWED_PURPOSES = frozenset(
    {"pretraining", "feature_design", "normalization", "augmentation", "fine_tuning"}
)
_READY_RIGHTS = frozenset({"open_verified", "holder_permission_required"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aware_datetime(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone offset")
    return parsed


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return value


@dataclass(frozen=True)
class SourcePolicy:
    source_id: str
    title: str
    source_kind: str
    official_url: str
    doi: str
    license_name: str
    rights_state: str
    eligible_problems: tuple[str, ...]
    allowed_purposes: tuple[str, ...]
    max_observation_time: str
    priority: str
    value_accessed: bool
    notes: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SourcePolicy:
        required = {
            "source_id",
            "title",
            "source_kind",
            "official_url",
            "doi",
            "license_name",
            "rights_state",
            "eligible_problems",
            "allowed_purposes",
            "max_observation_time",
            "priority",
            "value_accessed",
            "notes",
        }
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"source entry is missing {sorted(missing)}")
        problems = tuple(str(item) for item in raw["eligible_problems"])
        purposes = tuple(str(item) for item in raw["allowed_purposes"])
        if not problems or not set(problems) <= _ALLOWED_PROBLEMS:
            raise ValueError(f"invalid eligible_problems for {raw['source_id']}")
        if not purposes or not set(purposes) <= _ALLOWED_PURPOSES:
            raise ValueError(f"invalid allowed_purposes for {raw['source_id']}")
        _aware_datetime(str(raw["max_observation_time"]), field="max_observation_time")
        if raw["rights_state"] not in _READY_RIGHTS | {
            "review_required",
            "training_provenance_review_required",
        }:
            raise ValueError(f"invalid rights_state for {raw['source_id']}")
        if not str(raw["official_url"]).startswith("https://"):
            raise ValueError(f"official_url must use HTTPS for {raw['source_id']}")
        return cls(
            source_id=str(raw["source_id"]),
            title=str(raw["title"]),
            source_kind=str(raw["source_kind"]),
            official_url=str(raw["official_url"]),
            doi=str(raw["doi"]),
            license_name=str(raw["license_name"]),
            rights_state=str(raw["rights_state"]),
            eligible_problems=problems,
            allowed_purposes=purposes,
            max_observation_time=str(raw["max_observation_time"]),
            priority=str(raw["priority"]),
            value_accessed=bool(raw["value_accessed"]),
            notes=str(raw["notes"]),
        )


@dataclass(frozen=True)
class ApprovalReceipt:
    status: str
    received_at: str
    organizer_channel: str
    evidence_file: str
    evidence_sha256: str
    allowed_sources: tuple[str, ...]
    allowed_problems: tuple[str, ...]
    allowed_purposes: tuple[str, ...]
    cutoff_by_problem: Mapping[str, str]
    rights_evidence: Mapping[str, Mapping[str, str]]

    @classmethod
    def load(cls, path: Path) -> ApprovalReceipt:
        raw = _read_json(path)
        if raw.get("schema_version") != "1.0":
            raise PolicyError("approval receipt schema_version must be 1.0")
        required = {
            "status",
            "received_at",
            "organizer_channel",
            "evidence_file",
            "evidence_sha256",
            "allowed_sources",
            "allowed_problems",
            "allowed_purposes",
            "cutoff_by_problem",
            "rights_evidence",
        }
        missing = required - raw.keys()
        if missing:
            raise PolicyError(f"approval receipt is missing {sorted(missing)}")
        _aware_datetime(str(raw["received_at"]), field="received_at")
        return cls(
            status=str(raw["status"]),
            received_at=str(raw["received_at"]),
            organizer_channel=str(raw["organizer_channel"]),
            evidence_file=str(raw["evidence_file"]),
            evidence_sha256=str(raw["evidence_sha256"]).lower(),
            allowed_sources=tuple(str(item) for item in raw["allowed_sources"]),
            allowed_problems=tuple(str(item) for item in raw["allowed_problems"]),
            allowed_purposes=tuple(str(item) for item in raw["allowed_purposes"]),
            cutoff_by_problem={str(k): str(v) for k, v in raw["cutoff_by_problem"].items()},
            rights_evidence={str(k): dict(v) for k, v in raw["rights_evidence"].items()},
        )


@dataclass(frozen=True)
class ExternalManifest:
    source_id: str
    local_file: str
    file_sha256: str
    observed_start: str
    observed_end: str
    row_count: int
    variables: tuple[str, ...]
    transformation_log: str

    @classmethod
    def load(cls, path: Path) -> ExternalManifest:
        raw = _read_json(path)
        if raw.get("schema_version") != "1.0":
            raise PolicyError("external manifest schema_version must be 1.0")
        required = {
            "source_id",
            "local_file",
            "file_sha256",
            "observed_start",
            "observed_end",
            "row_count",
            "variables",
            "transformation_log",
        }
        missing = required - raw.keys()
        if missing:
            raise PolicyError(f"external manifest is missing {sorted(missing)}")
        start = _aware_datetime(str(raw["observed_start"]), field="observed_start")
        end = _aware_datetime(str(raw["observed_end"]), field="observed_end")
        if end < start:
            raise PolicyError("external manifest observed_end precedes observed_start")
        if int(raw["row_count"]) <= 0:
            raise PolicyError("external manifest row_count must be positive")
        return cls(
            source_id=str(raw["source_id"]),
            local_file=str(raw["local_file"]),
            file_sha256=str(raw["file_sha256"]).lower(),
            observed_start=str(raw["observed_start"]),
            observed_end=str(raw["observed_end"]),
            row_count=int(raw["row_count"]),
            variables=tuple(str(item) for item in raw["variables"]),
            transformation_log=str(raw["transformation_log"]),
        )


@dataclass(frozen=True)
class CatalogAudit:
    catalog_sha256: str
    source_count: int
    ready_open_source_count: int
    rights_blocked_source_count: int
    value_accessed_count: int
    unauthorized_value_access_count: int
    accepted: bool


def load_catalog(path: str | Path) -> dict[str, SourcePolicy]:
    catalog_path = Path(path)
    with catalog_path.open("rb") as stream:
        raw = tomllib.load(stream)
    if raw.get("catalog", {}).get("schema_version") != "1.0":
        raise ValueError("catalog schema_version must be 1.0")
    entries = raw.get("source")
    if not isinstance(entries, list) or not entries:
        raise ValueError("catalog must contain at least one [[source]] entry")
    policies = [SourcePolicy.from_mapping(item) for item in entries]
    ids = [item.source_id for item in policies]
    if len(ids) != len(set(ids)):
        raise ValueError("catalog source_id values must be unique")
    return {item.source_id: item for item in policies}


def audit_catalog(path: str | Path) -> CatalogAudit:
    catalog_path = Path(path)
    sources = load_catalog(catalog_path)
    value_accessed = sum(item.value_accessed for item in sources.values())
    unauthorized_access = sum(
        item.value_accessed and item.rights_state != "open_verified" for item in sources.values()
    )
    ready = sum(item.rights_state == "open_verified" for item in sources.values())
    blocked = len(sources) - ready
    return CatalogAudit(
        catalog_sha256=_sha256(catalog_path),
        source_count=len(sources),
        ready_open_source_count=ready,
        rights_blocked_source_count=blocked,
        value_accessed_count=value_accessed,
        unauthorized_value_access_count=unauthorized_access,
        accepted=unauthorized_access == 0,
    )


def _verify_evidence(path_value: str, expected_sha256: str, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_file():
        raise PolicyError(f"{label} evidence file is missing")
    actual = _sha256(path)
    if actual != expected_sha256.lower():
        raise PolicyError(f"{label} evidence SHA256 mismatch")
    return path


def preflight_external_use(
    *,
    catalog_path: str | Path,
    approval_receipt_path: str | Path,
    manifest_path: str | Path,
    problem: str,
    source_id: str,
    purpose: str,
) -> dict[str, Any]:
    """Reject external observations before any supplied path is touched."""

    del catalog_path, approval_receipt_path, manifest_path, problem, source_id, purpose
    raise PolicyError(
        "external-data use is forbidden by the active 2026-09-01 organizer policy; "
        "older FAQ receipts are superseded audit evidence only"
    )
