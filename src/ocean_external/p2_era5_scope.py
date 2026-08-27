"""Fail-closed, generation-scoped reconciliation for the P2 ERA5 artifact.

The legacy external-data catalog remains the global default.  This module
validates the append-only v2 amendment which supersedes, but never edits, the
chronologically invalid v1 record.  The override remains limited to the
catalog's legacy P3-only/time-cutoff fields for one hash-bound P2 ERA5
retrieval generation.  It performs no download, model execution, target
access, or submission work.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from ocean_external.policy import PolicyError, SourcePolicy, load_catalog

CANONICAL_AMENDMENT_RELATIVE_PATH = (
    "configs/external_data/era5_p2_scope_amendment_2026-08-21_v2.json"
)
CANONICAL_AMENDMENT_SHA256 = "880da06c982a1c6368e25b3481e29e73658dd170f468fb4be43cfe549d62c427"
SUPERSEDED_AMENDMENT_RELATIVE_PATH = "configs/external_data/era5_p2_scope_amendment_2026-08-21.json"
SUPERSEDED_AMENDMENT_SHA256 = "b2a1fa3059e1ee114b1be4d7f596b02e97ca8354365a3c1ee1bacd6f443940c9"

_SOURCE_ID = "era5_pre2024"
_PROBLEM = "P2"
_GENERATION_ID = "p2_era5_full_retrieval_v1"
_PURPOSES = ("feature_design", "fine_tuning")
_CUTOFF_KST = "2025-12-31T23:59:59+09:00"
_LEGACY_PROBLEMS = ("P3",)
_LEGACY_CUTOFF = "2023-12-31T23:59:59+09:00"
_V2_AMENDMENT_ID = "era5_p2_scope_amendment_2026-08-21_v2"
_V2_ISSUED_AT_KST = "2026-08-21T22:38:02+09:00"
_V1_AMENDMENT_ID = "era5_p2_scope_amendment_2026-08-21"
_V1_DECLARED_ISSUED_AT_KST = "2026-08-21T23:20:00+09:00"
_SUPERSESSION_REASON_CODE = "timestamp_chronology_correction"
_SUPERSESSION_REASON = (
    "The v1 issued_at timestamp was later than its actual creation and validation; "
    "this append-only v2 records the observed current KST without editing v1."
)
_VARIABLES = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "eastward_turbulent_surface_stress",
    "northward_turbulent_surface_stress",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_latent_heat_flux",
    "surface_sensible_heat_flux",
    "land_sea_mask",
)
_FEATURE_VARIABLES = _VARIABLES[:-1]
_ANCILLARY_VARIABLES = _VARIABLES[-1:]
_SURFACE_ENERGY_FLUX_VARIABLES = (
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_latent_heat_flux",
    "surface_sensible_heat_flux",
)
_UNITS_BY_VARIABLE = {
    "10m_u_component_of_wind": "m s**-1",
    "10m_v_component_of_wind": "m s**-1",
    "eastward_turbulent_surface_stress": "N m**-2 s",
    "northward_turbulent_surface_stress": "N m**-2 s",
    "surface_net_solar_radiation": "J m**-2",
    "surface_net_thermal_radiation": "J m**-2",
    "surface_latent_heat_flux": "J m**-2",
    "surface_sensible_heat_flux": "J m**-2",
    "land_sea_mask": "(0 - 1)",
}
_SIGN_BY_VARIABLE = {
    "10m_u_component_of_wind": "positive_eastward",
    "10m_v_component_of_wind": "positive_northward",
    "eastward_turbulent_surface_stress": "positive_eastward_native_accumulated",
    "northward_turbulent_surface_stress": "positive_northward_native_accumulated",
    "surface_net_solar_radiation": "positive_downward_native_accumulated",
    "surface_net_thermal_radiation": "positive_downward_native_accumulated",
    "surface_latent_heat_flux": "positive_downward_native_accumulated",
    "surface_sensible_heat_flux": "positive_downward_native_accumulated",
    "land_sea_mask": "fraction_0_to_1",
}
_LEGACY_MANIFEST_SIGN_SEMANTICS = {
    "10m_u_component_of_wind": "positive eastward",
    "10m_v_component_of_wind": "positive northward",
    "eastward_turbulent_surface_stress": "positive eastward accumulated N m-2 s",
    "northward_turbulent_surface_stress": "positive northward accumulated N m-2 s",
    "surface_net_solar_radiation": "net downward accumulated J m-2",
    "surface_net_thermal_radiation": "net upward accumulated J m-2",
    "surface_latent_heat_flux": "upward accumulated J m-2",
    "surface_sensible_heat_flux": "upward accumulated J m-2",
}
_NATIVE_QNET_FORMULA = (
    "surface_net_solar_radiation + surface_net_thermal_radiation + "
    "surface_latent_heat_flux + surface_sensible_heat_flux"
)


@dataclass(frozen=True)
class P2Era5ScopeGrant:
    """Validated, read-only handle for exactly one external Parquet artifact."""

    accepted: bool
    source_id: str
    problem: str
    purpose: str
    effective_cutoff_kst: str
    amendment_sha256: str
    superseded_amendment_sha256: str
    catalog_sha256: str
    permission_receipt_sha256: str
    permission_evidence_sha256: str
    retrieval_manifest_sha256: str
    parquet_sha256: str
    parquet_relative_path: str
    allowed_variables: tuple[str, ...]
    units_by_variable: Mapping[str, str]
    sign_by_variable: Mapping[str, str]
    flux_storage_convention: str
    surface_energy_flux_positive_direction: str
    native_qnet_formula: str
    parquet_path: Path = field(repr=False)

    def public_dict(self) -> dict[str, Any]:
        """Return aggregate evidence without exposing a local absolute path."""

        return {
            "accepted": self.accepted,
            "source_id": self.source_id,
            "problem": self.problem,
            "purpose": self.purpose,
            "effective_cutoff_kst": self.effective_cutoff_kst,
            "amendment_sha256": self.amendment_sha256,
            "superseded_amendment_sha256": self.superseded_amendment_sha256,
            "catalog_sha256": self.catalog_sha256,
            "permission_receipt_sha256": self.permission_receipt_sha256,
            "permission_evidence_sha256": self.permission_evidence_sha256,
            "retrieval_manifest_sha256": self.retrieval_manifest_sha256,
            "parquet_sha256": self.parquet_sha256,
            "parquet_relative_path": self.parquet_relative_path,
            "allowed_variables": list(self.allowed_variables),
            "units_by_variable": dict(self.units_by_variable),
            "sign_by_variable": dict(self.sign_by_variable),
            "flux_storage_convention": self.flux_storage_convention,
            "surface_energy_flux_positive_direction": (self.surface_energy_flux_positive_direction),
            "native_qnet_formula": self.native_qnet_formula,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PolicyError(f"{label} root must be an object")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PolicyError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise PolicyError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _exact_sequence(value: object, expected: Sequence[str], *, label: str) -> None:
    if not isinstance(value, list) or tuple(value) != tuple(expected):
        raise PolicyError(f"{label} changed")


def _exact_string_mapping(value: object, expected: Mapping[str, str], *, label: str) -> None:
    if not isinstance(value, dict) or value != dict(expected):
        raise PolicyError(f"{label} changed")


def _aware_datetime(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise PolicyError(f"{label} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PolicyError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise PolicyError(f"{label} must include a timezone offset")
    return parsed


def _validate_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PolicyError(f"{label} must be a lowercase SHA256")
    if value != value.lower() or any(char not in "0123456789abcdef" for char in value):
        raise PolicyError(f"{label} must be a lowercase SHA256")
    return value


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PolicyError(f"{label} must be a non-empty POSIX repository-relative path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or ":" in parsed.parts[0]:
        raise PolicyError(f"{label} must stay inside the repository")
    return parsed.as_posix()


def _repo_file(repo_root: Path, relative: str, *, label: str) -> Path:
    root = repo_root.resolve()
    path = (root / Path(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PolicyError(f"{label} resolves outside the repository") from exc
    if not path.is_file():
        raise PolicyError(f"{label} is missing")
    return path


def _validate_binding(
    bindings: Mapping[str, Any],
    name: str,
    *,
    with_details: bool = False,
) -> Mapping[str, Any]:
    binding = _mapping(bindings.get(name), label=f"bindings.{name}")
    expected = {"path", "sha256"}
    if with_details:
        expected |= {
            "bytes",
            "rows",
            "unique_hours",
            "grid_points",
            "chunk_count",
            "observed_start_kst",
            "observed_end_kst",
        }
    _exact_keys(binding, expected, label=f"bindings.{name}")
    _relative_path(binding["path"], label=f"bindings.{name}.path")
    _validate_sha256(binding["sha256"], label=f"bindings.{name}.sha256")
    return binding


def _require_exact_true(value: object, *, label: str) -> None:
    if value is not True:
        raise PolicyError(f"{label} must remain true")


def _require_exact_false(value: object, *, label: str) -> None:
    if value is not False:
        raise PolicyError(f"{label} must remain false")


def _validate_supersession(amendment: Mapping[str, Any], issued: datetime) -> None:
    supersession = _mapping(amendment["supersession"], label="supersession")
    _exact_keys(
        supersession,
        {
            "supersedes_amendment_id",
            "superseded_path",
            "superseded_sha256",
            "superseded_declared_issued_at_kst",
            "reason_code",
            "reason",
            "v1_in_place_edit_forbidden",
            "v1_standalone_actual_grant_forbidden",
        },
        label="supersession",
    )
    if supersession["supersedes_amendment_id"] != _V1_AMENDMENT_ID:
        raise PolicyError("v2 must supersede the exact v1 amendment id")
    if (
        _relative_path(supersession["superseded_path"], label="supersession.superseded_path")
        != SUPERSEDED_AMENDMENT_RELATIVE_PATH
    ):
        raise PolicyError("v2 superseded amendment path changed")
    if (
        _validate_sha256(supersession["superseded_sha256"], label="supersession.superseded_sha256")
        != SUPERSEDED_AMENDMENT_SHA256
    ):
        raise PolicyError("v2 superseded amendment SHA256 changed")
    if supersession["superseded_declared_issued_at_kst"] != _V1_DECLARED_ISSUED_AT_KST:
        raise PolicyError("v1 declared issued_at binding changed")
    superseded_issued = _aware_datetime(
        supersession["superseded_declared_issued_at_kst"],
        label="supersession.superseded_declared_issued_at_kst",
    )
    if superseded_issued.utcoffset() != timedelta(hours=9) or issued >= superseded_issued:
        raise PolicyError("v2 chronology correction must precede the invalid v1 declared time")
    if supersession["reason_code"] != _SUPERSESSION_REASON_CODE:
        raise PolicyError("supersession reason code changed")
    if supersession["reason"] != _SUPERSESSION_REASON:
        raise PolicyError("supersession reason changed")
    _require_exact_true(
        supersession["v1_in_place_edit_forbidden"],
        label="supersession.v1_in_place_edit_forbidden",
    )
    _require_exact_true(
        supersession["v1_standalone_actual_grant_forbidden"],
        label="supersession.v1_standalone_actual_grant_forbidden",
    )


def _validate_flux_value_contract(amendment: Mapping[str, Any]) -> None:
    contract = _mapping(amendment["flux_value_contract"], label="flux_value_contract")
    _exact_keys(
        contract,
        {
            "storage_convention",
            "surface_energy_flux_positive_direction",
            "surface_energy_flux_variables",
            "units_by_variable",
            "sign_by_variable",
            "stress_hourly_mean_conversion",
            "native_qnet",
            "legacy_retrieval_manifest_sign_metadata",
        },
        label="flux_value_contract",
    )
    if contract["storage_convention"] != "native_era5_accumulated_values_unmodified":
        raise PolicyError("ERA5 storage convention changed")
    if contract["surface_energy_flux_positive_direction"] != "downward":
        raise PolicyError("ERA5 surface energy fluxes must remain positive-downward")
    _exact_sequence(
        contract["surface_energy_flux_variables"],
        _SURFACE_ENERGY_FLUX_VARIABLES,
        label="flux_value_contract.surface_energy_flux_variables",
    )
    _exact_string_mapping(
        contract["units_by_variable"],
        _UNITS_BY_VARIABLE,
        label="flux_value_contract.units_by_variable",
    )
    _exact_string_mapping(
        contract["sign_by_variable"],
        _SIGN_BY_VARIABLE,
        label="flux_value_contract.sign_by_variable",
    )
    if contract["stress_hourly_mean_conversion"] != ("native_accumulated_N_m-2_s / 3600_s"):
        raise PolicyError("ERA5 stress conversion changed")
    native_qnet = _mapping(contract["native_qnet"], label="flux_value_contract.native_qnet")
    _exact_keys(
        native_qnet,
        {
            "formula",
            "positive_direction",
            "downstream_sign_flip_allowed",
            "hourly_mean_flux_conversion",
        },
        label="flux_value_contract.native_qnet",
    )
    if native_qnet["formula"] != _NATIVE_QNET_FORMULA:
        raise PolicyError("native qnet formula changed")
    if native_qnet["positive_direction"] != "downward":
        raise PolicyError("native qnet positive direction changed")
    _require_exact_false(
        native_qnet["downstream_sign_flip_allowed"],
        label="flux_value_contract.native_qnet.downstream_sign_flip_allowed",
    )
    if native_qnet["hourly_mean_flux_conversion"] != ("native_accumulated_J_m-2 / 3600_s"):
        raise PolicyError("native qnet hourly conversion changed")
    legacy = _mapping(
        contract["legacy_retrieval_manifest_sign_metadata"],
        label="flux_value_contract.legacy_retrieval_manifest_sign_metadata",
    )
    _exact_keys(
        legacy,
        {"status", "corrected_variables", "bound_manifest_sha256_unchanged"},
        label="flux_value_contract.legacy_retrieval_manifest_sign_metadata",
    )
    if legacy["status"] != "superseded_by_this_contract_without_mutating_bound_manifest":
        raise PolicyError("legacy manifest sign correction status changed")
    _exact_sequence(
        legacy["corrected_variables"],
        (
            "surface_net_thermal_radiation",
            "surface_latent_heat_flux",
            "surface_sensible_heat_flux",
        ),
        label="flux_value_contract.legacy_retrieval_manifest_sign_metadata.corrected_variables",
    )
    _require_exact_true(
        legacy["bound_manifest_sha256_unchanged"],
        label=(
            "flux_value_contract.legacy_retrieval_manifest_sign_metadata."
            "bound_manifest_sha256_unchanged"
        ),
    )


def _validate_amendment_shape(amendment: Mapping[str, Any]) -> None:
    _exact_keys(
        amendment,
        {
            "schema_version",
            "amendment_id",
            "status",
            "issued_at_kst",
            "supersession",
            "append_only_contract",
            "scope",
            "flux_value_contract",
            "bindings",
            "legacy_conflict",
            "precedence_contract",
            "safety_contract",
            "attribution",
        },
        label="amendment",
    )
    if amendment["schema_version"] != "2.0":
        raise PolicyError("amendment schema_version must be 2.0")
    if amendment["amendment_id"] != _V2_AMENDMENT_ID:
        raise PolicyError("amendment_id changed")
    if amendment["status"] != "active_internal_reconciliation":
        raise PolicyError("amendment status is not active")
    issued = _aware_datetime(amendment["issued_at_kst"], label="issued_at_kst")
    if issued.utcoffset() != timedelta(hours=9):
        raise PolicyError("issued_at_kst must use the KST +09:00 offset")
    if amendment["issued_at_kst"] != _V2_ISSUED_AT_KST:
        raise PolicyError("v2 issued_at_kst changed")
    _validate_supersession(amendment, issued)

    append_only = _mapping(amendment["append_only_contract"], label="append_only_contract")
    _exact_keys(
        append_only,
        {"append_only", "in_place_edit_forbidden", "supersession_requires_new_amendment_id"},
        label="append_only_contract",
    )
    for key in append_only:
        _require_exact_true(append_only[key], label=f"append_only_contract.{key}")

    scope = _mapping(amendment["scope"], label="scope")
    _exact_keys(
        scope,
        {
            "source_id",
            "problem",
            "purposes",
            "effective_observation_cutoff_kst",
            "retrieval_generation_id",
            "allowed_variables",
        },
        label="scope",
    )
    if scope["source_id"] != _SOURCE_ID or scope["problem"] != _PROBLEM:
        raise PolicyError("amendment scope must remain limited to era5_pre2024/P2")
    if scope["retrieval_generation_id"] != _GENERATION_ID:
        raise PolicyError("retrieval generation changed")
    _exact_sequence(scope["purposes"], _PURPOSES, label="scope.purposes")
    _exact_sequence(scope["allowed_variables"], _VARIABLES, label="scope.allowed_variables")
    if scope["effective_observation_cutoff_kst"] != _CUTOFF_KST:
        raise PolicyError("P2 ERA5 effective cutoff changed")
    _validate_flux_value_contract(amendment)

    bindings = _mapping(amendment["bindings"], label="bindings")
    _exact_keys(
        bindings,
        {
            "catalog",
            "official_permission_receipt",
            "official_permission_evidence",
            "retrieval_manifest",
            "retrieval_parquet",
        },
        label="bindings",
    )
    for name in (
        "catalog",
        "official_permission_receipt",
        "official_permission_evidence",
        "retrieval_manifest",
    ):
        _validate_binding(bindings, name)
    _validate_binding(bindings, "retrieval_parquet", with_details=True)

    legacy = _mapping(amendment["legacy_conflict"], label="legacy_conflict")
    _exact_keys(
        legacy,
        {
            "catalog_source_id",
            "catalog_eligible_problems",
            "catalog_max_observation_time",
            "reconciled_fields",
            "permission_receipt_scope_note_is_legacy_narrative",
        },
        label="legacy_conflict",
    )
    if legacy["catalog_source_id"] != _SOURCE_ID:
        raise PolicyError("legacy conflict source changed")
    _exact_sequence(
        legacy["catalog_eligible_problems"],
        _LEGACY_PROBLEMS,
        label="legacy_conflict.catalog_eligible_problems",
    )
    if legacy["catalog_max_observation_time"] != _LEGACY_CUTOFF:
        raise PolicyError("legacy catalog cutoff declaration changed")
    _exact_sequence(
        legacy["reconciled_fields"],
        ("eligible_problems", "max_observation_time"),
        label="legacy_conflict.reconciled_fields",
    )
    _require_exact_true(
        legacy["permission_receipt_scope_note_is_legacy_narrative"],
        label="legacy_conflict.permission_receipt_scope_note_is_legacy_narrative",
    )

    precedence = _mapping(amendment["precedence_contract"], label="precedence_contract")
    _exact_keys(
        precedence,
        {
            "mode",
            "rule",
            "limited_to_this_P2_ERA5_generation",
            "overrides_only_legacy_problem_and_cutoff",
            "global_catalog_semantics_unchanged",
            "other_sources_problems_generations_unchanged",
        },
        label="precedence_contract",
    )
    if precedence["mode"] != "scoped_override":
        raise PolicyError("precedence mode must remain a scoped override")
    expected_rule = (
        "official FAQ evidence, then this superseding v2 amendment, then the unchanged "
        "legacy catalog default"
    )
    if precedence["rule"] != expected_rule:
        raise PolicyError("precedence rule changed")
    for key in (
        "limited_to_this_P2_ERA5_generation",
        "overrides_only_legacy_problem_and_cutoff",
        "global_catalog_semantics_unchanged",
        "other_sources_problems_generations_unchanged",
    ):
        _require_exact_true(precedence[key], label=f"precedence_contract.{key}")

    safety = _mapping(amendment["safety_contract"], label="safety_contract")
    _exact_keys(
        safety,
        {
            "research_ablation_only",
            "public_meteorology_covariates_only",
            "attribution_required",
            "hidden_target_temperature_or_salinity_access_authorized",
            "competition_source_data_access_authorized",
            "frozen_oof_value_access_authorized_by_this_amendment",
            "model_execution_authorized_by_this_amendment",
            "submission_creation_authorized",
            "submission_upload_authorized",
            "runner_must_validate_before_external_values",
            "external_off_ablation_required_for_promotion",
        },
        label="safety_contract",
    )
    for key in (
        "research_ablation_only",
        "public_meteorology_covariates_only",
        "attribution_required",
        "runner_must_validate_before_external_values",
        "external_off_ablation_required_for_promotion",
    ):
        _require_exact_true(safety[key], label=f"safety_contract.{key}")
    for key in (
        "hidden_target_temperature_or_salinity_access_authorized",
        "competition_source_data_access_authorized",
        "frozen_oof_value_access_authorized_by_this_amendment",
        "model_execution_authorized_by_this_amendment",
        "submission_creation_authorized",
        "submission_upload_authorized",
    ):
        _require_exact_false(safety[key], label=f"safety_contract.{key}")

    attribution = _mapping(amendment["attribution"], label="attribution")
    _exact_keys(attribution, {"doi", "licence", "text"}, label="attribution")
    if not all(isinstance(attribution[key], str) and attribution[key] for key in attribution):
        raise PolicyError("attribution fields must be non-empty strings")


def _validate_catalog_source(
    source: SourcePolicy,
    *,
    purpose: str,
    legacy: Mapping[str, Any],
) -> None:
    if source.source_id != _SOURCE_ID:
        raise PolicyError("catalog source_id mismatch")
    if tuple(source.eligible_problems) != tuple(legacy["catalog_eligible_problems"]):
        raise PolicyError("catalog legacy eligible_problems no longer match the amendment")
    if source.max_observation_time != legacy["catalog_max_observation_time"]:
        raise PolicyError("catalog legacy cutoff no longer matches the amendment")
    if source.rights_state != "open_verified":
        raise PolicyError("ERA5 rights_state must remain open_verified")
    if purpose not in source.allowed_purposes:
        raise PolicyError("requested purpose is not allowed by the unchanged catalog semantics")


def _validate_permission(
    permission: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    purpose: str,
    evidence_binding: Mapping[str, Any],
) -> None:
    if permission.get("schema_version") != "1.0" or permission.get("status") != "approved":
        raise PolicyError("official permission receipt is not approved schema 1.0")
    if permission.get("organizer_channel") != "official public FAQ API, id=9":
        raise PolicyError("official permission organizer channel changed")
    if _SOURCE_ID not in permission.get("allowed_sources", []):
        raise PolicyError("official permission receipt does not include ERA5")
    if _PROBLEM not in permission.get("allowed_problems", []):
        raise PolicyError("official permission receipt does not include P2")
    if purpose not in permission.get("allowed_purposes", []):
        raise PolicyError("official permission receipt does not include the requested purpose")
    if permission.get("cutoff_by_problem", {}).get(_PROBLEM) != _CUTOFF_KST:
        raise PolicyError("official permission P2 cutoff changed")
    if permission.get("evidence_file") != evidence_binding["path"]:
        raise PolicyError("permission receipt evidence path differs from the amendment")
    if permission.get("evidence_sha256") != evidence_binding["sha256"]:
        raise PolicyError("permission receipt evidence SHA differs from the amendment")
    if (
        evidence.get("schema_version") != "1.0"
        or evidence.get("source_url") != "https://oceanaidata.org/api/faqs"
        or evidence.get("faq_id") != 9
    ):
        raise PolicyError("official FAQ evidence identity changed")
    answer = evidence.get("answer")
    if not isinstance(answer, str) or "외부 공개 데이터 활용도 허용" not in answer:
        raise PolicyError("official FAQ evidence no longer states external-data permission")
    if "출처를 반드시 명시" not in answer:
        raise PolicyError("official FAQ evidence no longer states the attribution condition")


def _validate_retrieval_manifest(
    manifest: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> None:
    bindings = _mapping(amendment["bindings"], label="bindings")
    parquet = _mapping(bindings["retrieval_parquet"], label="bindings.retrieval_parquet")
    scope = _mapping(amendment["scope"], label="scope")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("experiment_id") != _GENERATION_ID
        or manifest.get("research_only") is not True
        or manifest.get("upload_allowed") is not False
    ):
        raise PolicyError("retrieval manifest identity or research-only gate changed")

    source = _mapping(manifest.get("source"), label="retrieval_manifest.source")
    if source.get("catalog_sha256") != bindings["catalog"]["sha256"]:
        raise PolicyError("retrieval manifest catalog binding changed")
    if source.get("doi") != amendment["attribution"]["doi"]:
        raise PolicyError("retrieval manifest DOI differs from the amendment")
    if source.get("licence") != amendment["attribution"]["licence"]:
        raise PolicyError("retrieval manifest licence differs from the amendment")
    if source.get("attribution") != amendment["attribution"]["text"]:
        raise PolicyError("retrieval manifest attribution differs from the amendment")

    _exact_string_mapping(
        manifest.get("units"),
        _UNITS_BY_VARIABLE,
        label="retrieval_manifest.units",
    )
    _exact_string_mapping(
        manifest.get("sign_semantics"),
        _LEGACY_MANIFEST_SIGN_SEMANTICS,
        label="retrieval_manifest.sign_semantics",
    )

    gate = _mapping(
        manifest.get("policy_and_plan_gate"), label="retrieval_manifest.policy_and_plan_gate"
    )
    required_gate_values = {
        "passed": True,
        "chunk_count": parquet["chunk_count"],
        "unique_hour_count": parquet["unique_hours"],
        "expected_row_count": parquet["rows"],
        "maximum_allowed_time_kst": scope["effective_observation_cutoff_kst"],
        "catalog_sha256": bindings["catalog"]["sha256"],
        "permission_receipt_sha256": bindings["official_permission_receipt"]["sha256"],
        "permission_evidence_sha256": bindings["official_permission_evidence"]["sha256"],
        "frozen_oof_read": False,
    }
    for key, expected in required_gate_values.items():
        if gate.get(key) != expected:
            raise PolicyError(f"retrieval manifest policy gate field changed: {key}")

    retrieval_scope = _mapping(manifest.get("scope"), label="retrieval_manifest.scope")
    if retrieval_scope.get("problem") != _PROBLEM:
        raise PolicyError("retrieval manifest problem is not P2")
    _exact_sequence(
        retrieval_scope.get("feature_variables"),
        _FEATURE_VARIABLES,
        label="retrieval_manifest.scope.feature_variables",
    )
    _exact_sequence(
        retrieval_scope.get("validation_ancillary"),
        _ANCILLARY_VARIABLES,
        label="retrieval_manifest.scope.validation_ancillary",
    )
    for key in (
        "hidden_target_temperature_or_salinity_used",
        "competition_source_data_read",
        "frozen_oof_read",
        "model_or_submission_modified",
    ):
        _require_exact_false(retrieval_scope.get(key), label=f"retrieval_manifest.scope.{key}")

    retrieval = _mapping(manifest.get("retrieval"), label="retrieval_manifest.retrieval")
    if retrieval.get("chunk_count") != parquet["chunk_count"]:
        raise PolicyError("retrieval manifest chunk count changed")

    output = _mapping(manifest.get("output"), label="retrieval_manifest.output")
    expected_output = {
        "file": parquet["path"],
        "sha256": parquet["sha256"],
        "bytes": parquet["bytes"],
    }
    for key, expected in expected_output.items():
        if output.get(key) != expected:
            raise PolicyError(f"retrieval manifest output field changed: {key}")

    validation = _mapping(manifest.get("validation"), label="retrieval_manifest.validation")
    expected_validation = {
        "passed": True,
        "rows": parquet["rows"],
        "chunk_count": parquet["chunk_count"],
        "unique_hour_count": parquet["unique_hours"],
        "unique_grid_point_count": parquet["grid_points"],
        "duplicate_key_count": 0,
        "time_start_kst": parquet["observed_start_kst"],
        "time_end_kst": parquet["observed_end_kst"],
        "maximum_allowed_time_kst": scope["effective_observation_cutoff_kst"],
        "utc_to_kst_wall_clock_offset_minutes": 540,
    }
    for key, expected in expected_validation.items():
        if validation.get(key) != expected:
            raise PolicyError(f"retrieval manifest validation field changed: {key}")
    missing = validation.get("missing_values_by_variable")
    if not isinstance(missing, dict) or set(missing) != set(_VARIABLES):
        raise PolicyError("retrieval manifest missingness variables changed")
    if any(value != 0 for value in missing.values()):
        raise PolicyError("retrieval manifest reports missing ERA5 values")

    observed_start = _aware_datetime(parquet["observed_start_kst"], label="observed_start_kst")
    observed_end = _aware_datetime(parquet["observed_end_kst"], label="observed_end_kst")
    cutoff = _aware_datetime(
        scope["effective_observation_cutoff_kst"], label="effective_observation_cutoff_kst"
    )
    if observed_start > observed_end or observed_end > cutoff:
        raise PolicyError("bound ERA5 observation range violates the P2 cutoff")


def validate_p2_era5_scope_contract(
    *,
    amendment: Mapping[str, Any],
    catalog_source: SourcePolicy,
    permission_receipt: Mapping[str, Any],
    permission_evidence: Mapping[str, Any],
    retrieval_manifest: Mapping[str, Any],
    actual_hashes: Mapping[str, str],
    actual_parquet_bytes: int,
    problem: str,
    source_id: str,
    purpose: str,
) -> dict[str, Any]:
    """Purely validate supplied metadata and aggregate bindings.

    Callers cannot widen the source, problem, purpose, generation, variables,
    cutoff, or safety flags.  File opening is deliberately handled by the
    wrapper below so this function is straightforward to test with mappings.
    """

    if problem != _PROBLEM or source_id != _SOURCE_ID:
        raise PolicyError("this amendment only authorizes era5_pre2024 for P2")
    if purpose not in _PURPOSES:
        raise PolicyError(f"purpose is outside this P2 ERA5 amendment: {purpose}")
    _validate_amendment_shape(amendment)

    bindings = _mapping(amendment["bindings"], label="bindings")
    expected_hashes = {
        "amendment": CANONICAL_AMENDMENT_SHA256,
        "superseded_amendment": SUPERSEDED_AMENDMENT_SHA256,
        "catalog": bindings["catalog"]["sha256"],
        "permission_receipt": bindings["official_permission_receipt"]["sha256"],
        "permission_evidence": bindings["official_permission_evidence"]["sha256"],
        "retrieval_manifest": bindings["retrieval_manifest"]["sha256"],
        "retrieval_parquet": bindings["retrieval_parquet"]["sha256"],
    }
    if set(actual_hashes) != set(expected_hashes):
        raise PolicyError("actual hash binding names changed")
    for name, expected in expected_hashes.items():
        if actual_hashes[name] != expected:
            raise PolicyError(f"{name} SHA256 differs from the canonical amendment")

    legacy = _mapping(amendment["legacy_conflict"], label="legacy_conflict")
    _validate_catalog_source(catalog_source, purpose=purpose, legacy=legacy)
    evidence_binding = _mapping(
        bindings["official_permission_evidence"], label="bindings.official_permission_evidence"
    )
    _validate_permission(
        permission_receipt,
        permission_evidence,
        purpose=purpose,
        evidence_binding=evidence_binding,
    )
    _validate_retrieval_manifest(retrieval_manifest, amendment)

    parquet = _mapping(bindings["retrieval_parquet"], label="bindings.retrieval_parquet")
    if actual_parquet_bytes != parquet["bytes"]:
        raise PolicyError("retrieval Parquet byte size differs from the amendment")
    flux = _mapping(amendment["flux_value_contract"], label="flux_value_contract")
    native_qnet = _mapping(flux["native_qnet"], label="flux_value_contract.native_qnet")

    return {
        "accepted": True,
        "source_id": source_id,
        "problem": problem,
        "purpose": purpose,
        "effective_cutoff_kst": _CUTOFF_KST,
        "amendment_sha256": actual_hashes["amendment"],
        "superseded_amendment_sha256": actual_hashes["superseded_amendment"],
        "catalog_sha256": actual_hashes["catalog"],
        "permission_receipt_sha256": actual_hashes["permission_receipt"],
        "permission_evidence_sha256": actual_hashes["permission_evidence"],
        "retrieval_manifest_sha256": actual_hashes["retrieval_manifest"],
        "parquet_sha256": actual_hashes["retrieval_parquet"],
        "parquet_relative_path": parquet["path"],
        "allowed_variables": _VARIABLES,
        "units_by_variable": dict(_UNITS_BY_VARIABLE),
        "sign_by_variable": dict(_SIGN_BY_VARIABLE),
        "flux_storage_convention": flux["storage_convention"],
        "surface_energy_flux_positive_direction": flux["surface_energy_flux_positive_direction"],
        "native_qnet_formula": native_qnet["formula"],
    }


def validate_p2_era5_scope_amendment(
    *,
    repo_root: str | Path,
    amendment_path: str | Path,
    problem: str,
    source_id: str,
    purpose: str,
    candidate_parquet_path: str | Path | None = None,
) -> P2Era5ScopeGrant:
    """Validate the canonical superseding v2 and return its one allowed data path.

    A runner should call this before reading any ERA5 values and then read only
    ``grant.parquet_path``.  Passing v1 alone, an alternate amendment, or a
    candidate copy is rejected even when it has otherwise valid-looking
    metadata.
    """

    root = Path(repo_root).resolve()
    canonical_amendment = _repo_file(
        root, CANONICAL_AMENDMENT_RELATIVE_PATH, label="canonical P2 ERA5 amendment"
    )
    supplied_amendment = Path(amendment_path).resolve()
    if supplied_amendment != canonical_amendment:
        superseded = (root / SUPERSEDED_AMENDMENT_RELATIVE_PATH).resolve()
        if supplied_amendment == superseded:
            raise PolicyError("superseded v1 cannot issue a standalone actual grant")
        raise PolicyError("only the canonical P2 ERA5 v2 amendment path is accepted")
    amendment = _read_json(canonical_amendment, label="P2 ERA5 amendment")
    _validate_amendment_shape(amendment)
    if _sha256(canonical_amendment) != CANONICAL_AMENDMENT_SHA256:
        raise PolicyError("canonical P2 ERA5 amendment SHA256 changed")

    supersession = _mapping(amendment["supersession"], label="supersession")
    superseded_amendment = _repo_file(
        root,
        _relative_path(supersession["superseded_path"], label="supersession.superseded_path"),
        label="superseded P2 ERA5 v1 amendment",
    )
    if _sha256(superseded_amendment) != SUPERSEDED_AMENDMENT_SHA256:
        raise PolicyError("superseded P2 ERA5 v1 amendment SHA256 changed")
    superseded_record = _read_json(superseded_amendment, label="superseded P2 ERA5 v1")
    if (
        superseded_record.get("amendment_id") != _V1_AMENDMENT_ID
        or superseded_record.get("issued_at_kst") != _V1_DECLARED_ISSUED_AT_KST
    ):
        raise PolicyError("superseded P2 ERA5 v1 identity changed")

    bindings = _mapping(amendment["bindings"], label="bindings")
    resolved: dict[str, Path] = {}
    for name in (
        "catalog",
        "official_permission_receipt",
        "official_permission_evidence",
        "retrieval_manifest",
        "retrieval_parquet",
    ):
        binding = _mapping(bindings[name], label=f"bindings.{name}")
        relative = _relative_path(binding["path"], label=f"bindings.{name}.path")
        resolved[name] = _repo_file(root, relative, label=f"bound {name}")

    if candidate_parquet_path is not None:
        supplied_candidate = Path(candidate_parquet_path).resolve()
        if supplied_candidate != resolved["retrieval_parquet"]:
            raise PolicyError("candidate Parquet differs from the amendment-bound artifact")

    actual_hashes = {
        "amendment": _sha256(canonical_amendment),
        "superseded_amendment": _sha256(superseded_amendment),
        "catalog": _sha256(resolved["catalog"]),
        "permission_receipt": _sha256(resolved["official_permission_receipt"]),
        "permission_evidence": _sha256(resolved["official_permission_evidence"]),
        "retrieval_manifest": _sha256(resolved["retrieval_manifest"]),
        "retrieval_parquet": _sha256(resolved["retrieval_parquet"]),
    }
    catalog = load_catalog(resolved["catalog"])
    if source_id not in catalog:
        raise PolicyError("ERA5 source is absent from the bound catalog")
    result = validate_p2_era5_scope_contract(
        amendment=amendment,
        catalog_source=catalog[source_id],
        permission_receipt=_read_json(
            resolved["official_permission_receipt"], label="official permission receipt"
        ),
        permission_evidence=_read_json(
            resolved["official_permission_evidence"], label="official permission evidence"
        ),
        retrieval_manifest=_read_json(
            resolved["retrieval_manifest"], label="ERA5 retrieval manifest"
        ),
        actual_hashes=actual_hashes,
        actual_parquet_bytes=resolved["retrieval_parquet"].stat().st_size,
        problem=problem,
        source_id=source_id,
        purpose=purpose,
    )
    return P2Era5ScopeGrant(**result, parquet_path=resolved["retrieval_parquet"])
