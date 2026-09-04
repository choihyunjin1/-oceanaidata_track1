from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ocean_external.p2_era5_scope import (
    CANONICAL_AMENDMENT_RELATIVE_PATH,
    CANONICAL_AMENDMENT_SHA256,
    SUPERSEDED_AMENDMENT_RELATIVE_PATH,
    SUPERSEDED_AMENDMENT_SHA256,
    validate_p2_era5_scope_amendment,
    validate_p2_era5_scope_contract,
)
from ocean_external.policy import PolicyError, load_catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_PATH = REPO_ROOT / CANONICAL_AMENDMENT_RELATIVE_PATH
SUPERSEDED_AMENDMENT_PATH = REPO_ROOT / SUPERSEDED_AMENDMENT_RELATIVE_PATH
CATALOG_PATH = REPO_ROOT / "configs/external_data/catalog.toml"
PERMISSION_PATH = REPO_ROOT / "configs/external_data/official_faq_permission.json"
EVIDENCE_PATH = REPO_ROOT / ("configs/external_data/official_faq_external_data_2026-08-21.json")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture() -> dict[str, Any]:
    amendment = _read_json(AMENDMENT_PATH)
    bindings = amendment["bindings"]
    parquet = bindings["retrieval_parquet"]
    scope = amendment["scope"]
    permission = _read_json(PERMISSION_PATH)
    evidence = _read_json(EVIDENCE_PATH)
    manifest = {
        "schema_version": "1.0",
        "experiment_id": scope["retrieval_generation_id"],
        "research_only": True,
        "upload_allowed": False,
        "source": {
            "catalog_sha256": bindings["catalog"]["sha256"],
            "doi": amendment["attribution"]["doi"],
            "licence": amendment["attribution"]["licence"],
            "attribution": amendment["attribution"]["text"],
        },
        "policy_and_plan_gate": {
            "passed": True,
            "chunk_count": parquet["chunk_count"],
            "unique_hour_count": parquet["unique_hours"],
            "expected_row_count": parquet["rows"],
            "maximum_allowed_time_kst": scope["effective_observation_cutoff_kst"],
            "catalog_sha256": bindings["catalog"]["sha256"],
            "permission_receipt_sha256": bindings["official_permission_receipt"]["sha256"],
            "permission_evidence_sha256": bindings["official_permission_evidence"]["sha256"],
            "frozen_oof_read": False,
        },
        "scope": {
            "problem": "P2",
            "feature_variables": scope["allowed_variables"][:-1],
            "validation_ancillary": scope["allowed_variables"][-1:],
            "hidden_target_temperature_or_salinity_used": False,
            "competition_source_data_read": False,
            "frozen_oof_read": False,
            "model_or_submission_modified": False,
        },
        "retrieval": {"chunk_count": parquet["chunk_count"]},
        "output": {
            "file": parquet["path"],
            "sha256": parquet["sha256"],
            "bytes": parquet["bytes"],
        },
        "validation": {
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
            "missing_values_by_variable": {variable: 0 for variable in scope["allowed_variables"]},
        },
        "units": dict(amendment["flux_value_contract"]["units_by_variable"]),
        "sign_semantics": {
            "10m_u_component_of_wind": "positive eastward",
            "10m_v_component_of_wind": "positive northward",
            "eastward_turbulent_surface_stress": ("positive eastward accumulated N m-2 s"),
            "northward_turbulent_surface_stress": ("positive northward accumulated N m-2 s"),
            "surface_net_solar_radiation": "net downward accumulated J m-2",
            "surface_net_thermal_radiation": "net upward accumulated J m-2",
            "surface_latent_heat_flux": "upward accumulated J m-2",
            "surface_sensible_heat_flux": "upward accumulated J m-2",
        },
    }
    actual_hashes = {
        "amendment": CANONICAL_AMENDMENT_SHA256,
        "superseded_amendment": SUPERSEDED_AMENDMENT_SHA256,
        "catalog": bindings["catalog"]["sha256"],
        "permission_receipt": bindings["official_permission_receipt"]["sha256"],
        "permission_evidence": bindings["official_permission_evidence"]["sha256"],
        "retrieval_manifest": bindings["retrieval_manifest"]["sha256"],
        "retrieval_parquet": parquet["sha256"],
    }
    return {
        "amendment": amendment,
        "catalog_source": load_catalog(CATALOG_PATH)["era5_pre2024"],
        "permission_receipt": permission,
        "permission_evidence": evidence,
        "retrieval_manifest": manifest,
        "actual_hashes": actual_hashes,
        "actual_parquet_bytes": parquet["bytes"],
        "problem": "P2",
        "source_id": "era5_pre2024",
        "purpose": "feature_design",
    }


def test_pure_contract_accepts_only_the_bound_generation() -> None:
    result = validate_p2_era5_scope_contract(**_fixture())

    assert result["accepted"] is True
    assert result["problem"] == "P2"
    assert result["source_id"] == "era5_pre2024"
    assert result["amendment_sha256"] == CANONICAL_AMENDMENT_SHA256
    assert result["superseded_amendment_sha256"] == SUPERSEDED_AMENDMENT_SHA256
    assert result["surface_energy_flux_positive_direction"] == "downward"
    assert all(
        result["sign_by_variable"][name] == "positive_downward_native_accumulated"
        for name in (
            "surface_net_solar_radiation",
            "surface_net_thermal_radiation",
            "surface_latent_heat_flux",
            "surface_sensible_heat_flux",
        )
    )


def test_canonical_wrapper_accepts_current_local_artifact() -> None:
    amendment = _read_json(AMENDMENT_PATH)
    parquet_path = REPO_ROOT / amendment["bindings"]["retrieval_parquet"]["path"]
    manifest_path = REPO_ROOT / amendment["bindings"]["retrieval_manifest"]["path"]
    if not parquet_path.is_file() or not manifest_path.is_file():
        pytest.skip("ignored local ERA5 generation is not installed")

    grant = validate_p2_era5_scope_amendment(
        repo_root=REPO_ROOT,
        amendment_path=AMENDMENT_PATH,
        problem="P2",
        source_id="era5_pre2024",
        purpose="feature_design",
        candidate_parquet_path=parquet_path,
    )

    assert grant.accepted is True
    assert grant.parquet_path == parquet_path.resolve()
    assert "parquet_path" not in grant.public_dict()
    assert not Path(grant.public_dict()["parquet_relative_path"]).is_absolute()
    assert grant.superseded_amendment_sha256 == SUPERSEDED_AMENDMENT_SHA256
    assert grant.surface_energy_flux_positive_direction == "downward"


@pytest.mark.parametrize(
    ("field", "value"),
    [("problem", "P3"), ("source_id", "another_source"), ("purpose", "pretraining")],
)
def test_request_scope_cannot_be_widened(field: str, value: str) -> None:
    fixture = _fixture()
    fixture[field] = value

    with pytest.raises(PolicyError, match="only authorizes|outside this P2 ERA5 amendment"):
        validate_p2_era5_scope_contract(**fixture)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("append_only_contract", "append_only"),
        ("precedence_contract", "limited_to_this_P2_ERA5_generation"),
        ("precedence_contract", "global_catalog_semantics_unchanged"),
        ("safety_contract", "runner_must_validate_before_external_values"),
        ("safety_contract", "attribution_required"),
    ],
)
def test_required_positive_guards_fail_closed(section: str, field: str) -> None:
    fixture = _fixture()
    fixture["amendment"][section][field] = False

    with pytest.raises(PolicyError, match="must remain true"):
        validate_p2_era5_scope_contract(**fixture)


@pytest.mark.parametrize(
    "field",
    [
        "hidden_target_temperature_or_salinity_access_authorized",
        "competition_source_data_access_authorized",
        "frozen_oof_value_access_authorized_by_this_amendment",
        "model_execution_authorized_by_this_amendment",
        "submission_creation_authorized",
        "submission_upload_authorized",
    ],
)
def test_forbidden_authorizations_fail_closed(field: str) -> None:
    fixture = _fixture()
    fixture["amendment"]["safety_contract"][field] = True

    with pytest.raises(PolicyError, match="must remain false"):
        validate_p2_era5_scope_contract(**fixture)


@pytest.mark.parametrize(
    "name",
    [
        "amendment",
        "superseded_amendment",
        "catalog",
        "permission_receipt",
        "permission_evidence",
        "retrieval_manifest",
        "retrieval_parquet",
    ],
)
def test_every_bound_sha_is_fail_closed(name: str) -> None:
    fixture = _fixture()
    fixture["actual_hashes"][name] = "0" * 64

    with pytest.raises(PolicyError, match=f"{name} SHA256"):
        validate_p2_era5_scope_contract(**fixture)


def test_catalog_global_semantics_are_not_relaxed() -> None:
    fixture = _fixture()
    fixture["catalog_source"] = replace(fixture["catalog_source"], eligible_problems=("P2", "P3"))

    with pytest.raises(PolicyError, match="legacy eligible_problems"):
        validate_p2_era5_scope_contract(**fixture)


def test_unlisted_catalog_purpose_is_not_overridden() -> None:
    fixture = _fixture()
    fixture["catalog_source"] = replace(
        fixture["catalog_source"], allowed_purposes=("pretraining",)
    )

    with pytest.raises(PolicyError, match="unchanged catalog semantics"):
        validate_p2_era5_scope_contract(**fixture)


@pytest.mark.parametrize("field", ["status", "organizer_channel"])
def test_permission_receipt_identity_is_fail_closed(field: str) -> None:
    fixture = _fixture()
    fixture["permission_receipt"][field] = "changed"

    with pytest.raises(PolicyError, match="permission|organizer"):
        validate_p2_era5_scope_contract(**fixture)


def test_permission_cutoff_cannot_be_widened() -> None:
    fixture = _fixture()
    fixture["permission_receipt"]["cutoff_by_problem"]["P2"] = "2026-12-31T23:59:59+09:00"

    with pytest.raises(PolicyError, match="cutoff changed"):
        validate_p2_era5_scope_contract(**fixture)


def test_attribution_condition_cannot_be_removed() -> None:
    fixture = _fixture()
    fixture["permission_evidence"]["answer"] = "외부 공개 데이터 활용도 허용"

    with pytest.raises(PolicyError, match="attribution condition"):
        validate_p2_era5_scope_contract(**fixture)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden_target_temperature_or_salinity_used", True),
        ("competition_source_data_read", True),
        ("frozen_oof_read", True),
        ("model_or_submission_modified", True),
    ],
)
def test_retrieval_manifest_leakage_guards_are_fail_closed(field: str, value: bool) -> None:
    fixture = _fixture()
    fixture["retrieval_manifest"]["scope"][field] = value

    with pytest.raises(PolicyError, match=field):
        validate_p2_era5_scope_contract(**fixture)


def test_temperature_variable_cannot_be_added() -> None:
    fixture = _fixture()
    fixture["retrieval_manifest"]["scope"]["feature_variables"].append("temperature")

    with pytest.raises(PolicyError, match="feature_variables changed"):
        validate_p2_era5_scope_contract(**fixture)


def test_observation_range_cannot_exceed_cutoff() -> None:
    fixture = _fixture()
    fixture["amendment"]["bindings"]["retrieval_parquet"]["observed_end_kst"] = (
        "2026-01-01T00:00:00+09:00"
    )
    fixture["retrieval_manifest"]["validation"]["time_end_kst"] = "2026-01-01T00:00:00+09:00"

    with pytest.raises(PolicyError, match="violates the P2 cutoff"):
        validate_p2_era5_scope_contract(**fixture)


def test_absolute_binding_path_is_rejected() -> None:
    fixture = _fixture()
    fixture["amendment"]["bindings"]["retrieval_manifest"]["path"] = "C:/private/manifest.json"

    with pytest.raises(PolicyError, match="repository"):
        validate_p2_era5_scope_contract(**fixture)


def test_wrapper_rejects_an_alternate_candidate_before_use(tmp_path: Path) -> None:
    amendment = _read_json(AMENDMENT_PATH)
    parquet_path = REPO_ROOT / amendment["bindings"]["retrieval_parquet"]["path"]
    manifest_path = REPO_ROOT / amendment["bindings"]["retrieval_manifest"]["path"]
    if not parquet_path.is_file() or not manifest_path.is_file():
        pytest.skip("ignored local ERA5 generation is not installed")
    alternate = tmp_path / "alternate.parquet"
    alternate.write_bytes(b"not-the-bound-generation")

    with pytest.raises(PolicyError, match="differs from the amendment-bound artifact"):
        validate_p2_era5_scope_amendment(
            repo_root=REPO_ROOT,
            amendment_path=AMENDMENT_PATH,
            problem="P2",
            source_id="era5_pre2024",
            purpose="feature_design",
            candidate_parquet_path=alternate,
        )


def test_wrapper_rejects_a_config_copy(tmp_path: Path) -> None:
    copied = tmp_path / AMENDMENT_PATH.name
    copied.write_bytes(AMENDMENT_PATH.read_bytes())

    with pytest.raises(PolicyError, match="canonical P2 ERA5 v2 amendment path"):
        validate_p2_era5_scope_amendment(
            repo_root=REPO_ROOT,
            amendment_path=copied,
            problem="P2",
            source_id="era5_pre2024",
            purpose="feature_design",
        )


def test_v1_is_immutable_and_bound_by_v2_supersession() -> None:
    v1 = _read_json(SUPERSEDED_AMENDMENT_PATH)
    v2 = _read_json(AMENDMENT_PATH)

    assert _sha256(SUPERSEDED_AMENDMENT_PATH) == SUPERSEDED_AMENDMENT_SHA256
    assert v2["supersession"]["superseded_sha256"] == SUPERSEDED_AMENDMENT_SHA256
    assert v2["supersession"]["supersedes_amendment_id"] == v1["amendment_id"]
    assert v2["supersession"]["reason_code"] == "timestamp_chronology_correction"
    corrected = datetime.fromisoformat(v2["issued_at_kst"])
    invalid_declared = datetime.fromisoformat(v1["issued_at_kst"])
    assert corrected.utcoffset() == timedelta(hours=9)
    assert corrected < invalid_declared


def test_v1_alone_cannot_issue_an_actual_grant() -> None:
    with pytest.raises(PolicyError, match="superseded v1 cannot issue"):
        validate_p2_era5_scope_amendment(
            repo_root=REPO_ROOT,
            amendment_path=SUPERSEDED_AMENDMENT_PATH,
            problem="P2",
            source_id="era5_pre2024",
            purpose="feature_design",
        )


def test_v1_mapping_is_not_a_v2_scope_contract() -> None:
    fixture = _fixture()
    fixture["amendment"] = _read_json(SUPERSEDED_AMENDMENT_PATH)

    with pytest.raises(PolicyError, match="keys changed|schema_version"):
        validate_p2_era5_scope_contract(**fixture)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("surface_energy_flux_positive_direction",),
            "upward",
            "positive-downward",
        ),
        (
            ("sign_by_variable", "surface_latent_heat_flux"),
            "positive_upward_native_accumulated",
            "sign_by_variable changed",
        ),
        (
            ("units_by_variable", "surface_sensible_heat_flux"),
            "W m**-2",
            "units_by_variable changed",
        ),
        (
            ("native_qnet", "downstream_sign_flip_allowed"),
            True,
            "must remain false",
        ),
    ],
)
def test_positive_downward_native_flux_contract_is_fail_closed(
    path: tuple[str, ...], value: object, message: str
) -> None:
    fixture = _fixture()
    current = fixture["amendment"]["flux_value_contract"]
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value

    with pytest.raises(PolicyError, match=message):
        validate_p2_era5_scope_contract(**fixture)


def test_bound_legacy_manifest_labels_cannot_silently_change() -> None:
    fixture = _fixture()
    fixture["retrieval_manifest"]["sign_semantics"]["surface_latent_heat_flux"] = (
        "positive downward accumulated J m-2"
    )

    with pytest.raises(PolicyError, match="sign_semantics changed"):
        validate_p2_era5_scope_contract(**fixture)
