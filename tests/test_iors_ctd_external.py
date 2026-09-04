from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from ocean_external.iors_ctd import (  # noqa: E402
    IorsCtdError,
    YearProfile,
    build_loo_dataset,
    depth_linear_baseline,
    load_json_object,
    read_year_profile,
    validate_source_manifest,
    verify_archive,
)
from ocean_external.iors_precheck import apply_stop_gate, evaluate_precheck  # noqa: E402

SOURCE_MANIFEST = Path("configs/external_data/i_ors_ctd_v1_1_1.json")


def _days_since_1950(values: list[str]) -> np.ndarray:
    epoch = np.datetime64("1950-01-01T00:00:00", "s")
    time = np.asarray(values, dtype="datetime64[s]")
    return (time - epoch).astype("timedelta64[s]").astype(np.float64) / 86400.0


def _member_bytes(year: int, times: list[str] | None = None) -> bytes:
    values = times or [f"{year}-01-01T00:00:00", f"{year}-01-01T00:10:00"]
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as dataset:
        dataset.attrs["platform_code"] = "KORS I-ORS"
        dataset.attrs["citation"] = "doi.or.kr/10.22808/DATA-2024-6"
        dataset.attrs["license"] = "CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)"
        dataset.attrs["Conventions"] = "OceanSITES-1.4"
        time = dataset.create_dataset("TIME", data=_days_since_1950(values))
        time.attrs["units"] = "days since 1950-01-01T00:00:00Z"
        count = len(values)
        for suffix, depth, temp in (("1", 5.0, 10.0), ("2", 10.0, 12.0), ("3", 15.0, 14.0)):
            temperature = dataset.create_dataset(
                f"TEMP{suffix}", data=np.arange(count, dtype=float) + temp
            )
            temperature.attrs["comment"] = f"target depth = {depth:g}m"
            dataset.create_dataset(f"TEMP{suffix}_QC", data=np.ones(count))
            salinity = dataset.create_dataset(
                f"PSAL{suffix}", data=np.arange(count, dtype=float) * 0.1 + 33.0
            )
            salinity.attrs["comment"] = f"target depth = {depth:g}m"
            dataset.create_dataset(f"PSAL{suffix}_QC", data=np.ones(count))
            measured_depth = dataset.create_dataset(
                f"DEPTH{suffix}", data=np.full(count, depth, dtype=float)
            )
            measured_depth.attrs["comment"] = f"target depth = {depth:g}m"
            dataset.create_dataset(f"DEPTH{suffix}_QC", data=np.ones(count))
    return buffer.getvalue()


def _archive(tmp_path: Path, *, custom_2023_times: list[str] | None = None) -> tuple[Path, dict]:
    path = tmp_path / "ver1.1.1.zip"
    uncompressed = 0
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for year in range(2014, 2024):
            payload = _member_bytes(year, custom_2023_times if year == 2023 else None)
            uncompressed += len(payload)
            bundle.writestr(f"OS_I-ORS_{year}_D_ocean_CTD.nc", payload)
    manifest = load_json_object(SOURCE_MANIFEST)
    manifest["archive"]["size_bytes"] = path.stat().st_size
    manifest["archive"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["archive"]["uncompressed_size_bytes"] = uncompressed
    return path, manifest


def test_official_manifest_pins_exact_cutoff_and_open_license() -> None:
    manifest = load_json_object(SOURCE_MANIFEST)

    validate_source_manifest(manifest)

    assert manifest["coverage"]["hard_cutoff_kst"] == "2023-12-31T23:50:00+09:00"
    assert manifest["coverage"]["hard_cutoff_utc"] == "2023-12-31T14:50:00+00:00"
    assert manifest["license"]["spdx"] == "CC-BY-4.0"
    assert manifest["archive"]["sha256"] == (
        "c6a0ffaf6367c5a6555d6fe3e7fd5354797538ba52872d96569b3b5c4010f964"
    )


def test_manifest_rejects_relaxed_cutoff() -> None:
    manifest = load_json_object(SOURCE_MANIFEST)
    manifest["coverage"]["hard_cutoff_kst"] = "2024-01-01T00:00:00+09:00"
    manifest["coverage"]["hard_cutoff_utc"] = "2023-12-31T15:00:00+00:00"

    with pytest.raises(IorsCtdError, match="must not be relaxed"):
        validate_source_manifest(manifest)


def test_archive_verifies_all_ten_oceansites_members(tmp_path: Path) -> None:
    path, manifest = _archive(tmp_path)

    audit = verify_archive(path, manifest)

    assert audit["integrity_verified"] is True
    assert audit["member_count"] == 10
    assert [item["year"] for item in audit["members"]] == list(range(2014, 2024))
    assert all(all(item["checks"].values()) for item in audit["members"])


def test_reader_applies_qc1_depth_mapping_and_kst_cutoff(tmp_path: Path) -> None:
    times = [
        "2023-12-31T14:40:00",
        "2023-12-31T14:50:00",
        "2023-12-31T15:00:00",
    ]
    path, manifest = _archive(tmp_path, custom_2023_times=times)
    # Mark a pre-cutoff value bad and a post-cutoff value good; neither may leak through.
    with zipfile.ZipFile(path, "r") as source:
        members = {name: source.read(name) for name in source.namelist()}
    member_name = "OS_I-ORS_2023_D_ocean_CTD.nc"
    buffer = io.BytesIO(members[member_name])
    with h5py.File(buffer, "r+") as dataset:
        dataset["TEMP1_QC"][...] = [1.0, 4.0, 1.0]
        dataset["TEMP1"][...] = [10.0, 999.0, 20.0]
        dataset["DEPTH2_QC"][...] = [1.0, 9.0, 1.0]
    members[member_name] = buffer.getvalue()
    uncompressed = sum(len(value) for value in members.values())
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, value in members.items():
            target.writestr(name, value)
    manifest["archive"]["size_bytes"] = path.stat().st_size
    manifest["archive"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["archive"]["uncompressed_size_bytes"] = uncompressed

    profile = read_year_profile(
        path,
        manifest,
        year=2023,
        target_depth_by_layer={1: 5.0, 2: 10.0, 3: 15.0},
        max_mapping_distance_m=2.0,
    )

    assert profile.time_utc.tolist() == [
        np.datetime64("2023-12-31T14:40:00"),
        np.datetime64("2023-12-31T14:50:00"),
    ]
    assert profile.audit["time_rows_dropped_by_cutoff"] == 1
    assert profile.temp[0, 0] == 10.0
    assert np.isnan(profile.temp[1, 0])
    assert profile.depth_qc1[:, 1].tolist() == [True, False]
    assert profile.depth[1, 1] == 10.0  # QC1 series median fallback, not raw bad depth.
    assert [item["target_layer"] for item in profile.mapping] == [1, 2, 3]


def test_loo_features_mask_target_temperature(tmp_path: Path) -> None:
    path, manifest = _archive(tmp_path)
    profile = read_year_profile(
        path,
        manifest,
        year=2022,
        target_depth_by_layer={1: 5.0, 2: 10.0, 3: 15.0},
        max_mapping_distance_m=2.0,
    )

    dataset = build_loo_dataset([profile], min_peer_temperatures=2, max_rows_per_year_layer=None)

    target_rows = dataset.layer == 2
    target_peer_column = dataset.feature_names.index("peer_temp_layer_2")
    assert target_rows.sum() == 2
    assert np.isnan(dataset.x[target_rows, target_peer_column]).all()
    assert np.isfinite(dataset.y[target_rows]).all()
    assert set(dataset.group_counts) == {
        "2022:layer_1",
        "2022:layer_2",
        "2022:layer_3",
    }


def test_depth_linear_baseline_uses_masked_neighbors() -> None:
    profile = YearProfile(
        year=2023,
        time_utc=np.asarray(["2023-01-01T00:00:00"], dtype="datetime64[s]"),
        target_layers=np.asarray([1, 2, 3], dtype=np.int16),
        target_depths=np.asarray([5.0, 10.0, 15.0]),
        temp=np.asarray([[10.0, 999.0, 20.0]]),
        psal=np.full((1, 3), np.nan),
        depth=np.asarray([[5.0, 10.0, 15.0]]),
        depth_qc1=np.ones((1, 3), dtype=bool),
        mapping=(),
        audit={},
    )

    prediction = depth_linear_baseline(profile, 1)

    assert prediction.tolist() == [15.0]


def test_aggregate_metrics_and_gate_are_preregistered_only() -> None:
    dataset = type(
        "Fixture",
        (),
        {
            "y": np.asarray([1.0, 2.0, 3.0, 4.0]),
            "baseline": np.asarray([0.0, 3.0, 2.0, 5.0]),
            "layer": np.asarray([1, 1, 2, 2]),
        },
    )()
    predictions = {
        0.1: np.asarray([0.8, 1.8, 2.8, 3.8]),
        0.5: np.asarray([1.0, 2.0, 3.0, 4.0]),
        0.9: np.asarray([1.2, 2.2, 3.2, 4.2]),
    }
    metrics = evaluate_precheck(dataset, predictions)
    gate = {
        "minimum_holdout_rows": 4,
        "minimum_evaluated_layers": 2,
        "minimum_rmse_relative_improvement": 0.05,
        "minimum_mae_relative_improvement": 0.05,
        "minimum_fraction_layers_not_worse": 1.0,
        "maximum_single_layer_relative_rmse_degradation": 0.0,
        "q10_q90_coverage_min": 0.70,
        "q10_q90_coverage_max": 1.0,
        "all_source_integrity_checks_required": True,
    }

    decision = apply_stop_gate(metrics, gate, source_integrity_verified=True)

    assert decision["passed"] is True
    assert decision["decision"] == "GO_TO_ISOLATED_P1_OOF_ONLY"
    assert "not P1 OOF" in decision["scope"]


def test_external_runner_cannot_open_p1_training_or_outer_labels() -> None:
    paths = [
        Path("scripts/run_p1_iors_external_precheck.py"),
        Path("src/ocean_external/iors_ctd.py"),
        Path("src/ocean_external/iors_precheck.py"),
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "from p1_qc" not in source
    assert "train.csv" not in source
    assert "P1_DATA_DIR" not in source
    assert "load_train_test" not in source
    assert "run_cross_validation" not in source
