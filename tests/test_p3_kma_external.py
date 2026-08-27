from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave.kma_external import (
    CURRENT_RESPONSE_COLUMNS,
    DEFAULT_STATION_EPOCHS,
    EXPECTED_COLUMNS,
    KST,
    SOURCE_CUTOFF,
    KMAClient,
    KMACutoffError,
    KMAExternalError,
    KMAPrecheckError,
    KMASchemaError,
    KMAStationChangeError,
    StationEpoch,
    assert_credential_absent,
    build_independent_storm_anchors,
    iter_month_windows,
    parse_kma_buoy_payload,
    prepare_kma_external,
    run_source_precheck,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _row(
    timestamp: str,
    *,
    station_id: int = 22107,
    overrides: dict[str, object] | None = None,
) -> str:
    values: dict[str, object] = {
        "TM": timestamp,
        "STN": station_id,
        "WD1": 180,
        "WS1": 8,
        "WS1_GST": 11,
        "WD2": 185,
        "WS2": 7,
        "WS2_GST": 10,
        "PA": 1008,
        "HM": 75,
        "TA": 16,
        "TW": 20,
        "WH_MAX": 3.0,
        "WH_SIG": 2.0,
        "WH_AVE": 1.3,
        "WP": 8.0,
        "WO": 210,
    }
    values.update(overrides or {})
    return " ".join(str(values[column]) for column in EXPECTED_COLUMNS)


def _payload(*rows: str) -> bytes:
    content = ["# KMA buoy fixture", " ".join(EXPECTED_COLUMNS), *rows, "7777END"]
    return ("\n".join(content) + "\n").encode()


def _current_csv_payload(
    timestamp: str,
    *,
    aqc: str = "000000000000000",
    mqc: str = "---------------",
    terminator: str | None = "=",
) -> bytes:
    legacy_values = _row(timestamp).split()
    values = [*legacy_values, aqc, mqc]
    if terminator is not None:
        values.append(terminator)
    content = ["#START7777", ",".join(values), "#7777END"]
    return ("\n".join(content) + "\n").encode()


def _comment_only_payload() -> bytes:
    return b"#START7777\n# no observations for this request\n#7777END\n"


def test_parser_preserves_native_schema_target_and_cadence() -> None:
    frame = parse_kma_buoy_payload(
        _payload(_row("202312010000"), _row("202312010030")),
        expected_station_id=22107,
    )

    assert len(frame) == 2
    assert frame["WH_SIG"].tolist() == [2.0, 2.0]
    assert frame["TM"].dt.tz is not None
    assert frame["TM"].diff().iloc[1] == pd.Timedelta(minutes=30)
    assert frame["proxy_group"].eq("I_proxy").all()


def test_parser_accepts_current_comma_schema_and_retains_qc_vectors() -> None:
    frame = parse_kma_buoy_payload(
        _current_csv_payload("202312010000", mqc="0000000000000000"),
        expected_station_id=22107,
    )

    assert list(frame.columns[: len(CURRENT_RESPONSE_COLUMNS)]) == list(CURRENT_RESPONSE_COLUMNS)
    assert frame.loc[0, "WH_SIG"] == 2.0
    assert frame.loc[0, "AQC"] == "000000000000000"
    assert frame.loc[0, "MQC"] == "0000000000000000"


def test_parser_accepts_exact_live_aqc_na_vector() -> None:
    aqc_na = "NA" * 15
    frame = parse_kma_buoy_payload(
        _current_csv_payload("202312010000", aqc=aqc_na),
        expected_station_id=22107,
    )

    assert frame.loc[0, "AQC"] == aqc_na


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_current_csv_payload("202312010000", terminator="END"), "record terminator"),
        (_current_csv_payload("202312010000", aqc="000"), "AQC violates"),
        (_current_csv_payload("202312010000", aqc="(NA)" * 15), "AQC violates"),
        (_current_csv_payload("202312010000", aqc="00000000000000-"), "AQC violates"),
        (_current_csv_payload("202312010000", aqc="0000000000000000"), "AQC violates"),
        (_current_csv_payload("202312010000", mqc="bad quality flag"), "MQC violates"),
        (_current_csv_payload("202312010000", mqc="000000000000000-"), "MQC violates"),
        (_current_csv_payload("202312010000", mqc="000000000000000"), "MQC violates"),
    ],
)
def test_current_comma_schema_rejects_malformed_metadata(payload: bytes, message: str) -> None:
    with pytest.raises(KMASchemaError, match=message):
        parse_kma_buoy_payload(payload, expected_station_id=22107)


def test_exact_minus_99_sentinel_becomes_null() -> None:
    frame = parse_kma_buoy_payload(
        _payload(_row("202312010000", overrides={"WH_SIG": -99})),
        expected_station_id=22107,
    )
    assert np.isnan(frame.loc[0, "WH_SIG"])
    assert frame.loc[0, "quality_quarantined"] == np.False_
    assert frame.loc[0, "quality_provenance"] == "sentinel_minus_99:WH_SIG"
    assert frame.attrs["quality_summary"]["sentinel_minus_99_cell_count"] == 1


def test_out_of_range_negative_is_quarantined_not_treated_as_a_sentinel() -> None:
    frame = parse_kma_buoy_payload(
        _payload(_row("202312010000", overrides={"WH_SIG": -999})),
        expected_station_id=22107,
    )

    assert np.isnan(frame.loc[0, "WH_SIG"])
    assert frame.loc[0, "quality_quarantined"] == np.True_
    assert frame.loc[0, "quality_provenance"] == "range:WH_SIG"
    summary = frame.attrs["quality_summary"]
    assert summary["sentinel_minus_99_cell_count"] == 0
    assert summary["range_quarantine_counts_by_column"]["WH_SIG"] == 1


def test_negative_nine_air_temperature_is_preserved() -> None:
    frame = parse_kma_buoy_payload(
        _payload(_row("202312010000", overrides={"TA": -9})),
        expected_station_id=22107,
    )

    assert frame.loc[0, "TA"] == -9.0


def test_physical_range_failure_quarantines_only_the_bad_cell() -> None:
    frame = parse_kma_buoy_payload(
        _payload(_row("202312010000", overrides={"WP": 41})),
        expected_station_id=22107,
    )

    assert np.isnan(frame.loc[0, "WP"])
    assert frame.loc[0, "WH_SIG"] == 2.0
    assert frame.loc[0, "quality_provenance"] == "range:WP"
    summary = frame.attrs["quality_summary"]
    assert summary["range_quarantine_cell_count"] == 1
    assert summary["range_quarantine_counts_by_column"]["WP"] == 1
    assert summary["quarantined_row_count"] == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"WH_SIG": 4, "WH_MAX": 3},
        {"WH_AVE": 2.5},
    ],
)
def test_wave_order_contradiction_quarantines_the_triplet(
    overrides: dict[str, object],
) -> None:
    frame = parse_kma_buoy_payload(
        _payload(_row("202312010000", overrides=overrides)),
        expected_station_id=22107,
    )

    assert frame.loc[0, ["WH_MAX", "WH_SIG", "WH_AVE"]].isna().all()
    assert frame.loc[0, "quality_quarantined"] == np.True_
    assert frame.loc[0, "quality_provenance"] == "wave_order:triplet"
    assert frame.attrs["quality_summary"]["wave_order_quarantine_row_count"] == 1


def test_off_grid_and_non_schema_rows_fail_closed() -> None:
    with pytest.raises(KMASchemaError, match="native 00/30-minute"):
        parse_kma_buoy_payload(_payload(_row("202312010010")), expected_station_id=22107)
    with pytest.raises(KMASchemaError, match="expected 17 fields"):
        parse_kma_buoy_payload(b"202312010000 22107 1\n", expected_station_id=22107)


def test_cutoff_and_station_identity_are_enforced() -> None:
    with pytest.raises(KMACutoffError, match="after 2023"):
        parse_kma_buoy_payload(_payload(_row("202401010000")), expected_station_id=22107)
    with pytest.raises(KMAStationChangeError, match="different station"):
        parse_kma_buoy_payload(
            _payload(_row("202312010000", station_id=22102)),
            expected_station_id=22107,
        )
    with pytest.raises(KMAStationChangeError, match="known reused"):
        parse_kma_buoy_payload(
            _payload(_row("202312010000", station_id=22193)),
            expected_station_id=22193,
        )


def test_custom_station_epoch_cannot_be_crossed() -> None:
    epochs = {
        22107: StationEpoch(
            22107,
            "Marado",
            "I_proxy",
            datetime(2023, 1, 1, tzinfo=KST),
            datetime(2023, 11, 30, 23, 30, tzinfo=KST),
        )
    }
    with pytest.raises(KMAStationChangeError, match="identity epoch"):
        parse_kma_buoy_payload(
            _payload(_row("202312010000")),
            expected_station_id=22107,
            station_epochs=epochs,
        )


def test_client_receipt_and_repr_never_contain_credential() -> None:
    secret = "very-secret-fixture-key"

    def transport(url: str, timeout: float) -> bytes:
        assert timeout > 0
        assert secret in url
        return _payload(_row("202312010000"), _row("202312010030"))

    client = KMAClient(secret, transport=transport)
    frame, receipt = client.fetch_period(
        DEFAULT_STATION_EPOCHS[22107],
        datetime(2023, 12, 1, 0, 0, tzinfo=KST),
        datetime(2023, 12, 1, 0, 30, tzinfo=KST),
    )

    assert len(frame) == 2
    assert secret not in repr(client)
    serialized = json.dumps(receipt.__dict__, sort_keys=True)
    assert secret not in serialized
    assert_credential_absent(receipt.__dict__, secret)


def test_client_safely_returns_zero_row_receipt_for_comment_only_month() -> None:
    secret = "very-secret-fixture-key"

    client = KMAClient(secret, transport=lambda url, timeout: _comment_only_payload())
    frame, receipt = client.fetch_period(
        DEFAULT_STATION_EPOCHS[22107],
        datetime(2023, 12, 1, 0, 0, tzinfo=KST),
        datetime(2023, 12, 1, 0, 30, tzinfo=KST),
    )

    assert frame.empty
    assert receipt.row_count == 0
    assert receipt.observed_start is None
    assert receipt.observed_end is None
    assert receipt.quality_summary["quarantined_row_count"] == 0


def test_empty_payload_requires_both_api_markers_and_explicit_opt_in() -> None:
    with pytest.raises(KMASchemaError, match="no buoy observations"):
        parse_kma_buoy_payload(_comment_only_payload(), expected_station_id=22107)
    with pytest.raises(KMASchemaError, match="no buoy observations"):
        parse_kma_buoy_payload(
            b"# arbitrary empty response\n",
            expected_station_id=22107,
            allow_empty=True,
        )


def test_transport_error_is_sanitized() -> None:
    secret = "very-secret-fixture-key"

    def transport(url: str, timeout: float) -> bytes:
        raise RuntimeError(f"bad url {url} at {timeout}")

    client = KMAClient(secret, transport=transport)
    with pytest.raises(KMAExternalError, match="KMA transport failed") as caught:
        client.fetch_period(
            DEFAULT_STATION_EPOCHS[22107],
            datetime(2023, 12, 1, 0, 0, tzinfo=KST),
            datetime(2023, 12, 1, 0, 30, tzinfo=KST),
        )
    assert secret not in str(caught.value)


def test_missing_credential_performs_zero_network_requests(tmp_path: Path) -> None:
    calls = 0

    def forbidden_transport(url: str, timeout: float) -> bytes:
        nonlocal calls
        calls += 1
        raise AssertionError(f"network should not be called: {url}, {timeout}")

    result = prepare_kma_external(
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        mode="full",
        environment={},
        transport=forbidden_transport,
    )

    assert calls == 0
    assert result["status"] == "awaiting_credential"
    assert result["safety_invariants"]["model_trained"] is False
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "retrieval_manifest.json").read_text(encoding="utf-8"))
    assert status == manifest == result
    assert not list(tmp_path.glob("*.parquet"))


def test_status_mode_with_credential_also_performs_zero_requests(tmp_path: Path) -> None:
    secret = "very-secret-fixture-key"

    def forbidden_transport(url: str, timeout: float) -> bytes:
        raise AssertionError(f"network should not be called: {url}, {timeout}")

    result = prepare_kma_external(
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        mode="status",
        environment={"KMA_API_KEY": secret},
        transport=forbidden_transport,
    )
    assert result["status"] == "credential_ready"
    assert result["network_request_count"] == 0
    assert secret not in (tmp_path / "status.json").read_text(encoding="utf-8")


def test_smoke_mode_writes_only_safe_aggregate_receipts(tmp_path: Path) -> None:
    secret = "very-secret-fixture-key"

    def transport(url: str, timeout: float) -> bytes:
        assert secret in url
        assert timeout > 0
        return _payload(
            _row("202312010000", overrides={"WP": 41}),
            _row("202312010030", overrides={"WH_MAX": 1, "WH_SIG": 2}),
        )

    result = prepare_kma_external(
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        mode="smoke",
        environment={"KMA_API_KEY": secret},
        transport=transport,
        station_ids=[22107],
        start=datetime(2023, 12, 1, 0, 0, tzinfo=KST),
        end=datetime(2023, 12, 1, 6, 0, tzinfo=KST),
    )

    assert result["status"] == "smoke_complete"
    assert result["network_request_count"] == 1
    assert result["row_count"] == 2
    assert result["empty_response_count"] == 0
    assert result["quality_summary"]["range_quarantine_counts_by_column"]["WP"] == 1
    assert result["quality_summary"]["wave_order_quarantine_row_count"] == 1
    assert result["quality_summary"]["quarantined_row_count"] == 2
    assert result["response_receipts"][0]["quality_summary"] == result["quality_summary"]
    assert not list(tmp_path.glob("*.parquet"))
    for receipt_path in (tmp_path / "status.json", tmp_path / "retrieval_manifest.json"):
        assert secret not in receipt_path.read_text(encoding="utf-8")


def test_smoke_mode_records_comment_only_window_without_aborting(tmp_path: Path) -> None:
    result = prepare_kma_external(
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        mode="smoke",
        environment={"KMA_API_KEY": "very-secret-fixture-key"},
        transport=lambda url, timeout: _comment_only_payload(),
        station_ids=[22107],
        start=datetime(2023, 12, 1, 0, 0, tzinfo=KST),
        end=datetime(2023, 12, 1, 6, 0, tzinfo=KST),
    )

    assert result["status"] == "smoke_no_data"
    assert result["row_count"] == 0
    assert result["empty_response_count"] == 1
    assert result["observed_start"] is None
    assert result["observed_end"] is None
    assert result["response_receipts"][0]["row_count"] == 0


def test_full_mode_quarantines_but_rejects_insufficient_source(tmp_path: Path) -> None:
    secret = "very-secret-fixture-key"

    def transport(url: str, timeout: float) -> bytes:
        assert secret in url
        assert timeout > 0
        return _payload(
            _row("202312010000", overrides={"PA": 1200}),
            _row("202312010030", overrides={"WH_MAX": 1, "WH_SIG": 2}),
        )

    with pytest.raises(KMAPrecheckError, match="not trainable"):
        prepare_kma_external(
            repo_root=REPO_ROOT,
            output_dir=tmp_path,
            mode="full",
            environment={"KMA_API_KEY": secret},
            transport=transport,
            station_ids=[22107],
            start=datetime(2023, 12, 1, 0, 0, tzinfo=KST),
            end=datetime(2023, 12, 1, 0, 30, tzinfo=KST),
        )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert manifest["precheck"]["accepted"] is False
    assert manifest["local_file"] == "kma_buoy_pre2024.parquet"
    assert manifest["anchors_file"] == "storm_anchors.parquet"
    assert not Path(manifest["local_file"]).is_absolute()
    assert not Path(manifest["anchors_file"]).is_absolute()
    assert status["status"] == "source_precheck_failed_closed"
    assert manifest["quality_summary"]["range_quarantine_counts_by_column"]["PA"] == 1
    assert manifest["quality_summary"]["wave_order_quarantine_row_count"] == 1
    assert manifest["precheck"]["quality_summary"] == manifest["quality_summary"]
    assert status["quality_summary"] == manifest["quality_summary"]
    candidate = pd.read_parquet(tmp_path / "kma_buoy_pre2024.parquet")
    assert {"quality_quarantined", "quality_provenance"} <= set(candidate.columns)
    assert candidate["quality_quarantined"].all()
    assert secret not in json.dumps(manifest, sort_keys=True)
    assert secret not in json.dumps(status, sort_keys=True)


def test_full_mode_empty_source_reaches_overall_no_data_gate(tmp_path: Path) -> None:
    with pytest.raises(KMAPrecheckError, match="not trainable"):
        prepare_kma_external(
            repo_root=REPO_ROOT,
            output_dir=tmp_path,
            mode="full",
            environment={"KMA_API_KEY": "very-secret-fixture-key"},
            transport=lambda url, timeout: _comment_only_payload(),
            station_ids=[22107],
            start=datetime(2023, 12, 1, 0, 0, tzinfo=KST),
            end=datetime(2023, 12, 1, 0, 30, tzinfo=KST),
        )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert manifest["row_count"] == 0
    assert manifest["empty_response_count"] == 1
    assert "no observations" in manifest["precheck"]["reasons"]
    assert manifest["precheck"]["quality_summary"]["quarantined_row_count"] == 0
    assert status["status"] == "source_precheck_failed_closed"


def test_78_hour_independent_storm_anchor_gate() -> None:
    times = pd.date_range(
        "2023-01-01T00:00:00+09:00",
        periods=48 * 22,
        freq="30min",
    )
    hs = np.ones(len(times))
    high_positions = [96, 252, 408, 564, 720, 876]
    hs[high_positions] = 2.0
    frame = pd.DataFrame(
        {
            "TM": times,
            "STN": 22107,
            "WH_SIG": hs,
            "proxy_group": "I_proxy",
        }
    )

    anchors = build_independent_storm_anchors(frame)

    assert anchors["anchor_time_kst"].tolist() == [times[index] for index in high_positions]
    assert anchors["anchor_time_kst"].diff().dropna().min() == pd.Timedelta(hours=78)


def test_anchor_requires_all_six_future_leads() -> None:
    times = pd.date_range(
        "2023-01-01T00:00:00+09:00",
        periods=48 * 5,
        freq="30min",
    )
    hs = np.ones(len(times))
    hs[96] = 2.0
    hs[96 + 48] = np.nan
    frame = pd.DataFrame(
        {
            "TM": times,
            "STN": 22107,
            "WH_SIG": hs,
            "proxy_group": "I_proxy",
        }
    )
    anchors = build_independent_storm_anchors(frame)
    assert anchors.empty


def test_source_precheck_fails_closed_on_insufficient_domain_support() -> None:
    frame = parse_kma_buoy_payload(
        _payload(_row("202312010000"), _row("202312010030")),
        expected_station_id=22107,
    )
    anchors = pd.DataFrame(columns=["station_id", "station_name", "proxy_group", "anchor_time_kst"])
    report = run_source_precheck(frame, anchors)

    assert report["accepted"] is False
    assert report["domain_shift_local_comparison"] == "pending"
    assert "fewer than 4 stable stations" in report["reasons"]
    assert report["quality_summary"]["range_quarantine_cell_count"] == 0
    with pytest.raises(KMAPrecheckError):
        if not report["accepted"]:
            raise KMAPrecheckError("fail closed")


def test_month_windows_are_disjoint_and_never_cross_cutoff() -> None:
    start = datetime(2023, 11, 30, 23, 30, tzinfo=KST)
    end = datetime(2023, 12, 31, 23, 30, tzinfo=KST)
    windows = iter_month_windows(start, end)

    assert windows[0][0] == start
    assert windows[-1][1] == end
    assert windows[-1][1] <= SOURCE_CUTOFF
    assert all(
        right[0] - left[1] == timedelta(minutes=30)
        for left, right in zip(windows, windows[1:], strict=False)
    )
