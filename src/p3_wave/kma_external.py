"""Credential-safe KMA buoy preparation for P3 external pretraining.

This module is deliberately isolated from the frozen P3 model, OOF predictions,
test contexts, and submission code.  It only retrieves pre-2024 KMA buoy
observations into the ignored ``external_data/`` quarantine and performs source
quality checks.  A separate, explicit domain-shift gate is still required before
any training use.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ocean_external.policy import ApprovalReceipt, PolicyError, load_catalog

KST = ZoneInfo("Asia/Seoul")
SOURCE_ID = "kma_ocean_buoy_pre2024"
TARGET_COLUMN = "WH_SIG"
SOURCE_FLOOR = datetime(2016, 3, 1, 0, 0, tzinfo=KST)
SOURCE_CUTOFF = datetime(2023, 12, 31, 23, 59, 59, tzinfo=KST)
LAST_NATIVE_OBSERVATION = datetime(2023, 12, 31, 23, 30, tzinfo=KST)
NATIVE_CADENCE_MINUTES = 30
LEAD_HOURS = (3, 6, 9, 12, 18, 24)
KMA_PERIOD_ENDPOINT = "https://apihub.kma.go.kr/api/typ01/url/kma_buoy2.php"
KMA_API_DOCUMENTATION = "https://apihub.kma.go.kr/apiList.do?seqApi=3"
KMA_API_GUIDE = "https://apihub.kma.go.kr/apiInfo.do"
KMA_DATA_PORTAL = "https://data.kma.go.kr/data/sea/selectBuoyRltmList.do?pgmNo=52&tabNo=1"

EXPECTED_COLUMNS = (
    "TM",
    "STN",
    "WD1",
    "WS1",
    "WS1_GST",
    "WD2",
    "WS2",
    "WS2_GST",
    "PA",
    "HM",
    "TA",
    "TW",
    "WH_MAX",
    "WH_SIG",
    "WH_AVE",
    "WP",
    "WO",
)
# The current ``kma_buoy2.php`` wire format appends AQC/MQC vectors and an
# ``=`` record marker to the 17 observation fields documented on the API Hub
# page. Keep ``EXPECTED_COLUMNS`` as the stable modelling schema while
# retaining the QC metadata when it is present.
QUALITY_COLUMNS = ("AQC", "MQC")
CURRENT_RESPONSE_COLUMNS = EXPECTED_COLUMNS + QUALITY_COLUMNS
QUALITY_FLAG_PATTERNS = {
    "AQC": re.compile(r"(?:[0-9]{15}|-{15}|(?:NA){15})"),
    "MQC": re.compile(r"(?:[0-9]{16}|-{15})"),
}
ROW_QUALITY_COLUMNS = ("quality_quarantined", "quality_provenance")
NUMERIC_COLUMNS = EXPECTED_COLUMNS[2:]
VARIABLE_UNITS = {
    "WD1": "degree_true",
    "WS1": "m s-1",
    "WS1_GST": "m s-1",
    "WD2": "degree_true",
    "WS2": "m s-1",
    "WS2_GST": "m s-1",
    "PA": "hPa",
    "HM": "%",
    "TA": "degree_C",
    "TW": "degree_C",
    "WH_MAX": "m",
    "WH_SIG": "m",
    "WH_AVE": "m",
    "WP": "s",
    "WO": "degree_true",
}
NUMERIC_BOUNDS = {
    "WD1": (0.0, 360.0),
    "WS1": (0.0, 100.0),
    "WS1_GST": (0.0, 100.0),
    "WD2": (0.0, 360.0),
    "WS2": (0.0, 100.0),
    "WS2_GST": (0.0, 100.0),
    "PA": (800.0, 1100.0),
    "HM": (0.0, 100.0),
    "TA": (-60.0, 60.0),
    "TW": (-5.0, 45.0),
    "WH_MAX": (0.0, 30.0),
    "WH_SIG": (0.0, 30.0),
    "WH_AVE": (0.0, 30.0),
    "WP": (0.0, 40.0),
    "WO": (0.0, 360.0),
}
WAVE_HEIGHT_COLUMNS = ("WH_MAX", "WH_SIG", "WH_AVE")
MISSING_SENTINELS = frozenset({-99.0})
KNOWN_REUSED_STATION_IDS = frozenset({22193})


class KMAExternalError(RuntimeError):
    """Base error for the quarantined KMA preparation path."""


class KMASchemaError(KMAExternalError):
    """Raised when a response violates the declared KMA schema or units."""


class KMAStationChangeError(KMAExternalError):
    """Raised when station identity cannot be treated as stable."""


class KMACutoffError(KMAExternalError):
    """Raised when a request or observation crosses the pre-2024 boundary."""


class KMAPrecheckError(KMAExternalError):
    """Raised when source observations fail a fail-closed precheck."""


@dataclass(frozen=True)
class StationEpoch:
    station_id: int
    name: str
    proxy_group: str
    valid_from: datetime
    valid_to: datetime

    def __post_init__(self) -> None:
        if self.valid_from.tzinfo is None or self.valid_to.tzinfo is None:
            raise ValueError("station epoch boundaries must be timezone aware")
        if self.valid_to < self.valid_from:
            raise ValueError("station epoch valid_to precedes valid_from")


# Initial extraction is intentionally restricted to long-running station IDs.
# 22193 is excluded because KMA documents that it represented Gageodo in
# 2019-2020 and was later reused for Seohae143.
DEFAULT_STATION_EPOCHS: Mapping[int, StationEpoch] = {
    22101: StationEpoch(22101, "Deokjeokdo", "S_proxy", SOURCE_FLOOR, SOURCE_CUTOFF),
    22102: StationEpoch(22102, "Chilbaldo", "G_proxy", SOURCE_FLOOR, SOURCE_CUTOFF),
    22107: StationEpoch(22107, "Marado", "I_proxy", SOURCE_FLOOR, SOURCE_CUTOFF),
    22185: StationEpoch(22185, "Incheon", "S_proxy", SOURCE_FLOOR, SOURCE_CUTOFF),
}


@dataclass(frozen=True)
class FetchReceipt:
    station_id: int
    requested_start: str
    requested_end: str
    response_sha256: str
    row_count: int
    observed_start: str | None
    observed_end: str | None
    quality_summary: dict[str, Any]


Transport = Callable[[str, float], bytes]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _as_kst(value: datetime | pd.Timestamp, *, field: str) -> datetime:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise KMAExternalError(f"{field} must include a timezone")
    return stamp.tz_convert(KST).to_pydatetime()


def _validate_native_time(value: datetime, *, field: str) -> None:
    if value.second != 0 or value.microsecond != 0 or value.minute not in {0, 30}:
        raise KMAExternalError(f"{field} must lie on the native 00/30-minute grid")


def validate_request_window(
    station: StationEpoch,
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
) -> tuple[datetime, datetime]:
    start_kst = _as_kst(start, field="start")
    end_kst = _as_kst(end, field="end")
    _validate_native_time(start_kst, field="start")
    _validate_native_time(end_kst, field="end")
    if end_kst < start_kst:
        raise KMAExternalError("request end precedes start")
    if end_kst > SOURCE_CUTOFF:
        raise KMACutoffError("request exceeds the 2023-12-31 KST source cutoff")
    if start_kst < station.valid_from or end_kst > station.valid_to:
        raise KMAStationChangeError("request crosses the registered station identity epoch")
    return start_kst, end_kst


def _decode_payload(payload: bytes | str) -> str:
    if isinstance(payload, str):
        return payload
    for encoding in ("utf-8-sig", "euc-kr"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise KMASchemaError("KMA response is neither UTF-8 nor EUC-KR text")


def _reject_error_payload(text: str) -> None:
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise KMASchemaError("KMA returned malformed JSON instead of buoy rows") from exc
        status = parsed.get("status", "unknown") if isinstance(parsed, dict) else "unknown"
        raise KMAExternalError(f"KMA API returned status {status}")
    if stripped.startswith("<"):
        raise KMAExternalError("KMA API returned XML/HTML instead of buoy rows")


def _parse_observation_tokens(line: str) -> list[str | None] | None:
    """Parse legacy whitespace or current comma-delimited buoy rows."""

    if "," not in line:
        tokens = re.split(r"\s+", line)
        if tuple(tokens) == EXPECTED_COLUMNS:
            return None
        if len(tokens) != len(EXPECTED_COLUMNS):
            raise KMASchemaError(f"expected {len(EXPECTED_COLUMNS)} fields, received {len(tokens)}")
        return [*tokens, None, None]

    tokens = [token.strip() for token in next(csv.reader([line], skipinitialspace=True))]
    if tuple(tokens) == CURRENT_RESPONSE_COLUMNS:
        return None
    if len(tokens) == len(CURRENT_RESPONSE_COLUMNS) + 1:
        terminator = tokens.pop()
        if terminator != "=":
            raise KMASchemaError("current comma-delimited row has an invalid record terminator")
    elif len(tokens) != len(CURRENT_RESPONSE_COLUMNS):
        raise KMASchemaError(
            "expected 19 current comma-delimited fields plus an optional '=' terminator, "
            f"received {len(tokens)} fields"
        )
    for column, value in zip(QUALITY_COLUMNS, tokens[-2:], strict=True):
        if QUALITY_FLAG_PATTERNS[column].fullmatch(value) is None:
            raise KMASchemaError(f"{column} violates its declared QC flag format")
    return tokens


def _empty_quality_summary() -> dict[str, Any]:
    return {
        "sentinel_minus_99_cell_count": 0,
        "sentinel_minus_99_counts_by_column": {column: 0 for column in NUMERIC_COLUMNS},
        "range_quarantine_cell_count": 0,
        "range_quarantine_counts_by_column": {column: 0 for column in NUMERIC_COLUMNS},
        "wave_order_quarantine_row_count": 0,
        "quarantined_row_count": 0,
    }


def _aggregate_quality_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = _empty_quality_summary()
    for summary in summaries:
        for scalar in (
            "sentinel_minus_99_cell_count",
            "range_quarantine_cell_count",
            "wave_order_quarantine_row_count",
            "quarantined_row_count",
        ):
            aggregate[scalar] += int(summary.get(scalar, 0))
        for group in (
            "sentinel_minus_99_counts_by_column",
            "range_quarantine_counts_by_column",
        ):
            source = summary.get(group, {})
            if not isinstance(source, Mapping):
                raise KMASchemaError(f"{group} is not an aggregate mapping")
            for column in NUMERIC_COLUMNS:
                aggregate[group][column] += int(source.get(column, 0))
    return aggregate


def _append_quality_event(events: list[list[str]], mask: pd.Series, event: str) -> None:
    for position in np.flatnonzero(mask.to_numpy(dtype=bool)):
        events[int(position)].append(event)


def _quarantine_numeric_quality(frame: pd.DataFrame) -> dict[str, Any]:
    """Mask isolated bad cells while preserving aggregate and row provenance."""

    summary = _empty_quality_summary()
    events: list[list[str]] = [[] for _ in range(len(frame))]
    quarantined = pd.Series(False, index=frame.index, dtype=bool)

    for column in NUMERIC_COLUMNS:
        values = frame[column]
        sentinel = values.eq(-99.0)
        sentinel_count = int(sentinel.sum())
        summary["sentinel_minus_99_counts_by_column"][column] = sentinel_count
        summary["sentinel_minus_99_cell_count"] += sentinel_count
        if sentinel_count:
            _append_quality_event(events, sentinel, f"sentinel_minus_99:{column}")
            frame.loc[sentinel, column] = np.nan

        lower, upper = NUMERIC_BOUNDS[column]
        out_of_range = frame[column].notna() & (frame[column].lt(lower) | frame[column].gt(upper))
        out_of_range_count = int(out_of_range.sum())
        summary["range_quarantine_counts_by_column"][column] = out_of_range_count
        summary["range_quarantine_cell_count"] += out_of_range_count
        if out_of_range_count:
            _append_quality_event(events, out_of_range, f"range:{column}")
            quarantined |= out_of_range
            frame.loc[out_of_range, column] = np.nan

    max_below_sig = (
        frame["WH_MAX"].notna()
        & frame["WH_SIG"].notna()
        & frame["WH_MAX"].add(0.05).lt(frame["WH_SIG"])
    )
    sig_below_average = (
        frame["WH_SIG"].notna()
        & frame["WH_AVE"].notna()
        & frame["WH_SIG"].add(0.05).lt(frame["WH_AVE"])
    )
    wave_order = max_below_sig | sig_below_average
    wave_order_count = int(wave_order.sum())
    summary["wave_order_quarantine_row_count"] = wave_order_count
    if wave_order_count:
        _append_quality_event(events, wave_order, "wave_order:triplet")
        quarantined |= wave_order
        frame.loc[wave_order, list(WAVE_HEIGHT_COLUMNS)] = np.nan

    summary["quarantined_row_count"] = int(quarantined.sum())
    frame["quality_quarantined"] = quarantined.to_numpy(dtype=bool)
    frame["quality_provenance"] = ["|".join(row_events) or "source" for row_events in events]
    return summary


def _empty_parsed_frame(station: StationEpoch) -> pd.DataFrame:
    frame = pd.DataFrame(columns=(*CURRENT_RESPONSE_COLUMNS, *ROW_QUALITY_COLUMNS))
    frame["TM"] = pd.Series(dtype=pd.DatetimeTZDtype(tz=KST))
    frame["STN"] = pd.Series(dtype="int64")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.Series(dtype="float64")
    frame["quality_quarantined"] = pd.Series(dtype=bool)
    frame["station_name"] = pd.Series(dtype="object")
    frame["proxy_group"] = pd.Series(dtype="object")
    frame.attrs["quality_summary"] = _empty_quality_summary()
    frame.attrs["empty_station_name"] = station.name
    return frame


def parse_kma_buoy_payload(
    payload: bytes | str,
    *,
    expected_station_id: int,
    station_epochs: Mapping[int, StationEpoch] = DEFAULT_STATION_EPOCHS,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Parse one KMA period response and fail closed on metadata ambiguity.

    ``allow_empty`` accepts only a marker-delimited, comment-only API response;
    arbitrary blank payloads remain schema errors.
    """

    if expected_station_id in KNOWN_REUSED_STATION_IDS:
        raise KMAStationChangeError("known reused station ID is prohibited")
    station = station_epochs.get(expected_station_id)
    if station is None:
        raise KMAStationChangeError("station ID is not in the stable epoch registry")

    text = _decode_payload(payload)
    _reject_error_payload(text)
    records: list[list[str | None]] = []
    markers: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        marker = line.removeprefix("#").strip()
        if marker in {"START7777", "7777END"}:
            markers.add(marker)
            continue
        if line.startswith("#"):
            continue
        tokens = _parse_observation_tokens(line)
        if tokens is not None:
            records.append(tokens)
    if not records:
        if allow_empty and markers == {"START7777", "7777END"}:
            return _empty_parsed_frame(station)
        raise KMASchemaError("KMA response contains no buoy observations")

    frame = pd.DataFrame(records, columns=CURRENT_RESPONSE_COLUMNS)
    if not frame["TM"].astype(str).str.fullmatch(r"\d{12}").all():
        raise KMASchemaError("TM must use YYYYMMDDHHMM")
    try:
        frame["TM"] = pd.to_datetime(frame["TM"], format="%Y%m%d%H%M").dt.tz_localize(KST)
        frame["STN"] = pd.to_numeric(frame["STN"], errors="raise").astype("int64")
    except (TypeError, ValueError) as exc:
        raise KMASchemaError("TM or STN cannot be parsed") from exc
    if not frame["STN"].eq(expected_station_id).all():
        raise KMAStationChangeError("response contains a different station ID")

    for column in NUMERIC_COLUMNS:
        try:
            values = pd.to_numeric(frame[column], errors="raise").astype("float64")
        except (TypeError, ValueError) as exc:
            raise KMASchemaError(f"{column} is not numeric") from exc
        frame[column] = values

    if not frame["TM"].is_monotonic_increasing:
        raise KMASchemaError("KMA rows are not in increasing timestamp order")
    if frame["TM"].duplicated().any():
        raise KMASchemaError("KMA response contains duplicate timestamps")
    if not frame["TM"].dt.minute.isin([0, 30]).all() or not frame["TM"].dt.second.eq(0).all():
        raise KMASchemaError("observations are not on the native 00/30-minute grid")
    deltas = frame["TM"].diff().dropna().dt.total_seconds().div(60)
    if (deltas <= 0).any() or np.mod(deltas.to_numpy(), NATIVE_CADENCE_MINUTES).any():
        raise KMASchemaError("timestamp gaps are not integer multiples of 30 minutes")
    if frame["TM"].max().to_pydatetime() > SOURCE_CUTOFF:
        raise KMACutoffError("response includes an observation after 2023-12-31 KST")
    if (
        frame["TM"].min().to_pydatetime() < station.valid_from
        or frame["TM"].max().to_pydatetime() > station.valid_to
    ):
        raise KMAStationChangeError("response crosses the registered station identity epoch")
    quality_summary = _quarantine_numeric_quality(frame)
    frame["station_name"] = station.name
    frame["proxy_group"] = station.proxy_group
    frame.attrs["quality_summary"] = quality_summary
    return frame


def _default_transport(url: str, timeout_seconds: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "P3-KMA-pre2024/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read(25 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        raise KMAExternalError(f"KMA request failed with HTTP {exc.code}") from None
    except urllib.error.URLError:
        raise KMAExternalError("KMA request failed at the network layer") from None
    if len(payload) > 25 * 1024 * 1024:
        raise KMAExternalError("KMA response exceeds the 25 MiB safety limit")
    return payload


class KMAClient:
    """Small API client whose representation and receipts never expose its key."""

    def __init__(
        self,
        api_key: str,
        *,
        transport: Transport | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key.strip():
            raise KMAExternalError("KMA_API_KEY is empty")
        self._api_key = api_key.strip()
        self._transport = transport or _default_transport
        self._timeout_seconds = timeout_seconds

    def __repr__(self) -> str:
        return "KMAClient(api_key=<redacted>)"

    def fetch_period(
        self,
        station: StationEpoch,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
        *,
        station_epochs: Mapping[int, StationEpoch] = DEFAULT_STATION_EPOCHS,
    ) -> tuple[pd.DataFrame, FetchReceipt]:
        start_kst, end_kst = validate_request_window(station, start, end)
        query = urllib.parse.urlencode(
            {
                "tm1": start_kst.strftime("%Y%m%d%H%M"),
                "tm2": end_kst.strftime("%Y%m%d%H%M"),
                "stn": str(station.station_id),
                "help": "1",
                "authKey": self._api_key,
            }
        )
        try:
            payload = self._transport(f"{KMA_PERIOD_ENDPOINT}?{query}", self._timeout_seconds)
        except Exception:
            raise KMAExternalError("KMA transport failed") from None
        frame = parse_kma_buoy_payload(
            payload,
            expected_station_id=station.station_id,
            station_epochs=station_epochs,
            allow_empty=True,
        )
        if not frame.empty and frame["TM"].min().to_pydatetime() < start_kst:
            raise KMASchemaError("response begins before the requested window")
        if not frame.empty and frame["TM"].max().to_pydatetime() > end_kst:
            raise KMASchemaError("response ends after the requested window")
        quality_summary = frame.attrs.get("quality_summary", _empty_quality_summary())
        receipt = FetchReceipt(
            station_id=station.station_id,
            requested_start=start_kst.isoformat(),
            requested_end=end_kst.isoformat(),
            response_sha256=_sha256_bytes(payload),
            row_count=len(frame),
            observed_start=None if frame.empty else frame["TM"].min().isoformat(),
            observed_end=None if frame.empty else frame["TM"].max().isoformat(),
            quality_summary=dict(quality_summary),
        )
        return frame, receipt


def iter_month_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    start_kst = _as_kst(start, field="start")
    end_kst = _as_kst(end, field="end")
    _validate_native_time(start_kst, field="start")
    _validate_native_time(end_kst, field="end")
    if end_kst < start_kst:
        raise KMAExternalError("request end precedes start")
    windows: list[tuple[datetime, datetime]] = []
    cursor = start_kst
    while cursor <= end_kst:
        if cursor.month == 12:
            next_month = cursor.replace(year=cursor.year + 1, month=1, day=1, hour=0, minute=0)
        else:
            next_month = cursor.replace(month=cursor.month + 1, day=1, hour=0, minute=0)
        window_end = min(end_kst, next_month - timedelta(minutes=NATIVE_CADENCE_MINUTES))
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(minutes=NATIVE_CADENCE_MINUTES)
    return windows


def build_independent_storm_anchors(
    frame: pd.DataFrame,
    *,
    station_epochs: Mapping[int, StationEpoch] = DEFAULT_STATION_EPOCHS,
    anchor_minimum_hs: float = 1.5,
    context_hours: int = 48,
    lead_hours: Sequence[int] = LEAD_HOURS,
    minimum_separation_hours: int = 78,
    minimum_context_coverage: float = 0.95,
) -> pd.DataFrame:
    """Select complete-target storm anchors with same-station 78-hour isolation."""

    required = {"TM", "STN", TARGET_COLUMN, "proxy_group"}
    if not required <= set(frame.columns):
        raise KMAPrecheckError(f"anchor input is missing {sorted(required - set(frame.columns))}")
    if not 0 < minimum_context_coverage <= 1:
        raise ValueError("minimum_context_coverage must be in (0, 1]")
    cadence = pd.Timedelta(minutes=NATIVE_CADENCE_MINUTES)
    context_steps = int(context_hours * 60 / NATIVE_CADENCE_MINUTES)
    selected: list[dict[str, Any]] = []
    for station_id, station_frame in frame.groupby("STN", sort=True):
        station_id = int(station_id)
        if station_id not in station_epochs:
            raise KMAStationChangeError("anchor input contains an unregistered station ID")
        station_frame = station_frame.sort_values("TM")
        if station_frame["TM"].duplicated().any():
            raise KMAPrecheckError("anchor input contains duplicate station timestamps")
        full_index = pd.date_range(
            station_frame["TM"].min(),
            station_frame["TM"].max(),
            freq=cadence,
        )
        hs = station_frame.set_index("TM")[TARGET_COLUMN].reindex(full_index)
        present = hs.notna().astype("int16")
        context_fraction = (
            present.rolling(context_steps + 1, min_periods=1).sum().div(context_steps + 1)
        )
        position = pd.Series(np.arange(len(full_index)), index=full_index)
        eligible = hs.ge(anchor_minimum_hs)
        eligible &= position.ge(context_steps)
        eligible &= context_fraction.ge(minimum_context_coverage)
        for lead_h in lead_hours:
            lead_steps = int(lead_h * 60 / NATIVE_CADENCE_MINUTES)
            eligible &= hs.shift(-lead_steps).notna()
        candidate_times = full_index[eligible.fillna(False).to_numpy()]
        previous: pd.Timestamp | None = None
        separation = pd.Timedelta(hours=minimum_separation_hours)
        epoch = station_epochs[station_id]
        for timestamp in candidate_times:
            if previous is not None and timestamp - previous < separation:
                continue
            selected.append(
                {
                    "station_id": station_id,
                    "station_name": epoch.name,
                    "proxy_group": epoch.proxy_group,
                    "anchor_time_kst": timestamp,
                }
            )
            previous = timestamp
    return pd.DataFrame(
        selected,
        columns=["station_id", "station_name", "proxy_group", "anchor_time_kst"],
    )


def run_source_precheck(
    frame: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    minimum_anchor_count: int = 600,
    minimum_station_count: int = 4,
    minimum_anchors_per_proxy: int = 100,
    quality_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run aggregate, external-only gates; local domain shift remains pending."""

    reasons: list[str] = []
    if tuple(column for column in EXPECTED_COLUMNS if column not in frame.columns):
        reasons.append("declared KMA variables are missing")
    if frame.empty:
        reasons.append("no observations")
    if not frame.empty and frame["TM"].max().to_pydatetime() > SOURCE_CUTOFF:
        reasons.append("source cutoff exceeded")
    if not frame.empty and not frame["TM"].dt.minute.isin([0, 30]).all():
        reasons.append("native 30-minute cadence violated")
    station_ids = sorted(int(value) for value in frame["STN"].dropna().unique())
    if any(value in KNOWN_REUSED_STATION_IDS for value in station_ids):
        reasons.append("known reused station ID present")
    if len(station_ids) < minimum_station_count:
        reasons.append(f"fewer than {minimum_station_count} stable stations")

    coverage = {
        column: float(frame[column].notna().mean()) if column in frame else 0.0
        for column in NUMERIC_COLUMNS
    }
    if coverage[TARGET_COLUMN] < 0.90:
        reasons.append("WH_SIG coverage below 90%")
    for column in ("WH_MAX", "WP", "PA", "HM", "TA"):
        if coverage[column] < 0.50:
            reasons.append(f"{column} coverage below 50%")
    combined_groups: dict[str, float] = {}
    for name, columns in {
        "wind_speed": ("WS1", "WS2"),
        "wind_gust": ("WS1_GST", "WS2_GST"),
    }.items():
        combined_groups[name] = (
            float(frame[list(columns)].notna().any(axis=1).mean())
            if set(columns) <= set(frame.columns)
            else 0.0
        )
    for name, value in combined_groups.items():
        if float(value) < 0.50:
            reasons.append(f"{name} coverage below 50%")

    anchor_count = len(anchors)
    if anchor_count < minimum_anchor_count:
        reasons.append(f"fewer than {minimum_anchor_count} independent storm anchors")
    by_proxy = (
        {str(key): int(value) for key, value in anchors.groupby("proxy_group").size().items()}
        if not anchors.empty
        else {}
    )
    for proxy in sorted(set(frame["proxy_group"].dropna().astype(str))):
        if by_proxy.get(proxy, 0) < minimum_anchors_per_proxy:
            reasons.append(f"{proxy} has fewer than {minimum_anchors_per_proxy} anchors")

    source_quality = quality_summary or frame.attrs.get("quality_summary") or {}
    normalized_quality = _aggregate_quality_summaries([source_quality])

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "source_only_gate": True,
        "domain_shift_local_comparison": "pending",
        "native_cadence_minutes": NATIVE_CADENCE_MINUTES,
        "target": TARGET_COLUMN,
        "observation_count": int(len(frame)),
        "station_count": len(station_ids),
        "station_ids": station_ids,
        "independent_anchor_count": anchor_count,
        "anchors_by_station": (
            {str(key): int(value) for key, value in anchors.groupby("station_id").size().items()}
            if not anchors.empty
            else {}
        ),
        "anchors_by_proxy": by_proxy,
        "coverage": {key: round(value, 6) for key, value in coverage.items()},
        "combined_coverage": {
            key: round(float(value), 6) for key, value in combined_groups.items()
        },
        "quality_summary": normalized_quality,
    }


def validate_policy_contract(
    *,
    repo_root: Path,
    catalog_path: Path,
    approval_path: Path,
) -> dict[str, Any]:
    """Validate permission and cutoff metadata before any network access."""

    sources = load_catalog(catalog_path)
    source = sources.get(SOURCE_ID)
    if source is None:
        raise PolicyError(f"{SOURCE_ID} is absent from the external-source catalog")
    if source.rights_state != "open_verified":
        raise PolicyError(f"{SOURCE_ID} is not rights-cleared")
    if "P3" not in source.eligible_problems or "pretraining" not in source.allowed_purposes:
        raise PolicyError(f"{SOURCE_ID} is not approved for P3 pretraining")
    approval = ApprovalReceipt.load(approval_path)
    if approval.status != "approved":
        raise PolicyError("organizer external-data permission is not approved")
    if SOURCE_ID not in approval.allowed_sources or "P3" not in approval.allowed_problems:
        raise PolicyError("organizer receipt does not include KMA data for P3")
    if "pretraining" not in approval.allowed_purposes:
        raise PolicyError("organizer receipt does not include pretraining")
    cutoff = pd.Timestamp(approval.cutoff_by_problem["P3"])
    if cutoff.tzinfo is None or cutoff.tz_convert(KST).to_pydatetime() != SOURCE_CUTOFF:
        raise PolicyError("organizer receipt does not preserve the conservative P3 cutoff")
    catalog_cutoff = pd.Timestamp(source.max_observation_time)
    if (
        catalog_cutoff.tzinfo is None
        or catalog_cutoff.tz_convert(KST).to_pydatetime() != SOURCE_CUTOFF
    ):
        raise PolicyError("catalog cutoff differs from the conservative P3 cutoff")
    evidence_path = Path(approval.evidence_file)
    if not evidence_path.is_absolute():
        evidence_path = repo_root / evidence_path
    if not evidence_path.is_file() or _sha256_file(evidence_path) != approval.evidence_sha256:
        raise PolicyError("official FAQ evidence is missing or its SHA256 differs")
    return {
        "accepted": True,
        "source_id": SOURCE_ID,
        "rights_state": source.rights_state,
        "license_name": source.license_name,
        "effective_cutoff": SOURCE_CUTOFF.isoformat(),
        "catalog_sha256": _sha256_file(catalog_path),
        "approval_sha256": _sha256_file(approval_path),
        "evidence_sha256": approval.evidence_sha256,
    }


def assert_credential_absent(value: Any, credential: str | None) -> None:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if credential and credential in serialized:
        raise KMAExternalError("credential redaction invariant failed")
    if re.search(r"(?i)authkey\s*=", serialized):
        raise KMAExternalError("credential-bearing URL is prohibited in receipts")


def _base_receipt(policy: Mapping[str, Any], *, status: str, mode: str) -> dict[str, Any]:
    return {
        "schema_version": "kma-buoy-preparation-v1",
        "status": status,
        "mode": mode,
        "source_id": SOURCE_ID,
        "source_cutoff_kst": SOURCE_CUTOFF.isoformat(),
        "native_cadence_minutes": NATIVE_CADENCE_MINUTES,
        "target": TARGET_COLUMN,
        "authorization": "environment_only_not_recorded",
        "policy": dict(policy),
        "official_sources": [
            KMA_API_DOCUMENTATION,
            KMA_API_GUIDE,
            KMA_DATA_PORTAL,
        ],
        "safety_invariants": {
            "p3_test_context_read": False,
            "frozen_model_modified": False,
            "oof_modified": False,
            "submission_written": False,
            "model_trained": False,
        },
    }


def prepare_kma_external(
    *,
    repo_root: Path,
    output_dir: Path,
    mode: str = "status",
    environment: Mapping[str, str] | None = None,
    catalog_path: Path | None = None,
    approval_path: Path | None = None,
    station_ids: Sequence[int] | None = None,
    start: datetime = SOURCE_FLOOR,
    end: datetime = LAST_NATIVE_OBSERVATION,
    transport: Transport | None = None,
) -> dict[str, Any]:
    """Prepare KMA data without ever persisting the API credential.

    ``status`` never performs a request. ``smoke`` retrieves a six-hour sample
    from one stable station. ``full`` retrieves monthly chunks for the selected
    stable stations and writes only ignored, quarantined Parquet artifacts.
    """

    if mode not in {"status", "smoke", "full"}:
        raise ValueError("mode must be one of: status, smoke, full")
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    catalog = (catalog_path or repo_root / "configs/external_data/catalog.toml").resolve()
    approval = (
        approval_path or repo_root / "configs/external_data/official_faq_permission.json"
    ).resolve()
    policy = validate_policy_contract(
        repo_root=repo_root,
        catalog_path=catalog,
        approval_path=approval,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ if environment is None else environment
    credential = str(env.get("KMA_API_KEY", "")).strip()
    if not credential:
        receipt = _base_receipt(policy, status="awaiting_credential", mode=mode)
        receipt["next_action"] = "set KMA_API_KEY in the process environment"
        assert_credential_absent(receipt, credential)
        _atomic_write_json(output_dir / "status.json", receipt)
        _atomic_write_json(output_dir / "retrieval_manifest.json", receipt)
        return receipt

    if mode == "status":
        receipt = _base_receipt(policy, status="credential_ready", mode=mode)
        receipt["network_request_count"] = 0
        assert_credential_absent(receipt, credential)
        _atomic_write_json(output_dir / "status.json", receipt)
        _atomic_write_json(output_dir / "retrieval_manifest.json", receipt)
        return receipt

    selected_ids = tuple(station_ids or DEFAULT_STATION_EPOCHS.keys())
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise KMAStationChangeError("station IDs must be a unique non-empty selection")
    if any(value in KNOWN_REUSED_STATION_IDS for value in selected_ids):
        raise KMAStationChangeError("known reused station ID is prohibited")
    try:
        stations = [DEFAULT_STATION_EPOCHS[int(value)] for value in selected_ids]
    except KeyError:
        raise KMAStationChangeError("station ID is not in the stable epoch registry") from None

    client = KMAClient(credential, transport=transport)
    frames: list[pd.DataFrame] = []
    fetch_receipts: list[FetchReceipt] = []
    if mode == "smoke":
        requested_end = _as_kst(end, field="end")
        smoke_start = min(max(_as_kst(start, field="start"), SOURCE_FLOOR), requested_end)
        smoke_end = min(smoke_start + timedelta(hours=6), requested_end)
        windows_by_station = [(stations[0], [(smoke_start, smoke_end)])]
    else:
        windows_by_station = [
            (station, iter_month_windows(_as_kst(start, field="start"), _as_kst(end, field="end")))
            for station in stations
        ]
    for station, windows in windows_by_station:
        for window_start, window_end in windows:
            frame, fetch_receipt = client.fetch_period(station, window_start, window_end)
            if not frame.empty:
                frames.append(frame)
            fetch_receipts.append(fetch_receipt)

    combined = (
        pd.concat(frames, ignore_index=True).sort_values(["STN", "TM"])
        if frames
        else _empty_parsed_frame(stations[0])
    )
    if combined.duplicated(["STN", "TM"]).any():
        raise KMASchemaError("combined KMA chunks contain duplicate station timestamps")
    if not combined.empty and combined["TM"].max().to_pydatetime() > SOURCE_CUTOFF:
        raise KMACutoffError("combined KMA data exceed the source cutoff")

    quality_summary = _aggregate_quality_summaries(
        [item.quality_summary for item in fetch_receipts]
    )
    observed_start = None if combined.empty else combined["TM"].min().isoformat()
    observed_end = None if combined.empty else combined["TM"].max().isoformat()
    smoke_status = "smoke_no_data" if combined.empty else "smoke_complete"

    receipt = _base_receipt(
        policy,
        status=smoke_status if mode == "smoke" else "source_precheck_pending",
        mode=mode,
    )
    receipt.update(
        {
            "station_ids": list(selected_ids if mode == "full" else selected_ids[:1]),
            "network_request_count": len(fetch_receipts),
            "empty_response_count": sum(item.row_count == 0 for item in fetch_receipts),
            "response_receipts": [asdict(item) for item in fetch_receipts],
            "row_count": int(len(combined)),
            "observed_start": observed_start,
            "observed_end": observed_end,
            "variables": list(EXPECTED_COLUMNS),
            "quality_columns": list(QUALITY_COLUMNS),
            "derived_quality_columns": list(ROW_QUALITY_COLUMNS),
            "quality_summary": quality_summary,
            "variable_units": VARIABLE_UNITS,
            "sentinel_policy": sorted(MISSING_SENTINELS),
        }
    )
    if mode == "smoke":
        assert_credential_absent(receipt, credential)
        _atomic_write_json(output_dir / "status.json", receipt)
        _atomic_write_json(output_dir / "retrieval_manifest.json", receipt)
        return receipt

    observations_path = output_dir / "kma_buoy_pre2024.parquet"
    anchors_path = output_dir / "storm_anchors.parquet"
    manifest_path = output_dir / "manifest.json"
    if observations_path.exists() or anchors_path.exists() or manifest_path.exists():
        raise KMAExternalError("full output already exists; refusing to overwrite quarantine data")
    anchors = build_independent_storm_anchors(combined)
    precheck = run_source_precheck(combined, anchors, quality_summary=quality_summary)
    combined.to_parquet(observations_path, index=False, compression="zstd")
    anchors.to_parquet(anchors_path, index=False, compression="zstd")
    manifest = {
        "schema_version": "1.0",
        "source_id": SOURCE_ID,
        "local_file": observations_path.name,
        "file_sha256": _sha256_file(observations_path),
        "observed_start": observed_start,
        "observed_end": observed_end,
        "row_count": int(len(combined)),
        "empty_response_count": receipt["empty_response_count"],
        "variables": list(EXPECTED_COLUMNS),
        "quality_columns": list(QUALITY_COLUMNS),
        "derived_quality_columns": list(ROW_QUALITY_COLUMNS),
        "quality_summary": quality_summary,
        "transformation_log": (
            "KMA kma_buoy2 monthly responses; exact 00/30-minute observations; "
            "exact -99 sentinels converted to null; physical range failures quarantined "
            "per cell; wave-order contradictions quarantined as a three-column triplet; "
            "no interpolation or resampling"
        ),
        "native_cadence_minutes": NATIVE_CADENCE_MINUTES,
        "target": TARGET_COLUMN,
        "variable_units": VARIABLE_UNITS,
        "sentinel_policy": sorted(MISSING_SENTINELS),
        "station_epochs": [
            {
                "station_id": DEFAULT_STATION_EPOCHS[value].station_id,
                "name": DEFAULT_STATION_EPOCHS[value].name,
                "proxy_group": DEFAULT_STATION_EPOCHS[value].proxy_group,
                "valid_from": DEFAULT_STATION_EPOCHS[value].valid_from.isoformat(),
                "valid_to": DEFAULT_STATION_EPOCHS[value].valid_to.isoformat(),
            }
            for value in selected_ids
        ],
        "anchors_file": anchors_path.name,
        "anchors_sha256": _sha256_file(anchors_path),
        "precheck": precheck,
        "official_sources": receipt["official_sources"],
        "license_name": policy["license_name"],
        "authorization": "environment_only_not_recorded",
    }
    receipt["precheck"] = precheck
    receipt["candidate_sha256"] = manifest["file_sha256"]
    receipt["anchor_sha256"] = manifest["anchors_sha256"]
    receipt["status"] = (
        "source_precheck_passed_domain_shift_pending"
        if precheck["accepted"]
        else "source_precheck_failed_closed"
    )
    assert_credential_absent(manifest, credential)
    assert_credential_absent(receipt, credential)
    _atomic_write_json(manifest_path, manifest)
    _atomic_write_json(output_dir / "retrieval_manifest.json", receipt)
    _atomic_write_json(output_dir / "status.json", receipt)
    if not precheck["accepted"]:
        raise KMAPrecheckError("KMA source precheck failed; quarantined data are not trainable")
    return receipt
