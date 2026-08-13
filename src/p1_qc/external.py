"""Hard safety gate for optional historical external observations.

This module never downloads data. It only validates a local file after written
organizer approval has been recorded by the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MAX_EXTERNAL_TIME = pd.Timestamp("2023-12-31T23:59:59+09:00")
ALLOWED_STATION = "I-ORS"
ALLOWED_DOI = "10.22808/DATA-2024-6"


@dataclass(frozen=True)
class ExternalGateReport:
    rows: int
    max_time: str
    overlap_rows: int
    station_values: tuple[str, ...]
    accepted: bool


def validate_external_candidate(
    external_path: str | Path,
    local_train_path: str | Path,
    *,
    organizer_approval_path: str | Path,
    doi: str,
) -> ExternalGateReport:
    approval = Path(organizer_approval_path)
    if not approval.is_file() or not approval.read_text(encoding="utf-8").strip():
        raise PermissionError("written organizer approval is required before external data use")
    if doi != ALLOWED_DOI:
        raise PermissionError(f"only the approved CC BY candidate DOI {ALLOWED_DOI} is allowed")
    external = pd.read_csv(external_path)
    required = {"station", "time", "temp"}
    if not required.issubset(external.columns):
        raise ValueError(f"external data must contain {sorted(required)}")
    times = pd.to_datetime(external["time"], utc=True).dt.tz_convert("Asia/Seoul")
    if times.max() > MAX_EXTERNAL_TIME:
        raise PermissionError("external values after 2023-12-31 KST are permanently prohibited")
    stations = tuple(sorted(external["station"].dropna().astype(str).unique()))
    if stations != (ALLOWED_STATION,):
        raise PermissionError("only licensed historical I-ORS observations are allowed")
    local = pd.read_csv(local_train_path, usecols=["station", "time"])
    local_times = pd.to_datetime(local["time"], utc=True).dt.tz_convert("Asia/Seoul")
    local_keys = pd.MultiIndex.from_arrays([local["station"].astype(str), local_times])
    external_keys = pd.MultiIndex.from_arrays([external["station"].astype(str), times])
    overlap = int(external_keys.isin(local_keys).sum())
    if overlap:
        raise PermissionError(f"external/local station-time overlap is prohibited ({overlap} rows)")
    return ExternalGateReport(
        rows=len(external),
        max_time=times.max().isoformat(),
        overlap_rows=overlap,
        station_values=stations,
        accepted=True,
    )
