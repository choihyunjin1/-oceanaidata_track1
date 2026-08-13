from __future__ import annotations

import pandas as pd
import pytest

from p1_qc.external import validate_external_candidate


def test_external_requires_written_approval(tmp_path) -> None:
    external = tmp_path / "external.csv"
    local = tmp_path / "train.csv"
    approval = tmp_path / "approval.txt"
    pd.DataFrame(
        {"station": ["I-ORS"], "time": ["2023-01-01T00:00:00+09:00"], "temp": [10.0]}
    ).to_csv(external, index=False)
    pd.DataFrame({"station": ["I-ORS"], "time": ["2024-01-01T00:00:00+09:00"]}).to_csv(
        local, index=False
    )
    with pytest.raises(PermissionError, match="approval"):
        validate_external_candidate(
            external, local, organizer_approval_path=approval, doi="10.22808/DATA-2024-6"
        )


def test_external_rejects_2024_even_with_approval(tmp_path) -> None:
    external = tmp_path / "external.csv"
    local = tmp_path / "train.csv"
    approval = tmp_path / "approval.txt"
    approval.write_text("approved", encoding="utf-8")
    pd.DataFrame(
        {"station": ["I-ORS"], "time": ["2024-01-01T00:00:00+09:00"], "temp": [10.0]}
    ).to_csv(external, index=False)
    pd.DataFrame({"station": ["I-ORS"], "time": ["2025-01-01T00:00:00+09:00"]}).to_csv(
        local, index=False
    )
    with pytest.raises(PermissionError, match="2023"):
        validate_external_candidate(
            external, local, organizer_approval_path=approval, doi="10.22808/DATA-2024-6"
        )
