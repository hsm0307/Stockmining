"""Raw missing reference changes must survive panel construction and returns."""
import numpy as np
import pandas as pd
import pytest

from krxlimit.events import EventContext
from krxlimit.panel import build_panel


def _raw(changes):
    rows = []
    for date, close, change in zip(
        ("20260102", "20260105", "20260106"), (1000, 1100, 1210), changes
    ):
        rows.append({
            "BAS_DD": date, "ISU_CD": "000010", "ISU_NM": "test",
            "MKT_NM": "KOSPI", "SECT_TP_NM": "", "MARKET": "KOSPI",
            "TDD_CLSPRC": close, "CMPPREVDD_PRC": change, "FLUC_RT": "-",
            "TDD_OPNPRC": close, "TDD_HGPRC": close, "TDD_LWPRC": close,
            "ACC_TRDVOL": 100, "ACC_TRDVAL": close * 100,
            "MKTCAP": close * 1000, "LIST_SHRS": 1000,
        })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("missing", [None, "-", "", "nan"])
def test_raw_unknown_change_never_becomes_a_flat_daily_return(missing):
    panel = build_panel(_raw([0, missing, 110]), save=False)
    unknown = panel.iloc[1]
    assert pd.isna(unknown["chg"])
    assert pd.isna(unknown["base"])
    assert pd.isna(unknown["ret"])
    assert not unknown["limit_valid"]
    assert not unknown["up_close"] and not unknown["down_close"]

    ctx = EventContext(panel)
    start, code = np.array([0]), np.array([0])
    assert not ctx.observed_path_mask(start, code, [2])[0, 0]
    assert np.isnan(ctx.fwd_return(start, code, 2)[0][0])
    # The unknown move before a new entry must not discard its later known link.
    assert ctx.fwd_return(np.array([1]), code, 1)[0][0] == pytest.approx(0.10, abs=1e-6)


def test_observed_zero_change_remains_a_valid_zero_return():
    raw = _raw([0, 0, 110])
    for column in ("TDD_CLSPRC", "TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC"):
        raw.loc[1, column] = 1000
    panel = build_panel(raw, save=False)
    assert panel.iloc[1]["ret"] == 0
    ctx = EventContext(panel)
    assert ctx.fwd_return(np.array([0]), np.array([0]), 1)[0][0] == 0
