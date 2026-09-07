"""OptionsDX importer tests on a fixture in the vendor's wide, bracketed format."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from ingestion.optionsdx import (
    OptionsDXFormatError,
    completeness,
    import_optionsdx_dir,
    import_optionsdx_file,
    normalize_header,
    parse_optionsdx,
)
from ingestion.schema import OptionRight
from storage.store import ParquetStore

_HEADER = (
    "[QUOTE_UNIXTIME], [QUOTE_READTIME], [QUOTE_DATE], [QUOTE_TIME_HOURS], "
    "[UNDERLYING_LAST], [EXPIRE_DATE], [EXPIRE_UNIX], [DTE], "
    "[C_DELTA], [C_GAMMA], [C_VEGA], [C_THETA], [C_RHO], [C_IV], [C_VOLUME], [C_LAST], "
    "[C_SIZE], [C_BID], [C_ASK], [STRIKE], [P_BID], [P_ASK], [P_SIZE], [P_LAST], "
    "[P_DELTA], [P_GAMMA], [P_VEGA], [P_THETA], [P_RHO], [P_IV], [P_VOLUME], "
    "[STRIKE_DISTANCE], [STRIKE_DISTANCE_PCT]"
)


def _row(
    qd: str, spot: float, exp: str, strike: float, *, c_bid: float = 5.0, c_ask: float = 5.2,
    p_bid: float = 3.0, p_ask: float = 3.1, c_iv: str = "0.25", p_iv: str = "0.27",
    c_delta: str = "0.52", c_gamma: str = "0.03",
) -> str:
    return (
        f"1700000000, {qd} 16:00, {qd}, 16.0, {spot}, {exp}, 1700500000, 18, "
        f"{c_delta}, {c_gamma}, 0.4, -0.1, 0.05, {c_iv}, 1234, 5.1, 10 x 12, {c_bid}, {c_ask}, "
        f"{strike}, {p_bid}, {p_ask}, 8 x 9, 3.05, -0.48, 0.03, 0.4, -0.1, -0.05, {p_iv}, 999, "
        f"{abs(strike - spot):.1f}, {abs(strike - spot) / spot:.3f}"
    )


def _csv(rows: list[str]) -> str:
    return _HEADER + "\n" + "\n".join(rows) + "\n"


def _fixture() -> str:
    return _csv([
        _row("2024-01-02", 100.0, "2024-01-19", 100.0),
        _row("2024-01-02", 100.0, "2024-01-19", 105.0, c_bid=2.0, c_ask=2.1, p_bid=7.0, p_ask=7.2),
        _row("2024-01-03", 101.0, "2024-01-19", 100.0),
    ])


def test_normalize_header() -> None:
    assert normalize_header(" [C_BID]") == "C_BID"
    assert normalize_header("[quote_date]") == "QUOTE_DATE"


def test_parse_groups_by_quote_date_and_splits_sides() -> None:
    snaps, rep = parse_optionsdx(_fixture(), underlying="QQQ")
    assert [s.as_of for s in snaps] == [
        datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
        datetime(2024, 1, 3, 21, 0, tzinfo=UTC),
    ]
    day1 = snaps[0]
    assert day1.underlying_price == 100.0
    assert len(day1.contracts) == 4  # 2 strikes x call+put
    assert day1.source == "optionsdx"
    call = next(c for c in day1.contracts if c.strike == 100.0 and c.right is OptionRight.call)
    assert call.expiration == date(2024, 1, 19)
    assert call.bid == 5.0 and call.ask == 5.2
    assert call.iv == pytest.approx(0.25)
    assert call.delta == pytest.approx(0.52)
    assert call.volume == 1234
    assert call.open_interest == 0  # OptionsDX has no OI column
    assert call.source == "optionsdx"
    assert rep.n_rows == 3 and rep.n_contracts == 6 and rep.n_dropped == 0
    assert rep.has_open_interest is False


def test_crossed_quote_is_dropped_and_counted() -> None:
    text = _csv([_row("2024-01-02", 100.0, "2024-01-19", 100.0, c_bid=6.0, c_ask=5.0)])
    snaps, rep = parse_optionsdx(text, underlying="QQQ")
    assert len(snaps[0].contracts) == 1  # put survives, crossed call dropped
    assert rep.n_dropped == 1


def test_blank_greeks_become_none() -> None:
    text = _csv([_row("2024-01-02", 100.0, "2024-01-19", 100.0, c_iv="", c_delta="", c_gamma="")])
    snaps, _ = parse_optionsdx(text, underlying="QQQ")
    call = next(c for c in snaps[0].contracts if c.right is OptionRight.call)
    assert call.iv is None and call.delta is None and call.gamma is None


def test_missing_required_column_is_named() -> None:
    text = _fixture().replace("[C_BID]", "[C_BIDX]")
    with pytest.raises(OptionsDXFormatError, match=r"missing required column\(s\) \['C_BID'\]"):
        parse_optionsdx(text, underlying="QQQ")


def test_percent_iv_is_refused_unless_scale_declared() -> None:
    text = _csv([_row("2024-01-02", 100.0, "2024-01-19", 100.0, c_iv="25.0", p_iv="27.0")])
    with pytest.raises(OptionsDXFormatError, match=r"iv_scale=0\.01"):
        parse_optionsdx(text, underlying="QQQ")
    snaps, _ = parse_optionsdx(text, underlying="QQQ", iv_scale=0.01)
    call = next(c for c in snaps[0].contracts if c.right is OptionRight.call)
    assert call.iv == pytest.approx(0.25)


def test_optional_open_interest_column_is_used() -> None:
    header = _HEADER + ", [C_OI], [P_OI]"
    row = _row("2024-01-02", 100.0, "2024-01-19", 100.0) + ", 5000, 4000"
    snaps, rep = parse_optionsdx(header + "\n" + row + "\n", underlying="QQQ")
    assert rep.has_open_interest
    call = next(c for c in snaps[0].contracts if c.right is OptionRight.call)
    put = next(c for c in snaps[0].contracts if c.right is OptionRight.put)
    assert call.open_interest == 5000 and put.open_interest == 4000


def test_completeness_reports_the_ticket_fields() -> None:
    snaps, rep = parse_optionsdx(_fixture(), underlying="QQQ")
    c = completeness(snaps, dropped=rep.n_dropped)
    assert c.n_days == 2 and c.n_contracts == 6
    assert c.first_day == date(2024, 1, 2) and c.last_day == date(2024, 1, 3)
    assert c.two_sided_quote == 1.0
    assert c.iv == 1.0 and c.delta == 1.0 and c.gamma == 1.0
    assert c.open_interest == 0.0  # absent -> the report says so
    assert "open interest   0.0%" in "\n".join(c.as_lines())


def test_import_dir_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "qqq_2024_01.csv").write_text(_fixture(), encoding="utf-8")
    store = ParquetStore(tmp_path / "data", env="test")
    results, summary = import_optionsdx_dir(tmp_path, store, underlying="QQQ")
    assert len(results) == 1 and results[0].ok
    assert results[0].n_snapshots == 2
    assert results[0].n_persisted == 6 and results[0].n_deduplicated == 0
    assert summary.n_days == 2
    again, _ = import_optionsdx_dir(tmp_path, store, underlying="QQQ")
    assert again[0].n_persisted == 0 and again[0].n_deduplicated == 6


def test_import_file_reports_format_error_instead_of_raising(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("[QUOTE_DATE]\n2024-01-02\n", encoding="utf-8")
    store = ParquetStore(tmp_path / "data", env="test")
    result, snaps = import_optionsdx_file(bad, store, underlying="QQQ")
    assert not result.ok
    assert result.error is not None and "missing required column" in result.error
    assert snaps == []


def test_import_dir_with_nothing_to_import(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data", env="test")
    with pytest.raises(FileNotFoundError):
        import_optionsdx_dir(tmp_path, store, underlying="QQQ")


def test_bom_and_lowercase_headers_are_tolerated(tmp_path: Path) -> None:
    text = "﻿" + _fixture().lower().replace("[quote_date]", "[quote_date]")
    f = tmp_path / "lower.csv"
    f.write_text(text, encoding="utf-8")
    store = ParquetStore(tmp_path / "data", env="test")
    result, snaps = import_optionsdx_file(f, store, underlying="QQQ")
    assert result.ok and len(snaps) == 2
