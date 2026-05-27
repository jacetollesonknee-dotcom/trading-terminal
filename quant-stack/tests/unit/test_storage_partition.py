"""Partition-path helper tests. Pure path math; no I/O."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from storage import partition


def test_equity_paths_namespaced_by_env() -> None:
    p_dev = partition.equity_file(Path("data"), "dev", "NVDA", 2024, "1d")
    p_paper = partition.equity_file(Path("data"), "paper", "NVDA", 2024, "1d")
    assert p_dev != p_paper
    assert "dev" in str(p_dev)
    assert "paper" in str(p_paper)


def test_equity_symbol_uppercased() -> None:
    p = partition.equity_file(Path("data"), "dev", "nvda", 2024, "1d")
    assert "NVDA" in str(p)
    assert "nvda" not in str(p)


def test_equity_file_layout() -> None:
    p = partition.equity_file(Path("data"), "dev", "NVDA", 2024, "1d")
    s = str(p).replace("\\", "/")
    assert s.endswith("data/dev/equities/NVDA/year=2024/interval=1d/bars.parquet")


def test_options_chain_path_layout() -> None:
    p = partition.options_chain_file(
        Path("data"), "dev", "nvda",
        expiry=date(2024, 6, 21), snapshot=date(2024, 3, 15),
    )
    s = str(p).replace("\\", "/")
    assert s.endswith(
        "data/dev/options/NVDA/expiry=2024-06-21/snapshot=2024-03-15.parquet"
    )


def test_corporate_actions_path_layout() -> None:
    p = partition.corporate_actions_file(Path("data"), "paper", "spy")
    s = str(p).replace("\\", "/")
    assert s.endswith("data/paper/corporate_actions/SPY.parquet")


def test_earnings_path_layout() -> None:
    p = partition.earnings_file(Path("data"), "live", "AAPL")
    s = str(p).replace("\\", "/")
    assert s.endswith("data/live/earnings/AAPL.parquet")


def test_scan_returns_empty_when_root_missing(tmp_path: Path) -> None:
    assert partition.scan_equity_files(tmp_path, "dev") == []
    assert partition.scan_options_files(tmp_path, "dev") == []
    assert partition.scan_corporate_actions_files(tmp_path, "dev") == []
    assert partition.scan_earnings_files(tmp_path, "dev") == []


def test_scan_finds_only_correct_env(tmp_path: Path) -> None:
    dev_file = partition.equity_file(tmp_path, "dev", "X", 2024, "1d")
    paper_file = partition.equity_file(tmp_path, "paper", "X", 2024, "1d")
    dev_file.parent.mkdir(parents=True)
    paper_file.parent.mkdir(parents=True)
    dev_file.write_bytes(b"x")
    paper_file.write_bytes(b"x")

    dev_scan = partition.scan_equity_files(tmp_path, "dev")
    paper_scan = partition.scan_equity_files(tmp_path, "paper")
    assert dev_scan == [dev_file]
    assert paper_scan == [paper_file]


def test_equity_symbol_root_helper() -> None:
    p = partition.equity_symbol_root(Path("data"), "dev", "nvda")
    s = str(p).replace("\\", "/")
    assert s.endswith("data/dev/equities/NVDA")


def test_options_underlying_root_helper() -> None:
    p = partition.options_underlying_root(Path("data"), "dev", "spy")
    s = str(p).replace("\\", "/")
    assert s.endswith("data/dev/options/SPY")
