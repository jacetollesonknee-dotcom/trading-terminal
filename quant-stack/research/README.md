# research/

Standalone research scripts. Nothing here is imported by the engine; each file
runs on its own with the libraries named in its docstring.

## satellite_strategy_backtest.py

Dual-momentum / trend-regime / vol-targeted "satellite" overlay for a buy-and-hold
core, validated with a 70/30 chronological holdout and an optional rolling
walk-forward optimisation. See the module docstring for the full design.

```bash
pip install pandas numpy scipy yfinance            # matplotlib optional (chart)
python quant-stack/research/satellite_strategy_backtest.py              # live Yahoo data, 70/30 holdout
python quant-stack/research/satellite_strategy_backtest.py --wfo        # + rolling walk-forward
python quant-stack/research/satellite_strategy_backtest.py --synthetic  # offline smoke test, fake data
```

Downloaded prices are cached in `data_cache/` (refreshed when older than 5 days);
results land in `satellite_output/`. Both directories are git-ignored.
