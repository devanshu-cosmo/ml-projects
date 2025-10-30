
# SMA Crossover Trading Strategy Backtest

A Python implementation of a simple moving average crossover trading strategy for BTC, with backtesting and performance analysis.

## Strategy Overview

- **Trading Instrument**: BTC
- **Timeframe**: 1-minute data
- **Strategy**: SMA Crossover
  - Buy signal: Price crosses above 10-period SMA
  - Sell signal: Price crosses below 10-period SMA
- **Position Management**: 100% long or 100% short 
- **Initial Capital**: $100000
