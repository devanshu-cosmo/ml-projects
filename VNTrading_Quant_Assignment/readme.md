
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

## Implementation Details

### Data Processing
- Loaded and plotted 1-minute BTC price data
- Converted Unix timestamps to datetime index
- Calculated 1-minute returns: `return_t = (price_t - price_{t-1}) / price_{t-1}`
- Computed 10-period simple moving average (SMA)

### Signal Generation
- **Crossover Detection**: Used price-SMA difference sign changes forsignal generation
- **Buy Signal**: Previous difference < 0 AND current difference > 0
- **Sell Signal**: Previous difference > 0 AND current difference < 0

### Position Management
- Start flat (position = 0)
- Switch to long on buy signal (if not already long)
- Switch to short on sell signal (if not already short)
- Always maintain one position don't exit partially

### Backtesting Engine
- Simulated portfolio performance with $100,000 initial capital
- Calculated period returns: `position * asset_return`
- Built equity curve through return compounding
- Tracked running maximum for drawdown calculation

### Performance Metrics
- **Total Return**: (Final equity - Initial capital) / Initial capital
- **Number of Trades**: Count of position changes
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Annualized risk-adjusted returns

## Results

The strategy generated [X] trades with a total return of 2.6 % and maximum drawdown of 47%. 
The equity curve shows the strategy's performance over the test period.
