
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

- The strategy generated a total return of 2.1 %, Sharpe ratio of 6 and a maximum drawdown of 2 % for baseline SMA = 10. 
- The equity curve shows the strategy's performance over the test period.
- Changing the SMA window drastically affects the performance metrics


## Interpretation

The output plot for the baseline SMA = 10 suggests that the returns rise steadily, with occasional drops. This is followed by a rather oscillatory phase and the total returns stay of the same order. Towards the final few minutes, the curve falls and rises again. Moreover, the drawdown also rises, suggesting that the model is taking increasingly risky trades. However, the overall fluctuation of the drawdown is small in amplitude and stays comfortably below 5 percent. The SMA window has significant effects on this trend, as seen from the output plots.

### Update 1: Added a Grid Search feature that helps to find an optimal SMA window based on the performance metrics

### Update 2: Added a plot showing how the performance metrics vary with the variation in SMA window

### Update 3: Added a function that considers non-zero transaction fees while backtesting the strategy
