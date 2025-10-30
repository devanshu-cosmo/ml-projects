
### Strategy Sensitivity to SMA Window Length

During the backtesting phase, I experimented with different window lengths for the Simple Moving Average (SMA) used in the crossover trading strategy. The idea was to understand how the strategy’s responsiveness to price movements affects its overall profitability.

I found:

| SMA Window | Observed Performance |
| 5 | Low returns – too many false trades due to high sensitivity |
| 10 | Baseline performance – balanced responsiveness |
| 15 | Lower returns – delayed reactions and premature exits |
| 20 | Significantly higher returns – smoother signals capturing longer trends |
