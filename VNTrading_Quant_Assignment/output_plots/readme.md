
### Strategy Sensitivity to SMA Window Length

During the backtesting phase, I experimented with different window lengths for the Simple Moving Average (SMA) used in the crossover trading strategy. The idea was to understand how the strategy’s responsiveness to price movements affects its overall profitability.

I found:

| SMA Window | Observed Performance |
| 5 | Low returns – too many false trades due to high sensitivity |
| 10 | Baseline performance – balanced responsiveness |
| 15 | Lower returns – delayed reactions and premature exits |
| 20 | Significantly higher returns – smoother signals capturing longer trends |

The results suggests that the SMA window directly regulates the responsiveness of the signal to the market movements. Shorter windows respond too quickly, leading to unneccesary trades. Longer windows lead to smaller number of trades, but capture the actual trends in the chart. The results for SMA=15 is a bit unintuitive. For SMA=15, it is possible that the window is not reactive enough for short trends but also contains the noises for long term trends.
