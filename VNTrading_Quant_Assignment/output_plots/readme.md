
### Strategy Sensitivity to SMA Window Length

During the backtesting phase, I experimented with different window lengths for the Simple Moving Average (SMA) used in the crossover trading strategy. The idea was to understand how the strategy’s responsiveness to price movements affects its overall profitability.

I found:

- The performance metrics are affected by the variation in the SMA period
- The total percentage returns decrease marginally with an increase in the SMA period. In my understanding, this happens due to the reduced number of trades for a larger SMA period
- Overall, the total percentage returns are of the order 2 %
- The annualized Sharpe ratio is relatively more impacted by the SMA period variation. Although it also reduces (from 10 to -2) with the increase in the SMA period (from 5 to 50)
- For the baseline SMA = 10, the Sharpe ratio is ~ 6
- Such a behaviour of the Sharpe ratio indicates that the cumulative returns adjusted to the risk are highest for the shortest SMA window case
- The maximum drawdown has an oscillatory behaviour but with a small amplitude.
- For all the cases of SMA. Overall, the amplitude of drawdown remains < 5 % all the time
