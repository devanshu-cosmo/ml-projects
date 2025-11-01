
### Strategy Sensitivity to SMA Window Length

During the backtesting phase, I experimented with different window lengths for the Simple Moving Average (SMA) used in the crossover trading strategy. The idea was to understand how the strategy’s responsiveness to price movements affects its overall profitability.

I found:

- The performance metrics are affected by the variation in the SMA period
- The total returns decrease gradually with an increase in the SMA period. In my understanding, this happens due to the reduced number of trades for a larger SMA period
- The annualized Sharpe ratio is relatively less impacted by the SMA period variation. Although it also reduces with the increase in the SMA period
- Such a behaviour of the Sharpe ratio indicates that the cumulative returns adjusted to the risk are highest for the shortest SMA window case
- The maximum drawdown is nearly constant for all the cases of SMA. Overall, the amplitude oscillates but remains < 0.5 % all the time
