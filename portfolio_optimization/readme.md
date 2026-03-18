# Portfolio Optimization and Risk Management

This repository contains a Jupyter notebook that implements a modular pipeline for equity portfolio construction, risk modelling, and dynamic rebalancing. The aim is to turn core ideas from modern portfolio theory into clear, reusable code that can be extended later.

---

## Objectives

- Build long-only equity portfolios using mean–variance optimization (MVO).
- Construct a PCA-based statistical factor model as an alternative to the full covariance matrix.
- Compare tangency portfolios from MVO and factor-model covariance.
- Quantify tail risk with Monte Carlo VaR and CVaR.
- Evaluate a rebalanced MVO strategy against a buy-and-hold benchmark.

---

## Methodology

### Data and Returns

- Download daily prices for a diversified set of U.S. equities.
- Use adjusted close prices, compute daily returns, and annualize:

  - Annualized mean return: `mu = E[r] * 252`

### Mean–Variance Optimization

Using PyPortfolioOpt with a chosen covariance matrix `Sigma`:

- Global Minimum Variance (GMV) portfolio:

  - Objective: minimize `w^T * Sigma * w`  
  - Constraints: `1^T * w = 1`, `w >= 0`

- Tangency (max Sharpe) portfolio with risk-free rate `rf`:

  - Objective: maximize `(w^T * mu - rf) / sqrt(w^T * Sigma * w)`  
  - Constraints: `1^T * w = 1`, `w >= 0`

Sample and Ledoit–Wolf shrinkage are selectable via a small risk-model helper.

### PCA-Based Factor Model

- Apply PCA to returns to extract `K` statistical factors.
- Represent returns with a simple factor model:

  - Return decomposition: `R = L * F + eps`
  - `R`: asset return matrix  
  - `L`: loadings matrix  
  - `F`: factor returns  
  - `eps`: idiosyncratic (specific) returns

- Build the factor-based covariance:

  - `Sigma_factor = L * Omega * L^T + D`  
  - `Omega`: factor covariance matrix  
  - `D`: diagonal specific risk matrix

`Sigma_factor` is then used inside the same MVO routines to obtain a factor-model tangency portfolio.

---

## Risk and Performance

### Monte Carlo VaR / CVaR

- Simulate joint asset returns over a horizon using Cholesky decomposition of the covariance matrix.
- Map asset returns to portfolio returns using weights `w`, derive a loss distribution, and compute:

  - `VaR_CL = quantile_CL(Loss)`

- CVaR is computed as the mean loss beyond the VaR threshold.

VaR and CVaR are reported and visualized for both GMV and tangency portfolios.

### Efficient Frontier and Diagnostics

- Efficient frontier construction using PyPortfolioOpt.
- Random portfolio cloud for context.
- GMV and tangency portfolios highlighted on the risk–return plane.

---

## Dynamic Rebalancing

A calendar-based rebalancing engine:

1. At each rebalancing date, re-estimate expected returns and covariance from data up to that date.
2. Solve the tangency MVO problem and update portfolio weights.
3. Hold weights until the next rebalance and track portfolio value over time.

Output:

- Evolution of portfolio weights (stacked area chart).
- Cumulative returns vs. an equal-weight buy-and-hold benchmark.
- Rolling Sharpe ratio of the rebalanced strategy.

## Extensions for simulating realistic portfolio optimization

### Linear Programming (LP)
- Added a formulation that **maximises expected return under linear constraints**.  
- A simple baseline to compare against Mean-Variance Optimization.  
- Demonstrates how portfolio decisions change when risk is ignored.

### Mixed-Integer Programming (MILP)
- **Cardinality constraints** keep the portfolio asset number constrained.  
- Implemented using binary variables to model **asset selection decisions**.  
- Continuous allocation becomes **selection + allocation**.

### Rebalancing by including transaction costs
- Extended the rebalancing engine to include **transaction costs via an L1 penalty on weight changes**.  
- Reduces unnecessary trading and results in smoother portfolio evolution over time.  
- Introduces **path dependence**, where previous allocations influence current decisions.

### Compare turnover
- Added a comparison between strategies **with and without transaction costs**.  
- Helpful to understand how including transaction costs leads to **lower turnover and more stable portfolios**.

### Capacity / Exposure Constraints
- Added constraints to limit sector-wise exposure. 
