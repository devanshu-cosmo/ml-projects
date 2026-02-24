# Portfolio Optimization and Risk Management

This repository contains a Jupyter notebook that implements a modular pipeline for equity portfolio construction, risk modelling, and dynamic rebalancing. The aim is to turn core ideas from modern portfolio theory into clear, reusable code that can be extended later.

---

## Objectives

- Build long-only equity portfolios using **mean–variance optimization (MVO)**.
- Construct a **PCA-based statistical factor model** as an alternative to the full covariance matrix.
- Compare **tangency portfolios** from MVO and factor-model covariance.
- Quantify tail risk with **Monte Carlo VaR and CVaR**.
- Evaluate a **rebalanced MVO strategy** against a buy-and-hold benchmark.

---

## Methodology

### Data and Returns

- Download daily prices for a diversified set of U.S. equities.
- Use adjusted close prices, compute daily returns, and annualize:

\[
\mu = \mathbb{E}[r] \times 252
\]

### Mean–Variance Optimization

Using PyPortfolioOpt with a chosen covariance matrix \(\Sigma\):

- **GMV portfolio**:

\[
\min_{w} \; w^\top \Sigma w 
\quad \text{s.t.} \quad \mathbf{1}^\top w = 1,\; w \ge 0
\]

- **Tangency portfolio** (max Sharpe, risk-free rate \(r_f\)):

\[
\max_{w} \; \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}
\quad \text{s.t.} \quad \mathbf{1}^\top w = 1,\; w \ge 0
\]

Sample and Ledoit–Wolf shrinkage are selectable via a small risk-model helper.

### PCA-Based Factor Model

- Apply PCA to returns to extract \(K\) statistical factors.
- Estimate

\[
R = L F + \epsilon
\]

with loadings \(L\), factor returns \(F\), and idiosyncratic noise \(\epsilon\).

- Build the factor covariance:

\[
\Sigma_{\text{factor}} = L \,\Omega\, L^\top + D
\]

where \(\Omega\) is factor covariance and \(D\) is specific risk.  
Use \(\Sigma_{\text{factor}}\) inside the same MVO routines to obtain a factor-model tangency portfolio.

---

## Risk and Performance

### Monte Carlo VaR / CVaR

- Simulate joint returns over a horizon via Cholesky decomposition of the covariance matrix.
- Map to portfolio returns using weights \(w\), derive loss distribution, and compute:

\[
\text{VaR}_{\text{CL}} = \text{quantile}_{\text{CL}}(\text{Loss})
\]

- CVaR is taken as the mean loss beyond VaR.  
VaR and CVaR are reported and visualized for both GMV and tangency portfolios.

### Efficient Frontier and Diagnostics

- Efficient frontier from PyPortfolioOpt.
- Random portfolio cloud for context.
- GMV and tangency portfolios marked on the risk–return plane.

---

## Dynamic Rebalancing

A calendar-based engine:

1. At each rebalancing date, re-estimate expected returns and covariance.
2. Solve the tangency MVO problem and update weights.
3. Hold weights until the next rebalance and track portfolio value.

Outputs include:

- Stacked weight evolution.
- Cumulative returns vs. equal-weight buy-and-hold.
- Rolling Sharpe ratio.

The design makes it easy to swap in alternative covariances or expected return models.
