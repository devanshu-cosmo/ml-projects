# Stock Return Prediction Using Gaussian Processes and PCA

---

## Overview

This notebook contains a comprehensive Jupyter notebook exploring advanced supervised learning techniques for predicting stock returns, focusing on **Gaussian Process (GP) regression** and **Principal Component Analysis (PCA)**-enhanced GP modeling. The goal is to provide robust, interpretable, and uncertainty-aware return predictions leveraging both market technical indicators and macroeconomic data.

---

## Objectives

- Implement **Gaussian Process regression** to model and predict daily returns of Microsoft (MSFT) stock.
- Use **PCA for dimensionality reduction** of financial features to improve model stability and interpretability.
- Evaluate and compare predictive accuracy of the **GP vs PCA+GP models** using error and goodness-of-fit metrics.
- Visualize model predictions with confidence intervals to assess uncertainty.
- Analyze PCA results including explained variance, component loadings, and feature correlation.
- Include **backtesting methods** to assess practical trading strategy performance derived from the predictive models.

---

## Methodology & Workflow

1. **Data Acquisition:** Retrieve historical data for MSFT and S&P 500, including technical indicators and optional macroeconomic factors (via the FRED API).
2. **Feature Engineering:** Compute returns, moving averages, momentum, volatility, and lagged variables, ensuring proper alignment.
3. **Modeling:** Build GP regression models on both raw and PCA-transformed features.
4. **Evaluation:** Quantify performance using Mean Squared Error (MSE) and R-squared (R2).
5. **Visualization:** Present actual vs predicted returns with confidence intervals and comparative plots.
6. **Backtesting:** Simulate trading strategies to evaluate profitability and risk.

---

## Salient Features

- **Uncertainty quantification:** GP confidence intervals showcase prediction reliability.
- **Flexible pipeline:** Toggle macroeconomic data inclusion and adjust PCA components easily.
- **Backtesting framework:** Demonstrates real-world applicability of models.

---
