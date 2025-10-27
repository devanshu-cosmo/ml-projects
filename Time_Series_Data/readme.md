# ⚡ Hourly Electricity Price Forecasting & Forward Curve Modelling  

📈 **Project Goal:**  
Develop a robust **time series forecasting pipeline** to model and predict *hourly electricity prices* using **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with Exogenous Variables)**.  
The model captures **intraday, weekly, and seasonal patterns**, generating a **market-consistent Hourly Price Forward Curve (HPFC)** for future valuation and risk analysis.

---

## 🚀 Key Features

- **🔢 Data Preparation & Exploration**
  - Cleaned and interpolated 5-year (2019–2023) hourly electricity price data.
  - Conducted exploratory analysis on **daily**, **weekly**, and **seasonal** cycles.
  - Visualized trends and anomalies, including 2022 energy-crisis spikes.

- **🧠 Forecasting Model — SARIMAX**
  - Implemented a **SARIMAX model** with:
    - Seasonal structure (24-hour periodicity).  
    - Exogenous regressors (fuel prices, renewables, weather, demand indicators).  
    - Optional **crisis dummy** and **Fourier seasonality terms**.
  - Evaluated model via **Expanding** and **Sliding Window Backtesting** with metrics:  
    `MAE`, `MSE`, `RMSE`, and `MAPE`.

- **⚙️ Preprocessing Enhancements**
  - Added **Winsorization** to handle outliers and improve model robustness.  
  - Optional scaling to match a target **calendar-year forward price** (e.g., €85/MWh).

- **🧾 HPFC Construction**
  - Derived hourly price forward curve (HPFC) by scaling forecasted hourly profiles  
    to align with a specified forward-year price.  
  - Produces realistic **hourly**, **daily**, and **seasonal** price structures.

- **📊 Visualization & Insights**
  - Generated plots for:
    - Hourly intraday structure (peak vs. off-peak).  
    - Weekday/weekend differences.  
    - Monthly and seasonal averages.  
    - Forecast vs. actual residual analysis.  
  - Demonstrated how scaled forecasts remain *market-consistent* while preserving relative price shapes.

---

## 🧩 Technical Stack
- **Python** · `pandas` · `statsmodels` · `matplotlib` · `numpy`  
- **Machine Learning / Stats:** SARIMAX, time-series backtesting, winsorization  
- **Visualization:** Matplotlib, Seaborn  
- **Version Control:** GitHub  

---

## 🧪 Model Evaluation Summary

| Metric | Description | Use |
|:------:|:-------------|:----|
| **MAE / RMSE** | Quantifies average and squared forecast errors. | Accuracy check |
| **MAPE** | Percent deviation from actuals. | Normalized performance |
| **Backtesting (Expanding / Sliding)** | Evaluates model stability across time. | Robustness |

---

## 💡 Potential Extensions
- Integrate **machine learning residual models** (e.g., LightGBM on SARIMAX residuals).  
- Add **regime detection** for crisis vs. normal market conditions.  
- Use **probabilistic forecasts** or **quantile regression** for uncertainty estimation.  

---

## 📚 Author
👤 **Devanshu Sharma**  
_Ph.D. in Computational Physics — Time Series Modelling & Quantitative Forecasting_  
📧 [Contact via GitHub](mailto:devanshu@example.com) | 🌐 [LinkedIn Profile](https://linkedin.com/in/devanshu-sharma)

---

> ⚙️ *“The model defines the shape — the market defines the level.”*  
> A reproducible pipeline for quantitative electricity price forecasting.

