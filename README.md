# Dynamic Probabilistic Time Series Forecasting with LSTM

## Overview
---
- Point forecasts tell you what will happen. This project tells you how confident to be.

- This project applies a Long Short Term Memory (LSTM) neural network to monthly U.S. housing starts data (FRED, 1959–2022), producing dynamic probabilistic forecasts. The confidence intervals that vary by forecast horizon, reflecting real world uncertainty rather than assuming a flat margin of error.
- The core insight: static confidence intervals treat every forecast step equally, which is unrealistic. By backtesting the model across 10 historical windows and measuring how errors evolve at each step, we build conformal prediction intervals that are calibrated, horizon-aware, and statistically grounded.

## Results

### LSTM vs. Naive Seasonal Benchmark (24-month test set)

| Model | RMSE | MAPE | 
|---|---|---|
| **LSTM** | **180** | **9.7%** |
| Naive Seasonal | 327 | 15.7% |

The LSTM is nearly **2x more accurate** than the seasonal naive baseline. The test period (2021–2022) covers the post-COVID housing boom. One of the most volatile stretches in the dataset, making any improvement over a naive model meaningful.

## How to Run

```bash
git clone https://github.com/kevin-napi/housing-forecasting-model.git
cd housing-forecasting-model
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python forecasting.py
```
