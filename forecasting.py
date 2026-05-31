"""
Dynamic Probabilistic Forecasting with LSTM
U.S. Housing Starts (FRED, 1959-2022)

Pipeline -> ->:
  1. Load data.
  2. Log transform.
  3. Train LSTM with seasonal regressors.
  4. Backtest 10x to build residual matrix.
  5. Compute per-step conformal quantiles.
  6. Plot dynamic confidence intervals.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from scalecast.Forecaster import Forecaster
from scalecast.SeriesTransformer import SeriesTransformer
from tensorflow.keras.callbacks import EarlyStopping

# CONFIGURATION
START        = '1959-01-01'
END          = '2022-12-31'
HORIZON      = 24
N_ITER       = 10
JUMP_BACK    = 12
ALPHA        = 0.10

LSTM_PARAMS = dict(
    lags=12,
    layers_struct=[
        ('LSTM', {'units': 100, 'activation': 'tanh', 'dropout': 0.0, 'return_sequences': True}),
        ('LSTM', {'units': 50,  'activation': 'tanh', 'dropout': 0.0}),
    ],
    epochs=200,
    validation_split=0.2,
    callbacks=EarlyStopping(monitor='val_loss', patience=10),
)

# 1. Load data
print('Loading data...')
df = pdr.get_data_fred('HOUST', start=START, end=END)
raw = Forecaster(
    y=df.iloc[:, 0],
    current_dates=df.index,
    future_dates=HORIZON,
    test_length=24,
    validation_length=24,
)
print(f'Loaded {len(raw.y)} observations.')

# 2. Helper: transform + add features
def prepare(f):
    t = SeriesTransformer(f)
    f = t.LogTransform()
    f.add_seasonal_regressors('month', dummy=True)
    f.add_time_trend()
    f.add_ar_terms(12)
    return f, t

# 3. Backtest to build residual matrix
print(f'\nRunning {N_ITER} backtest iterations...')
residual_matrix = []

for i in range(N_ITER):
    cutoff = len(raw.y) - HORIZON - (i * JUMP_BACK)
    if cutoff < 100:
        break

    f_iter = Forecaster(
        y=raw.y.iloc[:cutoff],
        current_dates=raw.current_dates[:cutoff],
        future_dates=HORIZON,
        test_length=0,
    )
    f_iter, t_iter = prepare(f_iter)
    f_iter.set_estimator('rnn')
    f_iter.manual_forecast(**LSTM_PARAMS, call_me='lstm')
    f_iter = t_iter.LogRevert()

    actuals = raw.y.iloc[cutoff:cutoff + HORIZON].values
    preds   = f_iter.export('lvl_fcsts')['lstm'].values[:HORIZON]

    if len(actuals) == HORIZON:
        residual_matrix.append(np.abs(actuals - preds))
        print(f'  Iter {i+1}/{N_ITER} — MAE: {residual_matrix[-1].mean():.1f}')

residual_matrix = np.array(residual_matrix)

# 4. Compute per-step conformal quantiles
n = residual_matrix.shape[0]
q = min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)
dynamic_margins = np.quantile(residual_matrix, q, axis=0)

# 5. Fit final model on full data
print('\nFitting final model...')
f_final = Forecaster(
    y=raw.y,
    current_dates=raw.current_dates,
    future_dates=HORIZON,
    test_length=24,
)

f_final, t_final = prepare(f_final)
f_final.set_estimator('rnn')
f_final.manual_forecast(**LSTM_PARAMS, call_me='lstm')

# Benchmark
f_final.set_estimator('naive')
f_final.manual_forecast(seasonal=True, m=12, call_me='naive_seasonal')

f_final = t_final.LogRevert()

print('\n── Test Set Metrics ──')
print(f_final.export('model_summaries')[['ModelNickname','TestSetRMSE','TestSetMAPE','TestSetR2']])

point_forecast = f_final.export('lvl_fcsts')['lstm'].values[:HORIZON]
future_dates   = f_final.future_dates
lower = point_forecast - dynamic_margins
upper = point_forecast + dynamic_margins

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Panel 1: Forecast with dynamic CI
ax = axes[0]
hist_dates = f_final.current_dates[-48:]
hist_vals  = f_final.y.values[-48:]
ax.plot(hist_dates, hist_vals, color='black', linewidth=1.5, label='Historical')
ax.plot(future_dates, point_forecast, color='steelblue', linewidth=2, label='LSTM Forecast')
ax.fill_between(future_dates, lower, upper, alpha=0.3, color='steelblue', label='90% Dynamic CI')
peak = dynamic_margins.argmax()
ax.axvline(future_dates[peak], color='red', linestyle='--', alpha=0.6,
           label=f'Widest interval (step {peak+1})')
ax.set_title('U.S. Housing Starts — LSTM with Dynamic Confidence Intervals', fontsize=13)
ax.set_ylabel('Housing Starts (thousands)')
ax.legend()

# Panel 2: Interval width by step
ax2 = axes[1]
steps = np.arange(1, HORIZON + 1)
ax2.bar(steps, upper - lower, color='steelblue', alpha=0.7)
ax2.set_title('Confidence Interval Width by Forecast Step', fontsize=13)
ax2.set_xlabel('Forecast Step (months ahead)')
ax2.set_ylabel('Interval Width (thousands)')
ax2.set_xticks(steps)

plt.tight_layout()
plt.savefig('dynamic_forecast.png', dpi=150)
plt.close()
print('Saved dynamic_forecast.png')

pd.DataFrame({
    'date': future_dates,
    'forecast': point_forecast.round(0),
    'lower_90': lower.round(0),
    'upper_90': upper.round(0),
    'interval_width': (upper - lower).round(0),
}).to_csv('dynamic_forecast.csv', index=False)
print('Saved dynamic_forecast.csv')
print('\nDone.')