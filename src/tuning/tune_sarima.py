# src/tune_sarima.py
import ctypes
import gc
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppress convergence warnings
warnings.filterwarnings("ignore")

print("==================================================")
print("🚀 COMPILING & EVALUATING CHAMPION SARIMA (1,1,1)(0,1,1)[24]")
print("==================================================\n")

DATA_PATH = "data/model_inputs/df_univariate_deeplearning.parquet"
df_uni = pd.read_parquet(DATA_PATH, engine="fastparquet")
df_uni = df_uni.set_index('ds').sort_index()
df_uni = df_uni[~df_uni.index.duplicated()].asfreq('h').ffill()

train_boundary = pd.Timestamp('2017-12-31 23:00:00')
val_boundary = pd.Timestamp('2018-06-30 23:00:00')

y_train = df_uni.loc[df_uni.index <= train_boundary, 'y']
y_val = df_uni.loc[(df_uni.index > train_boundary) & (df_uni.index <= val_boundary), 'y']

print("⚙️ Fitting Champion Architecture on Training Data...")
model = SARIMAX(
    y_train, 
    order=(1, 1, 1), 
    seasonal_order=(0, 1, 1, 24),
    enforce_stationarity=False, 
    enforce_invertibility=False
)

champion_fit = model.fit(disp=True, maxiter=50, method='lbfgs')
print(f"✅ Training Complete. In-Sample AIC: {champion_fit.aic:,.2f}")

"""
print("\n🔮 Executing Day-Ahead (h=24) Rolling Forecast on Validation Set...")

preds = []
actuals = []

full_series = pd.concat([y_train, y_val])
val_start_idx = len(y_train)

steps = 24
# FIX 1: Reduce lookback to 4 days (96 hours). 
lookback = 24 * 14

for i in range(val_start_idx, len(full_series), steps):
    if i + steps > len(full_series):
        break
        
    context_start = max(0, i - lookback)
    context_data = full_series.iloc[context_start:i]
    target_data = full_series.iloc[i:i+steps]
    
    temp_model = SARIMAX(
        context_data,
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    temp_res = temp_model.filter(champion_fit.params)
    pred_chunk = temp_res.forecast(steps=steps)
    
    preds.extend(pred_chunk.values)
    actuals.extend(target_data.values)
    
    # FIX 2: Aggressive C-level memory reclamation
    del temp_model, temp_res
    gc.collect()
    try:
        # Forces Linux glibc to release unmapped memory back to the OS
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass
        
    # Print progress
    if len(preds) % (24 * 30) == 0:
        print(f"   ↳ Processed {len(preds) // 24} days...")

mae = mean_absolute_error(actuals, preds)
mape = np.mean(np.abs((np.array(actuals) - np.array(preds)) / np.array(actuals))) * 100

print(f"\n📊 Validation MAE  : {mae:,.1f} MW")
print(f"📊 Validation MAPE : {mape:.2f} %")
"""

Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(champion_fit, "models/sarima_champion.pkl")

"""
# Cache the predictions for the notebook
PREDS_CACHE_PATH = Path("models/sarima_preds.pkl")
joblib.dump((np.array(actuals), np.array(preds)), PREDS_CACHE_PATH)

print(f"💾 Champion saved and predictions cached to {PREDS_CACHE_PATH}")
"""
print("✨ Stage 0 SARIMA Complete!")