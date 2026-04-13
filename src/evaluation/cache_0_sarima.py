# src/evaluation/cache_0_sarima.py
import ctypes
import gc
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppress convergence warnings
warnings.filterwarnings("ignore")

print("==================================================")
print("🚀 CACHING STAGE 0: SARIMA ROLLING INFERENCE")
print("==================================================\n")

# ── 1. PATHS & DATES ──
DATA_PATH = Path("data/model_inputs/df_univariate_deeplearning.parquet")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = MODELS_DIR / "sarima_preds.pkl"

TRAIN_END = pd.Timestamp('2017-12-31 23:00:00')
VAL_END = pd.Timestamp('2018-06-30 23:00:00')

# ── 2. LOAD DATA ──
df_uni = pd.read_parquet(DATA_PATH, engine="fastparquet")
df_uni = df_uni.set_index('ds').sort_index()
df_uni = df_uni[~df_uni.index.duplicated()].asfreq('h').ffill()

y_train = df_uni.loc[df_uni.index <= TRAIN_END, 'y']
y_val = df_uni.loc[(df_uni.index > TRAIN_END) & (df_uni.index <= VAL_END), 'y']

# ── 3. TRAIN BASE MODEL ──
print("⚙️ Fitting Champion SARIMA Architecture on Training Data...")
model = SARIMAX(
    y_train, 
    order=(1, 1, 1), 
    seasonal_order=(0, 1, 1, 24),
    enforce_stationarity=False, 
    enforce_invertibility=False
)
champion_fit = model.fit(disp=False, maxiter=50, method='lbfgs')

# ── 4. ROLLING INFERENCE LOOP ──
print("\n🔮 Executing Day-Ahead (h=24) Rolling Forecast on Validation Set...")

preds = []
actuals = []

full_series = pd.concat([y_train, y_val])
val_start_idx = len(y_train)

steps = 24
lookback = 24 * 14  # The crucial 14-day window

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
    
    # Aggressive C-level memory reclamation
    del temp_model, temp_res
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass
        
    # Print progress
    if len(preds) % (24 * 30) == 0:
        print(f"   ↳ Processed {len(preds) // 24} days...")

# ── 5. CACHE RESULTS ──
joblib.dump({"actuals": np.array(actuals), "preds": np.array(preds)}, CACHE_FILE)
print(f"\n💾 Predictions successfully safely cached to {CACHE_FILE}")
print("✨ Stage 0 SARIMA Caching Complete!")