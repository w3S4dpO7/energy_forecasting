# src/evaluate_test_mlops.py
import json
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 1. PATHS & DATES ──
DATA_DIR = Path("data/model_inputs")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = MODELS_DIR / "mlops_test_predictions_probabilistic.pkl"
# Make sure to point to your chosen multivariate intraday parameters
PARAMS_FILE = MODELS_DIR / "lgb_intraday_params.json" 

# THE NEW BOUNDARIES: Train now includes the Validation set. Test is the rest.
TRAIN_VAL_END = '2018-06-30 23:00:00'
TEST_START    = '2018-07-01 00:00:00'

print("==================================================")
print("🚀 UNLOCKING THE VAULT: PROBABILISTIC TEST SET MLOPS")
print("==================================================\n")

# ── 2. LOAD & PREP DATA ──
print("📁 Loading Intraday Tree Data...")
df_t1 = pd.read_parquet(DATA_DIR / "df_intraday_trees.parquet", engine="fastparquet")

target_col = 'total load actual'

# Combine Train + Val into the new baseline knowledge
train_val_t1 = df_t1.loc[:TRAIN_VAL_END]
test_t1      = df_t1.loc[TEST_START:]

X_train_val = train_val_t1.drop(columns=[target_col])
y_train_val = train_val_t1[target_col].values

X_test = test_t1.drop(columns=[target_col])
y_test = test_t1[target_col].values

# ── 3. INITIALIZE PROBABILISTIC MODELS ──
current_X_train = X_train_val.copy()
current_y_train = pd.Series(y_train_val).copy()

if not PARAMS_FILE.exists():
    raise FileNotFoundError(f"Missing {PARAMS_FILE}. You need your champion parameters!")

with open(PARAMS_FILE, 'r') as f:
    champion_params = json.load(f)

# Strip out any conflicting objective parameters if they exist
champion_params.pop('objective', None)
champion_params.pop('metric', None)

# Add static execution parameters
base_kwargs = champion_params.copy()
base_kwargs.update({"random_state": 42, "n_jobs": -1, "verbose": -1, "objective": "quantile"})

print(f"⚙️ Initializing Quantile LightGBM models (tau = 0.05, 0.50, 0.95)...")
model_q05 = lgb.LGBMRegressor(**base_kwargs, alpha=0.05)
model_q50 = lgb.LGBMRegressor(**base_kwargs, alpha=0.50) # The Median (Point Forecast)
model_q95 = lgb.LGBMRegressor(**base_kwargs, alpha=0.95)

# Initial Fit
model_q05.fit(current_X_train, current_y_train)
model_q50.fit(current_X_train, current_y_train)
model_q95.fit(current_X_train, current_y_train)

# ── 4. THE MLOPS LOOP ON THE TEST SET ──
print("\n⚙️ Executing Hourly Inference with Nightly Retraining (Test Set)...")
start_time = time.time()

mlops_preds_05 = []
mlops_preds_50 = []
mlops_preds_95 = []
mlops_actuals = []

new_X_buffer = []
new_y_buffer = []

total_hours = len(X_test)

for i in range(total_hours):
    X_hour = X_test.iloc[[i]]
    y_hour = y_test[i]
    
    # Infer all three quantiles
    mlops_preds_05.append(model_q05.predict(X_hour)[0])
    mlops_preds_50.append(model_q50.predict(X_hour)[0])
    mlops_preds_95.append(model_q95.predict(X_hour)[0])
    mlops_actuals.append(y_hour)
    
    # Observe
    new_X_buffer.append(X_hour)
    new_y_buffer.append(y_hour)
    
    # Nightly Batch Retrain (All 3 models)
    if (i + 1) % 24 == 0:
        current_X_train = pd.concat([current_X_train] + new_X_buffer)
        current_y_train = pd.concat([current_y_train, pd.Series(new_y_buffer)])
        
        new_X_buffer = []
        new_y_buffer = []
        
        model_q05.fit(current_X_train, current_y_train)
        model_q50.fit(current_X_train, current_y_train)
        model_q95.fit(current_X_train, current_y_train)
        
        current_day = (i + 1) // 24
        if current_day % 30 == 0 or current_day == (total_hours // 24):
            print(f"   ↳ Completed Test Day {current_day:3d} | Matrix size: {len(current_X_train):,} rows")

# ── 5. CACHE THE RESULTS ──
joblib.dump({
    'actuals': mlops_actuals,
    'q05': mlops_preds_05,
    'q50': mlops_preds_50,
    'q95': mlops_preds_95
}, CACHE_FILE)

elapsed = time.time() - start_time

print("\n==================================================")
print(f"✅ Final Test Simulation complete in {elapsed:.1f} seconds.")
print(f"💾 Results safely cached to {CACHE_FILE}")
print("==================================================")