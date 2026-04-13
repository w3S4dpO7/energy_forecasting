# src/simulate_mlops.py
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

# ── 1. PATHS & DATES ──
DATA_DIR = Path("data/model_inputs")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = MODELS_DIR / "mlops_stage5_predictions.pkl"
PARAMS_FILE = MODELS_DIR / "lgb_intraday_params.json"

TRAIN_END = '2017-12-31 23:00:00'
VAL_START = '2018-01-01 00:00:00'
VAL_END   = '2018-06-30 23:00:00'

print("==================================================")
print("🚀 LAUNCHING MLOPS CONTINUOUS LEARNING SIMULATOR")
print("==================================================\n")

# ── 2. LOAD & PREP DATA ──
print("📁 Loading Intraday Tree Data...")
df_t1 = pd.read_parquet(DATA_DIR / "df_intraday_trees.parquet", engine="fastparquet")

target_col = 'total load actual'
train_t1 = df_t1.loc[:TRAIN_END]
val_t1   = df_t1.loc[VAL_START:VAL_END]

X_train_t1 = train_t1.drop(columns=[target_col])
y_train_t1 = train_t1[target_col].values
X_val_t1   = val_t1.drop(columns=[target_col])
y_val_t1   = val_t1[target_col].values

# ── 3. INITIALIZE MODEL (Using Stage 2 Optuna Champion) ──
current_X_train = X_train_t1.copy()
current_y_train = pd.Series(y_train_t1).copy()

if not PARAMS_FILE.exists():
    raise FileNotFoundError(f"Missing {PARAMS_FILE}. Run tune_intraday.py first!")

with open(PARAMS_FILE, 'r') as f:
    champion_params = json.load(f)

# Add static execution parameters
champion_params.update({"random_state": 42, "n_jobs": -1, "verbose": -1})

print(f"⚙️ Initializing LightGBM with Optuna Champion Hyperparameters...")
mlops_model = lgb.LGBMRegressor(**champion_params)
mlops_model.fit(current_X_train, current_y_train)

# ── 4. THE MLOPS LOOP ──
print("\n⚙️ Executing Hourly Inference with Nightly Retraining...")
start_time = time.time()

mlops_preds = []
mlops_actuals = []
new_X_buffer = []
new_y_buffer = []

total_hours = len(X_val_t1)

for i in range(total_hours):
    X_hour = X_val_t1.iloc[[i]]
    y_hour = y_val_t1[i]
    
    # Infer
    pred = mlops_model.predict(X_hour)[0]
    mlops_preds.append(pred)
    mlops_actuals.append(y_hour)
    
    # Observe
    new_X_buffer.append(X_hour)
    new_y_buffer.append(y_hour)
    
    # Nightly Batch Retrain
    if (i + 1) % 24 == 0:
        current_X_train = pd.concat([current_X_train] + new_X_buffer)
        current_y_train = pd.concat([current_y_train, pd.Series(new_y_buffer)])
        
        new_X_buffer = []
        new_y_buffer = []
        
        mlops_model.fit(current_X_train, current_y_train)
        
        current_day = (i + 1) // 24
        if current_day % 30 == 0 or current_day == (total_hours // 24):
            print(f"   ↳ Completed Day {current_day:3d} | Matrix size: {len(current_X_train):,} rows")

# ── 5. CACHE THE RESULTS ──
joblib.dump((mlops_actuals, mlops_preds), CACHE_FILE)
elapsed = time.time() - start_time

print("\n==================================================")
print(f"✅ MLOps Simulation complete in {elapsed:.1f} seconds.")
print(f"💾 Results safely cached to {CACHE_FILE}")
print("==================================================")