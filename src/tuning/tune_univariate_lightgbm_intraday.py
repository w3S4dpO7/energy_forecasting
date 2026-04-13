# src/tune_univariate_intraday.py
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/model_inputs")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = '2017-12-31 23:00:00'
VAL_START = '2018-01-01 00:00:00'
VAL_END   = '2018-06-30 23:00:00'

print("==================================================")
print("🚀 OPTIMIZING UNIVARIATE LIGHTGBM FOR INTRADAY (h=1)")
print("==================================================\n")

# 1. Load the Intraday Data
df_t1 = pd.read_parquet(DATA_DIR / "df_intraday_trees.parquet", engine="fastparquet")

# ── THE UNIVARIATE FIREWALL ──
# We drop ALL weather, calendar, and holiday columns.
# We keep ONLY 'total load actual', 'load_t_1', 'load_t_2', 'load_t_3', 'load_ramp_1h', and rolling stats.
allowed_prefixes = ['total load actual', 'load_', 'rolling_']
univariate_cols = [c for c in df_t1.columns if any(c.startswith(p) for p in allowed_prefixes)]
df_uni = df_t1[univariate_cols].copy()

print(f"Features retained for Univariate model: {list(df_uni.columns)}")

target_col = 'total load actual'
train_t1 = df_uni.loc[:TRAIN_END]
val_t1   = df_uni.loc[VAL_START:VAL_END]

X_train = train_t1.drop(columns=[target_col])
y_train = train_t1[target_col].values
X_val   = val_t1.drop(columns=[target_col])
y_val   = val_t1[target_col].values

# 2. Define the Optuna Objective
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    return mae

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="lgb_uni_intraday")
    study.optimize(objective, n_trials=50)
    
    print("\n🏆 BEST UNIVARIATE INTRADAY PARAMETERS FOUND:")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")
    print(f"\n📊 Best Frozen Validation MAE: {study.best_value:,.1f} MW")
    
    save_path = MODELS_DIR / "lgb_uni_intraday_params.json"
    with open(save_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"💾 Parameters permanently saved to {save_path}")