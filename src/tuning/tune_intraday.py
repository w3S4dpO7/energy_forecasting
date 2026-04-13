# src/tune_intraday.py
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
print("🚀 OPTIMIZING LIGHTGBM SPECIFICALLY FOR INTRADAY (h=1)")
print("==================================================\n")

# 1. Load & Prep the Intraday Data
# Using the dedicated intraday dataset (already contains load_t_1)
df_t1 = pd.read_parquet(DATA_DIR / "df_intraday_trees.parquet", engine="fastparquet")

target_col = 'total load actual'
train_t1 = df_t1.loc[:TRAIN_END]
val_t1   = df_t1.loc[VAL_START:VAL_END]

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
    
    # Fast vectorized h=1 prediction
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    return mae

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="lgb_intraday")
    study.optimize(objective, n_trials=50)
    
    print("\n🏆 BEST INTRADAY PARAMETERS FOUND:")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")
    print(f"\n📊 Best Frozen Validation MAE: {study.best_value:,.1f} MW")
    
    # Save the parameters for Stage 4 and Stage 5
    save_path = MODELS_DIR / "lgb_intraday_params.json"
    with open(save_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"💾 Parameters permanently saved to {save_path}")