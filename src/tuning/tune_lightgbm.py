# src/tune_lightgbm.py
import argparse
import warnings
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# ── 1. COMMAND LINE ARGUMENTS ──
parser = argparse.ArgumentParser(description="LightGBM Bayesian Optimization pipeline.")
parser.add_argument(
    "--mode", 
    type=str, 
    choices=["initial", "improved", "sandbox_a", "sandbox_b"], 
    default="improved",
    help="Select the feature engineering dataset to use."
)
args = parser.parse_args()

# ── 2. DYNAMIC PATH ROUTING ──
MODE = args.mode
if MODE == "initial":
    DATA_PATH = "data/model_inputs/df_trees.parquet" 
    MODEL_NAME = "lgb_initial"
elif MODE == "sandbox_a":
    DATA_PATH = "data/model_inputs/df_sandbox_a_trees.parquet"
    MODEL_NAME = "lgb_sandbox_a"
elif MODE == "sandbox_b":
    DATA_PATH = "data/model_inputs/df_sandbox_b_trees.parquet"
    MODEL_NAME = "lgb_sandbox_b"
else: # improved
    DATA_PATH = "data/model_inputs/df_improved_trees.parquet"
    MODEL_NAME = "lgb_improved"

print(f"==================================================")
print(f"🚀 RUNNING LIGHTGBM OPTIMIZATION IN [{MODE.upper()}] MODE")
print(f"📁 Loading data from: {DATA_PATH}")
print(f"💾 Will save model as: models/{MODEL_NAME}.txt")
print(f"==================================================\n")

# Load globally to save RAM
DF_TREES = pd.read_parquet(DATA_PATH, engine="fastparquet")

def objective(trial):
    # Split Data
    train_end = '2017-12-31 23:00:00'
    val_end = '2018-06-30 23:00:00'
    
    train = DF_TREES.loc[:train_end]
    val = DF_TREES.loc['2018-01-01 00:00:00':val_end]
    
    X_train = train.drop(columns=['total load actual'])
    y_train = train['total load actual']
    X_val = val.drop(columns=['total load actual'])
    y_val = val['total load actual']
    
    # ── THE DYNAMIC SEARCH SPACE (INFORMATION BOTTLENECK) ──
    if "sandbox" in MODE:
        # Constrained capacity for noisy human-dispatched data
        max_depth = trial.suggest_int("max_depth", 3, 7)
        num_leaves = trial.suggest_int("num_leaves", 10, 60)
        # Aggressive column dropping (Tree Dropout) to prevent generation feature dominance
        colsample = trial.suggest_float("colsample_bytree", 0.3, 0.7) 
    else:
        # Wide capacity for clean, stable thermodynamic data
        max_depth = trial.suggest_int("max_depth", 4, 12)
        num_leaves = trial.suggest_int("num_leaves", 20, 150)
        colsample = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "colsample_bytree": colsample,
        "random_state": 42,
        "n_jobs": 4, # Safe CPU threading
        "verbose": -1
    }
    
    # Train and Evaluate
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    
    return mean_absolute_error(y_val, preds)

if __name__ == "__main__":
    db_name = f"lgb_tuning_{MODE}"
    study = optuna.create_study(
        study_name=db_name,
        storage=f"sqlite:///{db_name}.db",
        load_if_exists=True,
        direction="minimize"
    )
    study.optimize(objective, n_trials=50)
    
    print(f"\n✅ Best Validation MAE: {study.best_value:,.1f}")
    print("Best Parameters:", study.best_params)
    
    # Train final champion model with best params
    train = DF_TREES.loc[:'2017-12-31 23:00:00']
    X_train = train.drop(columns=['total load actual'])
    y_train = train['total load actual']
    
    champion_lgb = lgb.LGBMRegressor(**study.best_params, random_state=42, n_jobs=4, verbose=-1)
    champion_lgb.fit(X_train, y_train)
    
    Path("models").mkdir(exist_ok=True)
    champion_lgb.booster_.save_model(f"models/{MODEL_NAME}.txt")
    print(f"🏆 Champion LightGBM cleanly saved to models/{MODEL_NAME}.txt")