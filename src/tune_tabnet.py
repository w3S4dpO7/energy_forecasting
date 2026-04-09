# src/tune_tabnet.py
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
import optuna
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import torch
import joblib

def objective(trial):
    # 1. Load the Pre-Scaled Matrix AND the Target Scaler
    df_mlp = pd.read_parquet("data/model_inputs/df_mlp.parquet", engine="fastparquet")
    y_scaler = joblib.load("data/model_inputs/mlp_y_scaler.pkl")
    
    # 2. Split Data
    train = df_mlp.loc[:'2017-12-31 23:00:00']
    val = df_mlp.loc['2018-01-01 00:00:00':'2018-06-30 23:00:00']
    
    # CRITICAL: TabNet requires pure NumPy arrays, not Pandas DataFrames
    X_train = train.drop(columns=['total load actual']).values
    # TabNet expects 2D targets: shape (N, 1)
    y_train = train['total load actual'].values.reshape(-1, 1) 
    
    X_val = val.drop(columns=['total load actual']).values
    y_val = val['total load actual'].values.reshape(-1, 1)
    
    # 3. Define Bayesian Search Space
    n_da = trial.suggest_int('n_da', 8, 64, step=8) 
    
    params = {
        "n_d": n_da,
        "n_a": n_da,
        "gamma": trial.suggest_float('gamma', 1.0, 2.0), # Sparsity relaxation
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=trial.suggest_float('lr', 1e-3, 1e-1, log=True)),
        "verbose": 0,
        "seed": 42
    }
    
    # 4. Train and Evaluate
    model = TabNetRegressor(**params)
    
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['mae'], # Internal early stopping uses scaled MAE
        max_epochs=50,
        patience=10, 
        batch_size=1024,
        virtual_batch_size=128
    )
    
    # Output will be scaled (e.g., 0.5)
    preds_scaled = model.predict(X_val)
    
    # 5. Inverse Transform to calculate REAL Megawatt error
    # TabNet predictions and y_val are already 2D (N, 1), so we directly transform and flatten
    preds_unscaled = y_scaler.inverse_transform(preds_scaled).flatten()
    y_val_unscaled = y_scaler.inverse_transform(y_val).flatten()
    
    return mean_absolute_error(y_val_unscaled, preds_unscaled)

if __name__ == "__main__":
    print("🚀 Initializing TabNet Bayesian Optimization...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    
    print(f"\n✅ Best Validation MAE: {study.best_value:,.1f} MW")
    print("Best Parameters:", study.best_params)
    
    # Retrain and save champion
    df_mlp = pd.read_parquet("data/model_inputs/df_mlp.parquet", engine="fastparquet")
    train = df_mlp.loc[:'2017-12-31 23:00:00']
    
    X_train = train.drop(columns=['total load actual']).values
    y_train = train['total load actual'].values.reshape(-1, 1)
    
    # Unpack best params
    best_n_da = study.best_params['n_da']
    best_gamma = study.best_params['gamma']
    best_lr = study.best_params['lr']
    
    champion_tabnet = TabNetRegressor(
        n_d=best_n_da, n_a=best_n_da, gamma=best_gamma,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=best_lr),
        verbose=1,
        seed=42
    )
    
    champion_tabnet.fit(
        X_train=X_train, y_train=y_train,
        max_epochs=100, 
        batch_size=1024,
        virtual_batch_size=128
    )
    
    Path("models").mkdir(exist_ok=True)
    champion_tabnet.save_model("models/tabnet_champion")
    print("✅ Champion saved to models/tabnet_champion.zip")