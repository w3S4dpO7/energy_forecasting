# src/tune_mlp.py
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import optuna
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import joblib

# Map database-safe strings to the actual Python tuples
LAYER_MAPPING = {
    "64": (64,),
    "128": (128,),
    "128_64": (128, 64),
    "256_128": (256, 128),
    "128_64_32": (128, 64, 32)
}

def objective(trial):
    # 1. Load the Pre-Scaled MLP matrix AND the target scaler
    df_mlp = pd.read_parquet("data/model_inputs/df_mlp.parquet", engine="fastparquet")
    y_scaler = joblib.load("data/model_inputs/mlp_y_scaler.pkl")
    
    # 2. Split Data
    train = df_mlp.loc[:'2017-12-31 23:00:00']
    val = df_mlp.loc['2018-01-01 00:00:00':'2018-06-30 23:00:00']
    
    X_train = train.drop(columns=['total load actual'])
    y_train = train['total load actual'] # MLPRegressor expects 1D array for training
    X_val = val.drop(columns=['total load actual'])
    y_val = val['total load actual']
    
    # 3. Define Neural Architecture Search Space
    layer_choice = trial.suggest_categorical("hidden_layer_sizes_str", list(LAYER_MAPPING.keys()))
    
    params = {
        "hidden_layer_sizes": LAYER_MAPPING[layer_choice],
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "activation": 'relu',
        "solver": 'adam',
        "max_iter": 50,
        "early_stopping": True,
        "random_state": 42
    }
    
    # 4. Train and Predict (Output will be in scaled format, e.g., 0.5)
    model = MLPRegressor(**params)
    model.fit(X_train, y_train)
    preds_scaled = model.predict(X_val)
    
    # 5. Inverse Transform to calculate REAL Megawatt error
    # Scaler requires 2D array, so we reshape, transform, and flatten back to 1D
    preds_unscaled = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    y_val_unscaled = y_scaler.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
    
    return mean_absolute_error(y_val_unscaled, preds_unscaled)

if __name__ == "__main__":
    print("🚀 Initializing MLP Bayesian Optimization...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    
    print(f"\n✅ Best Validation MAE: {study.best_value:,.1f} MW")
    print("Best Parameters:", study.best_params)
    
    # Retrain and save champion
    df_mlp = pd.read_parquet("data/model_inputs/df_mlp.parquet", engine="fastparquet")
    train = df_mlp.loc[:'2017-12-31 23:00:00']
    X_train = train.drop(columns=['total load actual'])
    y_train = train['total load actual']
    
    # Extract the best parameters and map the string back to the correct tuple format
    best_params = study.best_params.copy()
    best_layer_str = best_params.pop("hidden_layer_sizes_str")
    best_params["hidden_layer_sizes"] = LAYER_MAPPING[best_layer_str]
    
    champion_mlp = MLPRegressor(**best_params, activation='relu', solver='adam', max_iter=100, random_state=42)
    champion_mlp.fit(X_train, y_train)
    
    Path("models").mkdir(exist_ok=True)
    joblib.dump(champion_mlp, "models/mlp_champion.pkl")
    print("✅ Champion saved to models/mlp_champion.pkl")