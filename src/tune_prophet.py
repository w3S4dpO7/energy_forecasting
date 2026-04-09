# src/tune_prophet.py
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
import optuna
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import logging

# Suppress Prophet's heavy C++ logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

def objective(trial):
    # 1. Load the Prophet Native Matrix (Target + Holidays)
    df_prophet = pd.read_parquet("data/model_inputs/df_prophet.parquet", engine="fastparquet")
    
    # Define our specific holiday regressors
    holiday_regressors = ['is_holiday', 'is_bridge_day']
    
    # 2. Time Splits
    train_end = '2017-12-31 23:00:00'
    val_end = '2018-06-30 23:00:00'
    
    prophet_train = df_prophet.loc[df_prophet['ds'] <= train_end]
    prophet_val = df_prophet.loc[(df_prophet['ds'] > train_end) & (df_prophet['ds'] <= val_end)]
    
    # 3. Bayesian Search Space
    params = {
        "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
        "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True),
        "seasonality_mode": trial.suggest_categorical("seasonality_mode", ['additive', 'multiplicative'])
    }
    
    # 4. Build and Fit Model
    model = Prophet(**params, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    
    for reg in holiday_regressors:
        model.add_regressor(reg, mode=params["seasonality_mode"])
        
    model.fit(prophet_train[['ds', 'y'] + holiday_regressors])
    
    # 5. Evaluate
    forecast = model.predict(prophet_val[['ds'] + holiday_regressors])
    return mean_absolute_error(prophet_val['y'], forecast['yhat'])

if __name__ == "__main__":
    print("🚀 Initializing Prophet Bayesian Optimization (Native Paradigm)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=12) 
    
    print(f"\n✅ Best Validation MAE: {study.best_value:,.1f} MW")
    print("Best Parameters:", study.best_params)
    
    # Retrain Champion
    df_prophet = pd.read_parquet("data/model_inputs/df_prophet.parquet", engine="fastparquet")
    holiday_regressors = ['is_holiday', 'is_bridge_day']
    prophet_train = df_prophet.loc[df_prophet['ds'] <= '2017-12-31 23:00:00']
    
    champion_prophet = Prophet(**study.best_params, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    for reg in holiday_regressors:
        champion_prophet.add_regressor(reg, mode=study.best_params["seasonality_mode"])
        
    champion_prophet.fit(prophet_train[['ds', 'y'] + holiday_regressors])
    
    # Save via Prophet's JSON serializer
    Path("models").mkdir(exist_ok=True)
    with open('models/prophet_champion.json', 'w') as fout:
        fout.write(model_to_json(champion_prophet))
    print("✅ Champion saved to models/prophet_champion.json")