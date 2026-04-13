import gc

import joblib
import lightgbm as lgb
import numpy as np

# src/evaluation/evaluator.py
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from prophet.serialize import model_from_json
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_absolute_error


class ModelEvaluator:
    """
    Centralized evaluation orchestrator for time-series models.
    Instantiate with specific timeframe bounds to use seamlessly across Val and Test sets.
    """
    
    def __init__(self, train_end, eval_start, eval_end, results_list=None):
        self.train_end = train_end
        self.eval_start = eval_start
        self.eval_end = eval_end
        self.results_list = results_list if results_list is not None else []

    @staticmethod
    def compute_metrics(y_true, y_pred):
        """Returns (MAE, MAPE) for MW-scale actuals."""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        mae  = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mae, mape

    def record_result(self, stage, name, mae, mape, horizon="Day-Ahead (h=24)"):
        """Appends a result dict to the injected leaderboard accumulator."""
        self.results_list.append({
            'Stage': stage, 'Model': name,
            'MAE (MW)': mae, 'MAPE (%)': mape, 'Horizon': horizon
        })
        print(f"  ✅ {name:40s}  MAE = {mae:>8,.1f} MW   MAPE = {mape:.2f}%")

    def evaluate_lgb(self, model_path, data_path, target_col='total load actual'):
        """Load a LightGBM Booster, predict on evaluation set, return (y_true, y_pred)."""
        df = pd.read_parquet(data_path, engine="fastparquet")
        eval_df = df.loc[self.eval_start:self.eval_end]
        X_eval = eval_df.drop(columns=[target_col])
        y_true = eval_df[target_col].values
        
        booster = lgb.Booster(model_file=str(model_path))
        y_pred = booster.predict(X_eval)
        return y_true, y_pred

    def evaluate_sklearn(self, model_path, data_path, scaler_path, target_col='total load actual'):
        """Load sklearn model + target scaler, inverse-transform, return (y_true, y_pred)."""
        df = pd.read_parquet(data_path, engine="fastparquet")
        y_scaler = joblib.load(scaler_path)
        
        eval_df = df.loc[self.eval_start:self.eval_end]
        X_eval = eval_df.drop(columns=[target_col])
        y_true_scaled = eval_df[target_col].values
        
        model = joblib.load(model_path)
        preds_scaled = model.predict(X_eval)
        
        y_pred = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        y_true = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        return y_true, y_pred

    def evaluate_tabnet(self, model_path, data_path, scaler_path, target_col='total load actual'):
        """Load TabNet model + target scaler, inverse-transform, return (y_true, y_pred)."""
        
        df = pd.read_parquet(data_path, engine="fastparquet")
        y_scaler = joblib.load(scaler_path)
        
        eval_df = df.loc[self.eval_start:self.eval_end]
        X_eval = eval_df.drop(columns=[target_col]).values
        y_true_scaled = eval_df[target_col].values.reshape(-1, 1)
        
        model = TabNetRegressor()
        model.load_model(str(model_path))
        preds_scaled = model.predict(X_eval)
        
        y_pred = y_scaler.inverse_transform(preds_scaled).flatten()
        y_true = y_scaler.inverse_transform(y_true_scaled).flatten()
        return y_true, y_pred

    def evaluate_prophet(self, model_path, data_path):
        """Load Prophet JSON model, predict on eval bounds, return (y_true, y_pred)."""
        
        df = pd.read_parquet(data_path, engine="fastparquet")
        prophet_eval = df.loc[(df['ds'] > self.train_end) & (df['ds'] <= self.eval_end)]
        
        with open(model_path, 'r') as f:
            model = model_from_json(f.read())
            
        regressor_cols = [c for c in df.columns if c not in ['ds', 'y']]
        forecast = model.predict(prophet_eval[['ds'] + regressor_cols])
        return prophet_eval['y'].values, forecast['yhat'].values

    def evaluate_darts(self, model_class, model_path, data_path):
        """Load a Darts model, run rolling historical_forecasts, return (y_true, y_pred)."""

        df_dl = pd.read_parquet(data_path, engine="fastparquet")
        dl_dt = df_dl.set_index('ds')
        dl_dt = dl_dt[~dl_dt.index.duplicated()].asfreq('h').ffill().reset_index()
        
        cov_cols = [c for c in dl_dt.columns if c not in ['unique_id', 'ds', 'y']]
        target_ts = TimeSeries.from_dataframe(dl_dt, time_col='ds', value_cols='y')
        
        master_covariates = TimeSeries.from_dataframe(dl_dt, time_col='ds', value_cols=cov_cols) if cov_cols else None

        # Split dynamically based on class instantiation dates
        train_series, val_test = target_ts.split_after(pd.Timestamp(self.train_end))
        eval_series, _ = val_test.split_after(pd.Timestamp(self.eval_end))

        scaler = Scaler()
        train_scaled = scaler.fit_transform(train_series)
        eval_scaled  = scaler.transform(eval_series)
        full_series  = train_scaled.append(eval_scaled)

        model = model_class.load(str(model_path))

        cls_name = model_class.__name__
        cov_kwargs = {}
        if master_covariates is not None:
            cov_kwargs = {'future_covariates': master_covariates} if 'TFT' in cls_name else {'past_covariates': master_covariates}

        pred_list = model.historical_forecasts(
            series=full_series, **cov_kwargs,
            start=eval_scaled.start_time(),
            forecast_horizon=24, stride=24,
            retrain=False, last_points_only=False, verbose=False
        )

        preds_unscaled = np.concatenate([scaler.inverse_transform(ts).values().flatten() for ts in pred_list])
        y_true_unscaled = scaler.inverse_transform(eval_scaled).values().flatten()
        preds_unscaled = preds_unscaled[:len(y_true_unscaled)]

        torch.cuda.empty_cache(); gc.collect()
        return y_true_unscaled, preds_unscaled

    def safe_evaluate(self, eval_fn, *args, model_name="Unknown", stage="?", horizon="Day-Ahead (h=24)"):
        """Wraps an evaluation function with graceful error handling."""
        try:
            y_true, y_pred = eval_fn(*args)
            mae, mape = self.compute_metrics(y_true, y_pred)
            self.record_result(stage, model_name, mae, mape, horizon)
            return True
        except FileNotFoundError:
            print(f"  ⚠️  {model_name:40s}  MODEL FILE NOT FOUND — skipping.")
            return False
        except Exception as e:
            print(f"  ❌  {model_name:40s}  ERROR: {e}")
            return False