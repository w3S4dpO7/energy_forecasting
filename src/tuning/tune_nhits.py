import argparse
import gc
import logging
import shutil
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import mean_absolute_error

# Suppress PyTorch Lightning/Darts console spam
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# ── 1. COMMAND LINE ARGUMENTS ──
parser = argparse.ArgumentParser(description="N-HiTS Bayesian Optimization pipeline.")
parser.add_argument(
    "--mode", 
    type=str, 
    choices=["initial", "improved", "univariate", "univariate_long"], 
    default="improved",
    help="Select the feature engineering dataset to use."
)
args = parser.parse_args()

# ── 2. DYNAMIC PATH & ARCHITECTURE ROUTING ──
MODE = args.mode
INPUT_CHUNK = 168 

if MODE == "initial":
    DATA_PATH = "data/model_inputs/df_deeplearning.parquet" 
    MODEL_NAME = "nhits_initial"
elif MODE == "univariate":
    DATA_PATH = "data/model_inputs/df_univariate_deeplearning.parquet"
    MODEL_NAME = "nhits_univariate"
elif MODE == "univariate_long":
    DATA_PATH = "data/model_inputs/df_univariate_deeplearning.parquet"
    MODEL_NAME = "nhits_univariate_long"
    INPUT_CHUNK = 720 # 30 Days
else: # improved
    DATA_PATH = "data/model_inputs/df_improved_deeplearning.parquet"
    MODEL_NAME = "nhits_improved"


print(f"==================================================")
print(f"🚀 RUNNING N-HiTS OPTIMIZATION IN [{MODE.upper()}] MODE")
print(f"📁 Loading data from: {DATA_PATH}")
print(f"🧠 Input Chunk (Lookback): {INPUT_CHUNK}h")
print(f"==================================================\n")

def load_and_prep_data(path):
    df_dl = pd.read_parquet(path, engine="fastparquet")
    dl_dt = df_dl.set_index('ds')
    dl_dt = dl_dt[~dl_dt.index.duplicated()].asfreq('h').ffill().reset_index()
    
    cov_cols = [c for c in dl_dt.columns if c not in ['unique_id', 'ds', 'y']]
    target_ts = TimeSeries.from_dataframe(dl_dt, time_col='ds', value_cols='y')
    
    if len(cov_cols) > 0:
        master_covariates = TimeSeries.from_dataframe(dl_dt, time_col='ds', value_cols=cov_cols)
    else:
        master_covariates = None
    
    split_train_val = pd.Timestamp('2017-12-31 23:00:00')
    train_series, val_test_series = target_ts.split_after(split_train_val)
    val_series, _ = val_test_series.split_after(pd.Timestamp('2018-06-30 23:00:00'))
    
    target_scaler = Scaler()
    train_scaled = target_scaler.fit_transform(train_series)
    val_scaled   = target_scaler.transform(val_series)
    
    return train_scaled, val_scaled, master_covariates, target_scaler

TRAIN_SCALED, VAL_SCALED, MASTER_COVS, TARGET_SCALER = load_and_prep_data(DATA_PATH)

def objective(trial):
    """Dynamic Bayesian Search Space based on Horizon AND Exogenous Variables."""
    
    # ── THE DIVERGING SEARCH SPACE LOGIC ──
    if "univariate" in MODE:
        if INPUT_CHUNK == 168:
            num_layers = trial.suggest_int("num_layers", 2, 4) 
            layer_widths = trial.suggest_categorical("layer_widths", [256, 512])
            dropout_rate = trial.suggest_float("dropout", 0.0, 0.2)
        else: # 720h
            num_layers = trial.suggest_int("num_layers", 3, 6) 
            layer_widths = trial.suggest_categorical("layer_widths", [256, 512, 1024])
            dropout_rate = trial.suggest_float("dropout", 0.05, 0.25)
    else:
        # MULTIVARIATE: 39 inputs. We MUST bottleneck to prevent overfitting.
        num_layers = trial.suggest_int("num_layers", 1, 3) 
        layer_widths = trial.suggest_categorical("layer_widths", [64, 128, 256])
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.4)
        
    num_blocks_sampled = trial.suggest_int("num_blocks", 1, 3)

    # ── THE FIX: DYNAMIC POOLING KERNELS ──
    # Multiply the inner tuple by num_blocks so Darts doesn't crash
    if INPUT_CHUNK == 168:
        custom_pooling = tuple((k,) * num_blocks_sampled for k in [24, 6, 1])
    elif INPUT_CHUNK == 720:
        custom_pooling = tuple((k,) * num_blocks_sampled for k in [168, 24, 1])
    else:
        custom_pooling = None

    early_stopper = EarlyStopping(
        monitor="val_loss", patience=7, min_delta=1e-4, mode="min"
    )

    params = {
        "input_chunk_length": INPUT_CHUNK,  
        "output_chunk_length": 24,  
        "num_stacks": 3, 
        "pooling_kernel_sizes": custom_pooling,
        "num_layers": num_layers,
        "layer_widths": layer_widths,         
        "num_blocks": num_blocks_sampled,
        "dropout": dropout_rate,
        "batch_size": 512 if (INPUT_CHUNK == 720 or "univariate" not in MODE) else 1024,      
        "n_epochs": 100,             
        "optimizer_kwargs": {"lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True)},
        "random_state": 42,
        "save_checkpoints": True,    
        "model_name": f"nhits_{MODE}_trial_{trial.number}", 
        "work_dir": "models/tuning_checkpoints",
        "pl_trainer_kwargs": {
            "accelerator": "cuda", "devices": [0],
            "enable_progress_bar": False, "logger": False,
            "callbacks": [early_stopper]
        }
    }
    
    try:
        model = NHiTSModel(**params)
        model.fit(
            series=TRAIN_SCALED, past_covariates=MASTER_COVS, 
            val_series=VAL_SCALED, val_past_covariates=MASTER_COVS,
            verbose=False, dataloader_kwargs={"num_workers": 8}
        )
        
        best_model = NHiTSModel.load_from_checkpoint(
            model_name=f"nhits_{MODE}_trial_{trial.number}", work_dir="models/tuning_checkpoints", best=True
        )

        full_series = TRAIN_SCALED.append(VAL_SCALED)
        pred_list = best_model.historical_forecasts(
            series=full_series, past_covariates=MASTER_COVS,
            start=VAL_SCALED.start_time(), forecast_horizon=24,
            stride=24, retrain=False, last_points_only=False, verbose=False
        )
        
        preds_unscaled_list = [TARGET_SCALER.inverse_transform(ts) for ts in pred_list]
        preds_unscaled_values = np.concatenate([ts.values().flatten() for ts in preds_unscaled_list])
        y_val_unscaled_values = TARGET_SCALER.inverse_transform(VAL_SCALED).values().flatten()
        
        val_len = len(y_val_unscaled_values)
        preds_unscaled_values = preds_unscaled_values[:val_len]
        
        mae = mean_absolute_error(y_val_unscaled_values, preds_unscaled_values)
        
        del model, best_model
        torch.cuda.empty_cache()
        gc.collect()
        return mae

    except Exception as e:
        torch.cuda.empty_cache()
        gc.collect()
        if "out of memory" in str(e).lower(): return 1e9 
        raise e

if __name__ == "__main__":
    Path("models/tuning_checkpoints").mkdir(parents=True, exist_ok=True)
    db_name = f"nhits_full_{MODE}"
    study = optuna.create_study(
        study_name=db_name, storage=f"sqlite:///{db_name}.db",
        load_if_exists=True, direction="minimize"
    )
    
    study.optimize(objective, n_trials=20) 
    
    print(f"\n🏆 Best Configuration Found! MAE: {study.best_value:,.1f} MW")
    best_trial_number = study.best_trial.number
    
    champion_nhits = NHiTSModel.load_from_checkpoint(
        model_name=f"nhits_{MODE}_trial_{best_trial_number}", 
        work_dir="models/tuning_checkpoints", best=True
    )
    
    Path("models").mkdir(parents=True, exist_ok=True)
    champion_nhits.save(f"models/{MODEL_NAME}.pt")
    
    for item in Path("models/tuning_checkpoints").iterdir():
        if item.is_dir() and f"nhits_{MODE}_trial_" in item.name:
            shutil.rmtree(item)
    print("✨ Cleanup complete!")