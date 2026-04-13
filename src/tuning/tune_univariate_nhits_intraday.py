# src/tune_univariate_nhits_intraday.py
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
import torch.nn as nn
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel
from pytorch_lightning.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

DATA_PATH = "data/model_inputs/df_univariate_deeplearning.parquet"

print("==================================================")
print("🚀 OPTIMIZING N-HiTS FOR INTRADAY (h=1) [DETERMINISTIC / MAE]")
print("==================================================\n")

def load_and_prep_data(path):
    df_dl = pd.read_parquet(path, engine="fastparquet")
    
    # FIX 1: Safely drop duplicates on the correct index
    df_dl = df_dl.set_index('ds')
    df_dl = df_dl[~df_dl.index.duplicated(keep='first')]
    df_dl = df_dl.asfreq('h').ffill().reset_index()
    
    target_ts = TimeSeries.from_dataframe(df_dl, time_col='ds', value_cols='y')
    
    split_train_val = pd.Timestamp('2017-12-31 23:00:00')
    train_series, val_test_series = target_ts.split_after(split_train_val)
    val_series, _ = val_test_series.split_after(pd.Timestamp('2018-06-30 23:00:00'))
    
    target_scaler = Scaler()
    train_scaled = target_scaler.fit_transform(train_series)
    val_scaled   = target_scaler.transform(val_series)
    
    return train_scaled, val_scaled

TRAIN_SCALED, VAL_SCALED = load_and_prep_data(DATA_PATH)

def objective(trial):
    input_chunk = trial.suggest_categorical("input_chunk_length", [12, 24, 48])
    num_blocks = trial.suggest_int("num_blocks", 1, 2)
    custom_pooling = tuple((k,) * num_blocks for k in [4, 2, 1])

    early_stopper = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4, mode="min")

    params = {
        "input_chunk_length": input_chunk,
        "output_chunk_length": 1,
        "num_stacks": 3,
        "pooling_kernel_sizes": custom_pooling,
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "layer_widths": trial.suggest_categorical("layer_widths", [64, 128]),
        "num_blocks": num_blocks,
        "dropout": trial.suggest_float("dropout", 0.0, 0.2),
        "batch_size": 1024,
        "n_epochs": 50,
        "optimizer_kwargs": {"lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True)},
        "random_state": 42,
        
        # ── THE FAIRNESS FIX: Explicitly optimize for MAE (L1 Loss) ──
        "loss_fn": nn.L1Loss(), 
        
        # FIX 2: Force Darts to save the model to disk during training
        "save_checkpoints": True,
        
        "model_name": f"nhits_intraday_trial_{trial.number}",
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
            series=TRAIN_SCALED, 
            val_series=VAL_SCALED, 
            verbose=False
        )
        
        val_loss = early_stopper.best_score.item()
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return val_loss

    except Exception as e:
        torch.cuda.empty_cache()
        gc.collect()
        if "out of memory" in str(e).lower(): return 1e9 
        raise e

if __name__ == "__main__":
    Path("models/tuning_checkpoints").mkdir(parents=True, exist_ok=True)
    
    db_name = "nhits_intraday_h1"
    study = optuna.create_study(
        study_name=db_name, storage=f"sqlite:///{db_name}.db",
        load_if_exists=True, direction="minimize"
    )
    
    study.optimize(objective, n_trials=15)
    
    print(f"\n🏆 Best Configuration Found!")
    best_trial_number = study.best_trial.number
    
    champion_nhits = NHiTSModel.load_from_checkpoint(
        model_name=f"nhits_intraday_trial_{best_trial_number}", 
        work_dir="models/tuning_checkpoints", best=True
    )
    
    champion_nhits.save("models/nhits_univariate_intraday.pt")
    
    for item in Path("models/tuning_checkpoints").iterdir():
        if item.is_dir() and "nhits_intraday_trial" in item.name:
            shutil.rmtree(item)
    print("✨ Champion saved and cleanup complete!")