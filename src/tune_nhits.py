# src/tune_nhits.py
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import optuna
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import mean_absolute_error
import warnings
import logging
import torch
import gc

# Suppress PyTorch Lightning/Darts console spam
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# ── 1. COMMAND LINE ARGUMENTS ──
parser = argparse.ArgumentParser(description="N-HiTS Bayesian Optimization pipeline.")
parser.add_argument(
    "--mode", 
    type=str, 
    choices=["initial", "improved", "sandbox", "univariate"], 
    default="improved",
    help="Select the feature engineering dataset to use."
)
args = parser.parse_args()

# ── 2. DYNAMIC PATH ROUTING ──
MODE = args.mode
if MODE == "initial":
    DATA_PATH = "data/model_inputs/df_deeplearning.parquet" # Initial was saved here originally
    MODEL_NAME = "nhits_initial"
elif MODE == "sandbox":
    DATA_PATH = "data/model_inputs/df_sandbox_deeplearning.parquet"
    MODEL_NAME = "nhits_sandbox"
elif MODE == "univariate":
    DATA_PATH = "data/model_inputs/df_univariate_deeplearning.parquet"
    MODEL_NAME = "nhits_univariate"
else: # improved
    DATA_PATH = "data/model_inputs/df_improved_deeplearning.parquet"
    MODEL_NAME = "nhits_improved"

print(f"==================================================")
print(f"🚀 RUNNING N-HiTS OPTIMIZATION IN [{MODE.upper()}] MODE")
print(f"📁 Loading data from: {DATA_PATH}")
print(f"💾 Will save model as: models/{MODEL_NAME}.pt")
print(f"==================================================\n")

def load_and_prep_data(path):
    df_dl = pd.read_parquet(path, engine="fastparquet")
    dl_dt = df_dl.set_index('ds')
    dl_dt = dl_dt[~dl_dt.index.duplicated()].asfreq('h').ffill().reset_index()
    
    # Identify covariates (columns that aren't target or ID)
    cov_cols = [c for c in dl_dt.columns if c not in ['unique_id', 'ds', 'y']]
    
    target_ts = TimeSeries.from_dataframe(dl_dt, time_col='ds', value_cols='y')
    
    # ── UNIVARIATE LOGIC CHECK ──
    if len(cov_cols) > 0:
        master_covariates = TimeSeries.from_dataframe(dl_dt, time_col='ds', value_cols=cov_cols)
    else:
        print("⚠️ No covariates found. Switching to PURE UNIVARIATE mode.")
        master_covariates = None
    
    split_train_val = pd.Timestamp('2017-12-31 23:00:00')
    train_series, val_test_series = target_ts.split_after(split_train_val)
    val_series, _ = val_test_series.split_after(pd.Timestamp('2018-06-30 23:00:00'))
    
    target_scaler = Scaler()
    train_scaled = target_scaler.fit_transform(train_series)
    val_scaled   = target_scaler.transform(val_series)
    
    return train_scaled, val_scaled, master_covariates, target_scaler

# Load globally to avoid I/O overhead during trials
TRAIN_SCALED, VAL_SCALED, MASTER_COVS, TARGET_SCALER = load_and_prep_data(DATA_PATH)

def objective(trial):
    """Bayesian Search Space with Architectural Bottlenecking."""
    num_layers = trial.suggest_int("num_layers", 2, 6) 
    
    if num_layers <= 3:
        layer_widths = trial.suggest_int("layer_widths", 256, 512, step=128)
    elif num_layers <= 5:
        layer_widths = trial.suggest_int("layer_widths", 128, 256, step=64)
    else: # 6 layers
        layer_widths = trial.suggest_int("layer_widths", 64, 128, step=32)

    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=7,        
        min_delta=1e-4,
        mode="min"
    )

    params = {
        "input_chunk_length": 168,  
        "output_chunk_length": 24,  
        "num_layers": num_layers,
        "layer_widths": layer_widths,         
        "num_blocks": trial.suggest_int("num_blocks", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "batch_size": 1024,         
        "n_epochs": 100,             
        "optimizer_kwargs": {"lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)},
        "random_state": 42,
        "save_checkpoints": True,    
        "model_name": f"nhits_{MODE}_trial_{trial.number}", 
        "work_dir": "models/tuning_checkpoints",
        "pl_trainer_kwargs": {
            "accelerator": "cuda",
            "devices": [0],
            "enable_progress_bar": False, 
            "logger": False,
            "callbacks": [early_stopper]
        }
    }
    
    try:
        model = NHiTSModel(**params)
        # fit handles past_covariates=None automatically for univariate
        model.fit(
            series=TRAIN_SCALED,
            past_covariates=MASTER_COVS, 
            val_series=VAL_SCALED,
            val_past_covariates=MASTER_COVS,
            verbose=False,
            dataloader_kwargs={"num_workers": 8}
        )
        
        # Load best weights found by EarlyStopping
        best_model = NHiTSModel.load_from_checkpoint(
            model_name=f"nhits_{MODE}_trial_{trial.number}", 
            work_dir="models/tuning_checkpoints", 
            best=True
        )


        # ── THE FIX: True Day-Ahead Rolling Origin Evaluation ──
        # We must pass the FULL series so the model can read the true validation data as it rolls forward.
        full_series = TRAIN_SCALED.append(VAL_SCALED)
        
        pred_list = best_model.historical_forecasts(
            series=full_series, 
            past_covariates=MASTER_COVS,
            start=VAL_SCALED.start_time(),
            forecast_horizon=24,
            stride=24,
            retrain=False,
            last_points_only=False, # Keep all 24 hours of each daily forecast
            verbose=False
        )
        
        # Un-scale and flatten the 24-hour chunks
        preds_unscaled_list = [TARGET_SCALER.inverse_transform(ts) for ts in pred_list]
        preds_unscaled_values = np.concatenate([ts.values().flatten() for ts in preds_unscaled_list])
        
        # Flatten true validation values and trim overlap
        y_val_unscaled_values = TARGET_SCALER.inverse_transform(VAL_SCALED).values().flatten()
        val_len = len(y_val_unscaled_values)
        preds_unscaled_values = preds_unscaled_values[:val_len]
        
        mae = mean_absolute_error(y_val_unscaled_values, preds_unscaled_values)



        """
        preds_scaled = best_model.predict(
            n=len(VAL_SCALED), 
            series=TRAIN_SCALED, 
            past_covariates=MASTER_COVS,
            dataloader_kwargs={"num_workers": 8}
        )
        
        preds_unscaled = TARGET_SCALER.inverse_transform(preds_scaled)
        y_val_unscaled = TARGET_SCALER.inverse_transform(VAL_SCALED)
        mae = mean_absolute_error(y_val_unscaled.values(), preds_unscaled.values())
        """
        
        # Explicit VRAM Cleanup
        del model, best_model
        torch.cuda.empty_cache()
        gc.collect()
        
        return mae

    except Exception as e:
        torch.cuda.empty_cache()
        gc.collect()
        if "out of memory" in str(e).lower():
            return 1e9 # Penalize OOM configuration in Optuna
        raise e

if __name__ == "__main__":
    # Ensure checkpoint directory exists
    Path("models/tuning_checkpoints").mkdir(parents=True, exist_ok=True)

    # Dynamic DB name ensures Optuna results don't mix between paradigms
    db_name = f"nhits_full_{MODE}"
    study = optuna.create_study(
        study_name=db_name,
        storage=f"sqlite:///{db_name}.db",
        load_if_exists=True,
        direction="minimize"
    )
    
    study.optimize(objective, n_trials=25) 
    
    print(f"\n🏆 Best Configuration Found! MAE: {study.best_value:,.1f} MW")
    print(study.best_params)
    
    # ── EXTRACT AND SAVE THE CHAMPION ──
    best_trial_number = study.best_trial.number
    print(f"\n💾 Retrieving exact weights from winning run (Trial {best_trial_number})...")
    
    champion_nhits = NHiTSModel.load_from_checkpoint(
        model_name=f"nhits_{MODE}_trial_{best_trial_number}", 
        work_dir="models/tuning_checkpoints", 
        best=True
    )
    
    Path("models").mkdir(parents=True, exist_ok=True)
    champion_nhits.save(f"models/{MODEL_NAME}.pt")
    print(f"✅ Final Champion cleanly saved to models/{MODEL_NAME}.pt")

    # ── CLEANUP: REMOVE TRIAL CHECKPOINTS ──
    print(f"🧹 Cleaning up temporary Optuna checkpoints for {MODE} mode...")
    import shutil
    checkpoint_dir = Path("models/tuning_checkpoints")
    
    # Iterate and remove only the folders belonging to this specific mode
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and f"nhits_{MODE}_trial_" in item.name:
            shutil.rmtree(item)
            
    print("✨ Cleanup complete!")