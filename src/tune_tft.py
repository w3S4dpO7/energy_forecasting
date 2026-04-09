# src/tune_tft.py
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import optuna
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import mean_absolute_error
import warnings
import logging
import torch
import time
import gc
import sys

# Suppress PyTorch Lightning/Darts console spam
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# ── 1. COMMAND LINE ARGUMENTS ──
parser = argparse.ArgumentParser(description="TFT Bayesian Optimization pipeline.")
parser.add_argument(
    "--mode", 
    type=str, 
    choices=["initial", "improved", "sandbox"], 
    default="improved",
    help="Select the feature engineering dataset to use."
)
args = parser.parse_args()

# ── 2. DYNAMIC PATH ROUTING ──
MODE = args.mode
if MODE == "initial":
    DATA_PATH = "data/model_inputs/df_deeplearning.parquet" 
    MODEL_NAME = "tft_initial"
elif MODE == "sandbox":
    DATA_PATH = "data/model_inputs/df_sandbox_deeplearning.parquet"
    MODEL_NAME = "tft_sandbox"
else: # improved
    DATA_PATH = "data/model_inputs/df_improved_deeplearning.parquet"
    MODEL_NAME = "tft_improved"

print(f"==================================================")
print(f"🚀 RUNNING TFT OPTIMIZATION IN [{MODE.upper()}] MODE")
print(f"📁 Loading data from: {DATA_PATH}")
print(f"💾 Will save model as: models/{MODEL_NAME}.pt")
print(f"==================================================\n")

def load_and_prep_data(path):
    df_dl = pd.read_parquet(path, engine="fastparquet")
    dl_dt = df_dl.set_index('ds')
    dl_dt = dl_dt[~dl_dt.index.duplicated()].asfreq('h').ffill().reset_index()
    
    cov_cols = [c for c in dl_dt.columns if c not in ['unique_id', 'ds', 'y']]
    target_ts = TimeSeries.from_dataframe(dl_dt, time_col='ds', value_cols='y')
    master_covariates = TimeSeries.from_dataframe(dl_dt, time_col='ds', value_cols=cov_cols)
    
    split_train_val = pd.Timestamp('2017-12-31 23:00:00')
    train_series, val_test_series = target_ts.split_after(split_train_val)
    val_series, _ = val_test_series.split_after(pd.Timestamp('2018-06-30 23:00:00'))
    
    target_scaler = Scaler()
    train_scaled = target_scaler.fit_transform(train_series)
    val_scaled   = target_scaler.transform(val_series)
    
    return train_scaled, val_scaled, master_covariates, target_scaler

# Load globally so Optuna doesn't reload the Parquet file 30 times
TRAIN_SCALED, VAL_SCALED, MASTER_COVS, TARGET_SCALER = load_and_prep_data(DATA_PATH)

def objective(trial):
    """PHASE 1: The Sprint. Focused entirely on Architectural Topology."""
    
    num_attention_heads = trial.suggest_categorical("num_attention_heads", [2, 4, 8])
    head_dim = trial.suggest_int("head_dim", 16, 128, step=16) 
    hidden_size = num_attention_heads * head_dim
    hidden_continuous_size = trial.suggest_int("hidden_continuous_size", 8, 32, step=8)

    params = {
        "input_chunk_length": 168,  
        "output_chunk_length": 24,  
        "hidden_size": hidden_size,               
        "num_attention_heads": num_attention_heads, 
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
        "hidden_continuous_size": hidden_continuous_size,
        "dropout": 0.1,             
        "batch_size": 512,   
        "n_epochs": 4,              
        "add_relative_index": True,
        "optimizer_kwargs": {"lr": 1e-3}, 
        "random_state": 42,
        "pl_trainer_kwargs": {
            "accelerator": "cuda",
            "devices": [0],
            "enable_progress_bar": False, 
            "logger": False
        }
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model = TFTModel(**params)
            model.fit(
                series=TRAIN_SCALED,
                future_covariates=MASTER_COVS,
                val_series=VAL_SCALED,
                val_future_covariates=MASTER_COVS,
                verbose=False,
                dataloader_kwargs={"num_workers": 8} 
            )
            
            # ── THE FIX: True Day-Ahead Rolling Origin Evaluation ──
            # We must pass the FULL series so the model can read the true validation data as it rolls forward.
            full_series = TRAIN_SCALED.append(VAL_SCALED)
            
            pred_list = model.historical_forecasts(
                series=full_series, 
                future_covariates=MASTER_COVS,
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
            
            del model
            torch.cuda.empty_cache()
            
            return mean_absolute_error(y_val_unscaled_values, preds_unscaled_values)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache()
                gc.collect()
                if attempt < max_retries - 1:
                    time.sleep(300)
                else:
                    raise optuna.exceptions.TrialPruned()
            else:
                raise e

if __name__ == "__main__":
    # ── DYNAMIC OPTUNA DATABASE ──
    # Prevents mixing "sandbox" results with "improved" results
    db_name = f"tft_topology_{MODE}"
    
    study = optuna.create_study(
        study_name=db_name,
        storage=f"sqlite:///{db_name}.db", 
        load_if_exists=True,
        direction="minimize"
    )
    
    study.optimize(objective, n_trials=20) 
    
    print("\n⚙️ Commencing PHASE 2: Marathon Training of the Champion Model...")
    
    best_params = study.best_params
    champ_heads = best_params["num_attention_heads"]
    champ_hidden = champ_heads * best_params["head_dim"]

    champion_params = {
        "input_chunk_length": 168,
        "output_chunk_length": 24,
        "hidden_size": champ_hidden,                
        "lstm_layers": best_params["lstm_layers"],
        "num_attention_heads": champ_heads,         
        "hidden_continuous_size": best_params["hidden_continuous_size"], 
        "dropout": 0.1,              
        "batch_size": 256,
        "n_epochs": 100,             
        "add_relative_index": True,
        "optimizer_kwargs": {"lr": 1e-3}, 
        "random_state": 42,
        "save_checkpoints": True,    
        "model_name": f"{MODEL_NAME}_ckpt", # Dynamic checkpoint folder
        "work_dir": "models",
        "pl_trainer_kwargs": {
            "accelerator": "cuda",
            "devices": [0],
            "callbacks": [EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4, mode="min")],
            "logger": CSVLogger(save_dir="logs", name=f"tft_champion_{MODE}"),
            "enable_model_summary": True
        }
    }
    
    try:
        # ── HARD VRAM FLUSH BEFORE MARATHON ──
        gc.collect()
        torch.cuda.empty_cache()

        champion_tft = TFTModel(**champion_params)
        champion_tft.fit(
            series=TRAIN_SCALED,
            future_covariates=MASTER_COVS,
            val_series=VAL_SCALED,
            val_future_covariates=MASTER_COVS,
            verbose=True,
            dataloader_kwargs={"num_workers": 8} 
        )
        
        # Load best weights
        best_tft = TFTModel.load_from_checkpoint(model_name=f"{MODEL_NAME}_ckpt", work_dir="models", best=True)
        
        # Save dynamically named final model
        Path("models").mkdir(exist_ok=True)
        final_save_path = f"models/{MODEL_NAME}.pt"
        best_tft.save(final_save_path)
        print(f"🏆 Champion TFT compiled and saved to {final_save_path}")

        # ── CLEANUP: REMOVE MARATHON CHECKPOINTS ──
        print(f"🧹 Cleaning up temporary Lightning checkpoints for {MODEL_NAME}...")
        import shutil
        
        # PyTorch Lightning creates a folder matching the model_name
        checkpoint_folder = Path(f"models/{MODEL_NAME}_ckpt")
        
        if checkpoint_folder.exists() and checkpoint_folder.is_dir():
            shutil.rmtree(checkpoint_folder)
            print("✨ Cleanup complete!")
        else:
            print("⚠️ Checkpoint folder not found for cleanup.")
        
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print("❌ FATAL OOM ERROR DURING MARATHON RUN.")
        else:
            raise e