# src/train_tft.py
import argparse
import gc
import logging
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# Suppress console spam
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# ── 1. COMMAND LINE ARGUMENTS ──
parser = argparse.ArgumentParser(description="TFT Static Training pipeline.")
parser.add_argument(
    "--mode", 
    type=str, 
    choices=["initial", "improved", "sandbox_a"], 
    default="improved",
    help="Select the feature engineering dataset to use."
)
args = parser.parse_args()

MODE = args.mode
if MODE == "initial":
    DATA_PATH = "data/model_inputs/df_deeplearning.parquet" 
    MODEL_NAME = "tft_initial"
elif MODE == "sandbox_a":
    DATA_PATH = "data/model_inputs/df_sandbox_a_deeplearning.parquet"
    MODEL_NAME = "tft_sandbox_a"
else: # improved
    DATA_PATH = "data/model_inputs/df_improved_deeplearning.parquet"
    MODEL_NAME = "tft_improved"

print(f"==================================================")
print(f"🚀 RUNNING THEORETICAL TFT IN [{MODE.upper()}] MODE")
print(f"📁 Loading data from: {DATA_PATH}")
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

TRAIN_SCALED, VAL_SCALED, MASTER_COVS, TARGET_SCALER = load_and_prep_data(DATA_PATH)

if __name__ == "__main__":
    print("\n⚙️ Commencing Marathon Training with Theoretical Hyperparameters...")
    
    # ── DYNAMIC ARCHITECTURE ROUTING ──
    if MODE == "sandbox_a":
        print("   ↳ Sandbox Mode Detected: Applying strict Information Bottleneck to prevent Overfitting.")
        hidden_size = 64              # Halved capacity to prevent memorization
        lstm_layers = 2               # Shallower network
        num_attention_heads = 4       # Fewer attention heads
        hidden_continuous_size = 16   # Force extreme compression in the VSN
        dropout = 0.3                 # Aggressive regularization
    else:
        print("   ↳ Standard Mode Detected: Using balanced heuristic architecture.")
        hidden_size = 128
        lstm_layers = 3
        num_attention_heads = 8
        hidden_continuous_size = 32
        dropout = 0.2

    theoretical_params = {
        "input_chunk_length": 168,
        "output_chunk_length": 24,
        "hidden_size": hidden_size,                
        "lstm_layers": lstm_layers,
        "num_attention_heads": num_attention_heads,         
        "hidden_continuous_size": hidden_continuous_size, 
        "dropout": dropout, 
        "batch_size": 512,
        "n_epochs": 100,             
        "add_relative_index": True,
        "optimizer_kwargs": {"lr": 5e-4}, 
        "random_state": 42,
        "save_checkpoints": True,    
        "model_name": f"{MODEL_NAME}_ckpt",
        "work_dir": "models",
        "pl_trainer_kwargs": {
            "accelerator": "cuda",
            "devices": [0],
            # Early stopping will monitor val_loss and revert to best weights
            "callbacks": [EarlyStopping(monitor="val_loss", patience=7, min_delta=1e-4, mode="min")],
            "logger": CSVLogger(save_dir="logs", name=f"tft_theoretical_{MODE}"),
            "enable_model_summary": True
        }
    }
    
    try:
        gc.collect()
        torch.cuda.empty_cache()

        champion_tft = TFTModel(**theoretical_params)
        champion_tft.fit(
            series=TRAIN_SCALED,
            future_covariates=MASTER_COVS,
            val_series=VAL_SCALED,
            val_future_covariates=MASTER_COVS,
            verbose=True,
            dataloader_kwargs={"num_workers": 4} 
        )
        
        best_tft = TFTModel.load_from_checkpoint(model_name=f"{MODEL_NAME}_ckpt", work_dir="models", best=True)
        
        Path("models").mkdir(exist_ok=True)
        final_save_path = f"models/{MODEL_NAME}.pt"
        best_tft.save(final_save_path)
        print(f"🏆 Theoretical TFT compiled and saved to {final_save_path}")

        print(f"🧹 Cleaning up temporary Lightning checkpoints...")
        checkpoint_folder = Path(f"models/{MODEL_NAME}_ckpt")
        if checkpoint_folder.exists() and checkpoint_folder.is_dir():
            shutil.rmtree(checkpoint_folder)
            print("✨ Cleanup complete!")
        
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"❌ FATAL ERROR: {e}")