#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=================================================="
echo "🧹 INITIATING HARD CLEANUP (REPRODUCIBLE MODE)"
echo "=================================================="

# 1. Wipe ALL old Optuna SQLite databases in the root folder
echo "Cleaning up all Optuna databases..."
rm -f *.db

# 2. Wipe temporary PyTorch Lightning checkpoint directories
echo "Removing orphaned training checkpoints..."
if [ -d "models/tuning_checkpoints" ]; then
    rm -rf models/tuning_checkpoints/*
fi
# Remove any TFT marathon checkpoint folders
rm -rf models/*_ckpt

# 3. Wipe previously generated champion models to guarantee a fresh start
echo "Deleting old model artifacts..."
rm -f models/prophet_champion.json
rm -f models/mlp_champion.pkl
rm -f models/tabnet_champion.zip
rm -f models/lgb_*.txt
rm -f models/nhits_*.pt*
rm -f models/tft_*.pt*

echo "✨ Cleanup Complete! Starting with a completely blank slate."
echo ""

echo "=================================================="
echo "⚙️ STAGE 0: DATA PREPARATION & FEATURE ENGINEERING"
echo "=================================================="
uv run python -m src.data_preprocessing
uv run python -m src.feature_engineering

echo ""
echo "=================================================="
echo "🚀 STAGE 1: STATISTICAL BASELINES"
echo "=================================================="
uv run python -m src.tune_prophet

echo ""
echo "=================================================="
echo "🚀 STAGE 2: TABULAR NEURAL NETWORKS (INITIAL ONLY)"
echo "=================================================="
uv run python -m src.tune_mlp
uv run python -m src.tune_tabnet

echo ""
echo "=================================================="
echo "🚀 STAGE 3: GRADIENT BOOSTING (THE CHAMPION)"
echo "=================================================="
uv run python -m src.tune_lightgbm --mode initial
uv run python -m src.tune_lightgbm --mode improved
uv run python -m src.tune_lightgbm --mode sandbox

echo ""
echo "=================================================="
echo "🚀 STAGE 4: SEQUENCE DL - N-HiTS (THE ABLATION)"
echo "=================================================="
uv run python -m src.tune_nhits --mode univariate
uv run python -m src.tune_nhits_long --mode univariate
uv run python -m src.tune_nhits --mode initial
uv run python -m src.tune_nhits --mode improved

echo ""
echo "=================================================="
echo "🚀 STAGE 5: SEQUENCE DL - TFT (THE ORACLE)"
echo "=================================================="
uv run python -m src.tune_tft --mode initial
uv run python -m src.tune_tft --mode improved
uv run python -m src.tune_tft --mode sandbox

echo "=================================================="
echo "🏆 ALL EXPERIMENTS COMPLETED SUCCESSFULLY! 🏆"
echo "=================================================="