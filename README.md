# Advanced Time Series Forecasting: Spain Energy Demand

## Project Overview & Business Value

Predicting national electricity demand with high precision is critical for grid stability and economic efficiency. Inaccurate forecasts result in operators purchasing emergency power at penalizing spot market prices, while overestimations lead to wasted baseload generation.

This repository contains the codebase and research for a Master's level Final Project targeting Spain's National Energy Demand. The core objective is predicting the continuous physical grid load (in Megawatts) while navigating the structural realities of the European energy market. 

**Headline Achievement:** The automated pipeline successfully shattered the official Transmission System Operator (TSO) benchmark on a strictly partitioned, unseen Test Set. By pivoting to a high-frequency continuous learning paradigm, an **Intraday LightGBM** architecture achieved a definitive Test-Set Mean Absolute Error (MAE) of **251.9 MW (0.88% MAPE)**, surpassing the TSO's formidable ceiling of **258.6 MW (0.89% MAPE)**.

## Methodology & "Honest" Forecasting

To ensure absolute operational fidelity and prevent theoretical hallucination, the methodology is strictly constrained by the mathematical definition of physical causality. The pipeline is evaluated across two distinct paradigms:

* **Day-Ahead Horizon ($h=24$):** Mirrors the constraints of the European Day-Ahead market (OMIE). The algorithm predicts 24 consecutive hours of load based entirely on strictly causal data available prior to the operational deadline.
* **Intraday Horizon ($h=1$):** Explores the high-frequency trading and rapid dispatch space. The model ingests fresh SCADA telemetry on a rolling 60-minute interval, dynamically simulating real-time grid balancing.

**Causal Integrity \& Leakage Prevention:**
A core tenet of this repository is "honest forecasting." The pipeline actively guards against the **Auto-Regression Trap** (compounding mathematical hallucinations in sequence models) by implementing a precise Day-Ahead Rolling Origin evaluation with frozen weights. Furthermore, rigorous causal feature engineering guarantees that all weather features, external behavioral covariates (like holidays), and historical lags strictly avoid temporal data leakage (such as incorporating $t=0$ generation data).

## Architectures

Following an exhaustive cross-paradigm ablation study—evaluating Deep Sequences, Tabular Trees, and Classical Statistical frameworks—two distinct champions emerged:

1. **Univariate N-HiTS (Day-Ahead Winner):** Under strict $h=24$ constraints, deep sequential signal decomposition proved computationally superior to mapping uncertain exogenous covariates. N-HiTS achieved an MAE of **1,466.4 MW (5.08% MAPE)**, proving that stable physical grid waveforms offer stronger generalization than volatile meteorological forecasts at long horizons.
2. **Multivariate LightGBM (Intraday Winner):** When the environment shifts to the $h=1$ high-frequency horizon, the architectural hierarchy flips. Granted real-time thermodynamic state data, tree-based algorithms easily compress correlated multidimensional noise. The gradient-boosted LightGBM model dominated this paradigm, culminating in the **251.9 MW** superhuman test-set performance.

## Repository Structure

The repository is logically partitioned between narrative exploration and modular production code.

```text
Final Project/
├── 01_eda_and_preprocessing.ipynb          # Exploratory analysis and structural imputation
├── 02_baselines_and_model_selection.ipynb  # Architecture evaluation and sliding-window backtests
├── 03_operational_evaluation.ipynb         # High-frequency intraday and continuous learning (MLOps)
├── run_all_experiments.sh                  # Automation script for batch orchestrating models
└── src/                                    # Modular Python packages
    ├── data/               # Feature engineering, temporal alignment, and leakage prevention
    ├── tuning/             # Hyperparameter optimization algorithms (Optuna)
    ├── training/           # Model instantiation and Quantile Regression logic
    ├── evaluation/         # Strict causal sliding-window backtesting loops
    ├── mlops/              # Continuous learning and retraining simulation loops
    └── visualization/      # Narrative-driven matplotlib dashboards and diagnostics
```

## Installation & Execution

This project utilizes a centralized Python virtual environment housed in the parent directory (`Time Series/`). 

```bash
# 1. Navigate to the parent directory housing the environment
cd "../" # Assuming you are in the 'Final project' folder, or cd into 'Time Series/'

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Enter the project repository
cd "Final project/"

# 4. (Optional) Run the master execution script to train the entire pipeline
bash run_all_experiments.sh

# 5. Alternatively, launch the Jupyter Notebook narrative
jupyter notebook
```

*Note: For the exact replication of state-space dimensions and recursive lags, run the notebooks chronologically (`01` -> `02` -> `03`).*
