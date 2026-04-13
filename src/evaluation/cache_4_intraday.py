import json
import logging
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# ── 1. PATHS & DATES ──
DATA_DIR = Path("data/model_inputs")
MODELS_DIR = Path("models")
CACHE_FILE = MODELS_DIR / "stage4_intraday_preds.pkl"

TRAIN_END = '2017-12-31 23:00:00'
VAL_START = '2018-01-01 00:00:00'
VAL_END   = '2018-06-30 23:00:00'

print("==================================================")
print("🚀 CACHING STAGE 4: INTRADAY FROZEN BASELINES")
print("==================================================\n")

# ── 2. LIGHTGBM DATA PREP ──
df_t1 = pd.read_parquet(DATA_DIR / "df_intraday_trees.parquet", engine="fastparquet")
target_col = 'total load actual'
train_t1 = df_t1.loc[:TRAIN_END]
val_t1   = df_t1.loc[VAL_START:VAL_END]

X_train_multi = train_t1.drop(columns=[target_col])
y_train       = train_t1[target_col].values
X_val_multi   = val_t1.drop(columns=[target_col])
y_val_pandas  = val_t1[target_col].values # Explicit Pandas actuals

allowed_prefixes = ['total load actual', 'load_', 'rolling_']
uni_cols = [c for c in df_t1.columns if any(c.startswith(p) for p in allowed_prefixes) and c != target_col]
X_train_uni = train_t1[uni_cols]
X_val_uni   = val_t1[uni_cols]

# ── 3. EVALUATE LIGHTGBM MULTIVARIATE ──
print("⚙️ Evaluating LightGBM (Multivariate)...")
with open(MODELS_DIR / "lgb_intraday_params.json", 'r') as f: 
    lgb_params_multi = json.load(f)
lgb_params_multi.update({"random_state": 42, "n_jobs": -1, "verbose": -1})

model_multi = lgb.LGBMRegressor(**lgb_params_multi)
model_multi.fit(X_train_multi, y_train)
y_pred_multi = model_multi.predict(X_val_multi)

# ── 4. EVALUATE LIGHTGBM UNIVARIATE ──
print("⚙️ Evaluating LightGBM (Univariate)...")
with open(MODELS_DIR / "lgb_uni_intraday_params.json", 'r') as f: 
    lgb_params_uni = json.load(f)
lgb_params_uni.update({"random_state": 42, "n_jobs": -1, "verbose": -1})

model_uni = lgb.LGBMRegressor(**lgb_params_uni)
model_uni.fit(X_train_uni, y_train)
y_pred_uni = model_uni.predict(X_val_uni)

# ── 5. EVALUATE N-HITS UNIVARIATE ──
print("⏳ Evaluating N-HiTS (Univariate) - This takes ~2.5 minutes...")
df_dl = pd.read_parquet(DATA_DIR / "df_univariate_deeplearning.parquet", engine="fastparquet")
df_dl = df_dl.set_index('ds')
df_dl = df_dl[~df_dl.index.duplicated()].asfreq('h').ffill().reset_index()
df_dl = df_dl[df_dl['ds'] <= VAL_END]

target_ts = TimeSeries.from_dataframe(df_dl, time_col='ds', value_cols='y')
split_train_val = pd.Timestamp(TRAIN_END)
train_series, _ = target_ts.split_after(split_train_val)

target_scaler = Scaler()
target_scaler.fit(train_series)

nhits_model = NHiTSModel.load(str(MODELS_DIR / "nhits_univariate_intraday.pt"))

preds_scaled = nhits_model.historical_forecasts(
    series=target_scaler.transform(target_ts),
    start=pd.Timestamp(VAL_START),
    forecast_horizon=1,
    stride=1,
    retrain=False,
    verbose=False
)

preds_ts = target_scaler.inverse_transform(preds_scaled)
y_pred_nhits = preds_ts.values().flatten()
y_val_darts = target_ts.slice_intersect(preds_ts).values().flatten() # Explicit Darts actuals

# ── 6. CACHE DECOUPLED RESULTS ──
results = {
    "lgbm_actuals": y_val_pandas,
    "lgbm_multi": y_pred_multi,
    "lgbm_uni": y_pred_uni,
    "nhits_actuals": y_val_darts,
    "nhits_uni": y_pred_nhits
}

joblib.dump(results, CACHE_FILE)
print(f"\n✅ Predictions successfully decoupled and cached to {CACHE_FILE}")