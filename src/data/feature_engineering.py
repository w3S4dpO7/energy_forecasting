# src/feature_engineering.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import holidays

# ─── CORE HONEST FEATURES ────────────────────────────────────────────────────────

def build_prophet_features(master_df: pd.DataFrame) -> pd.DataFrame:
    df = master_df.copy()
    df = df.reset_index().rename(columns={'index': 'ds', 'time': 'ds', 'dt_iso': 'ds', 'total load actual': 'y'})
    
    es_holidays = holidays.Spain(years=df['ds'].dt.year.unique().tolist())
    holiday_series = df['ds'].map(lambda x: x in es_holidays)
    df['is_holiday'] = holiday_series.astype(int)
    
    dayofweek = pd.Series(df['ds'].dt.dayofweek, index=df.index)
    holiday_dates = pd.Series(holiday_series, index=df.index).astype(bool)
    
    is_holiday_tomorrow = holiday_dates.shift(-24).fillna(False).astype(bool)
    is_holiday_yesterday = holiday_dates.shift(24).fillna(False).astype(bool)
    
    df['is_bridge_day'] = (
        ((dayofweek == 0) & is_holiday_tomorrow) | 
        ((dayofweek == 4) & is_holiday_yesterday)
    ).astype(int)
    
    return df[['ds', 'y', 'is_holiday', 'is_bridge_day']].dropna()

def build_tree_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """The master Multivariate Tabular matrix (formerly 'Improved')."""
    df = master_df.copy()
    
    # Mathematical Cyclical Encodings
    hour, dayofweek, dayofyear = df.index.hour, df.index.dayofweek, df.index.dayofyear
    df['hour_sin'], df['hour_cos'] = np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)
    df['dow_sin'], df['dow_cos'] = np.sin(2 * np.pi * dayofweek / 7.0), np.cos(2 * np.pi * dayofweek / 7.0)
    df['doy_sin'], df['doy_cos'] = np.sin(2 * np.pi * dayofyear / 365.25), np.cos(2 * np.pi * dayofyear / 365.25)
    df['is_weekend'] = (dayofweek >= 5).astype(int)
    
    # Holiday & Puente Flags
    es_holidays = holidays.Spain(years=df.index.year.unique().tolist())
    holiday_series = df.index.map(lambda x: x in es_holidays)
    df['is_holiday'] = holiday_series.astype(int)
    
    holiday_dates = pd.Series(holiday_series, index=df.index).astype(bool)
    dayofweek_series = pd.Series(df.index.dayofweek, index=df.index)
    
    df['is_bridge_day'] = (
        ((dayofweek_series == 0) & holiday_dates.shift(-24).fillna(False).astype(bool)) | 
        ((dayofweek_series == 4) & holiday_dates.shift(24).fillna(False).astype(bool))
    ).astype(int)

    # Core Lags & Fuzzy Lags
    df['load_t_24'] = df['total load actual'].shift(24)
    df['load_t_168'] = df['total load actual'].shift(168)
    df['load_t_23'] = df['total load actual'].shift(23)
    df['load_t_25'] = df['total load actual'].shift(25)
    df['load_t_167'] = df['total load actual'].shift(167)
    df['load_t_169'] = df['total load actual'].shift(169)
    
    # 24h Momentum
    prev_day_series = df['total load actual'].shift(24)
    df['rolling_mean_24h'] = prev_day_series.rolling(window=24).mean()
    df['rolling_std_24h']  = prev_day_series.rolling(window=24).std()
    df['rolling_max_24h']  = prev_day_series.rolling(window=24).max()
    df['rolling_min_24h']  = prev_day_series.rolling(window=24).min()
    
    return df.dropna()

def build_intraday_tree_features(master_df: pd.DataFrame) -> pd.DataFrame:
    df = build_tree_features(master_df)
    
    # High-Frequency t-1 Lags for Intraday (h=1)
    df['load_t_1'] = df['total load actual'].shift(1)
    df['load_t_2'] = df['total load actual'].shift(2)
    df['load_t_3'] = df['total load actual'].shift(3)
    df['load_ramp_1h'] = df['load_t_1'] - df['load_t_2']
    
    prev_hour_series = df['total load actual'].shift(1)
    df['rolling_mean_6h'] = prev_hour_series.rolling(window=6).mean()
    df['rolling_max_6h']  = prev_hour_series.rolling(window=6).max()
    df['rolling_min_6h']  = prev_hour_series.rolling(window=6).min()
    
    # Drop Day-Ahead specific lags
    df = df.drop(columns=['load_t_24', 'load_t_168', 'load_t_23', 'load_t_25', 'load_t_167', 'load_t_169'], errors='ignore')
    return df.dropna()

def build_mlp_features(master_df: pd.DataFrame, train_end_date: str = '2017-12-31 23:00:00') -> tuple[pd.DataFrame, StandardScaler]:
    df = build_tree_features(master_df)
    target_col = 'total load actual'
    features = [c for c in df.columns if c != target_col]
    
    train_mask = df.index <= pd.to_datetime(train_end_date)
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    
    x_scaler.fit(df.loc[train_mask, features])
    df[features] = x_scaler.transform(df[features])
    y_scaler.fit(df.loc[train_mask, [target_col]])
    df[target_col] = y_scaler.transform(df[[target_col]])
    
    return df, y_scaler

def build_dl_features(master_df: pd.DataFrame, train_end_date: str = '2017-12-31 23:00:00') -> pd.DataFrame:
    """The master Deep Learning matrix (formerly 'Improved')."""
    df = master_df.copy()
    
    # Fourier Encodings & Holidays
    hour, dayofweek, dayofyear = df.index.hour, df.index.dayofweek, df.index.dayofyear
    df['hour_sin'], df['hour_cos'] = np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)
    df['dow_sin'], df['dow_cos'] = np.sin(2 * np.pi * dayofweek / 7.0), np.cos(2 * np.pi * dayofweek / 7.0)
    df['doy_sin'], df['doy_cos'] = np.sin(2 * np.pi * dayofyear / 365.25), np.cos(2 * np.pi * dayofyear / 365.25)
    
    es_holidays = holidays.Spain(years=df.index.year.unique().tolist())
    holiday_series = df.index.map(lambda x: x in es_holidays)
    df['is_holiday'] = holiday_series.astype(int)
    
    dayofweek_series = pd.Series(df.index.dayofweek, index=df.index)
    holiday_dates = pd.Series(holiday_series, index=df.index).astype(bool)
    
    df['is_bridge_day'] = (
        ((dayofweek_series == 0) & holiday_dates.shift(-24).fillna(False).astype(bool)) | 
        ((dayofweek_series == 4) & holiday_dates.shift(24).fillna(False).astype(bool))
    ).astype(int)
    
    # Scale continuous thermodynamics
    continuous_cols = [c for c in df.columns if any(x in c for x in ['HDD', 'CDD', 'wind_speed', 'humidity', 'cloud_cover_pct'])]
    train_mask = df.index <= pd.to_datetime(train_end_date)
    
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, continuous_cols])
    df[continuous_cols] = scaler.transform(df[continuous_cols])
    
    df = df.reset_index().rename(columns={'index': 'ds', 'time': 'ds', 'total load actual': 'y'})
    df['unique_id'] = 'Spain_Grid'
    return df.dropna()

def build_univariate_dl_features(master_df: pd.DataFrame) -> pd.DataFrame:
    df = master_df[['total load actual']].copy()
    df = df.reset_index().rename(columns={'index': 'ds', 'time': 'ds', 'total load actual': 'y'})
    df['unique_id'] = 'Spain_Grid'
    return df

# ─── SANDBOX LOGIC ─────────────────────────────────────────────────────────────

def _clean_sparse_kaggle_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_drop = [
        'generation fossil coal-derived gas', 'generation fossil oil shale', 
        'generation fossil peat', 'generation geothermal', 
        'generation hydro pumped storage aggregated', 'generation marine', 
        'generation wind offshore', 'forecast wind offshore day ahead', 
        'total load forecast' 
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    sparse_cols = ['generation fossil brown coal / lignite', 'generation hydro pumped storage consumption']
    for col in sparse_cols:
        if col in df.columns:
            df[f'{col}_is_active'] = (df[col] > 0).astype(int)
            df[col] = np.log1p(df[col].fillna(0))
            
    kaggle_cols = [c for c in df.columns if c.startswith('generation') or 'forecast' in c or 'price' in c]
    df[kaggle_cols] = df[kaggle_cols].ffill().fillna(0)
    
    return df

def process_sandbox_a_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_sparse_kaggle_cols(df)
    gen_price_cols = [c for c in df.columns if (c.startswith('generation') or c == 'price actual')]
    for col in gen_price_cols:
        df[f"{col}_t_24"] = df[col].shift(24)
    df = df.drop(columns=gen_price_cols)
    return df.dropna()

def process_sandbox_b_columns(df: pd.DataFrame) -> pd.DataFrame:
    return _clean_sparse_kaggle_cols(df).dropna()

def scale_dl_sandbox_features(df: pd.DataFrame, train_end_date: str) -> pd.DataFrame:
    exclude_from_scaling = ['ds', 'unique_id', 'y', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
                            'doy_sin', 'doy_cos', 'is_holiday', 'is_bridge_day']
    exclude_from_scaling += [c for c in df.columns if '_is_active' in c]
    
    continuous_cols = [c for c in df.columns if c not in exclude_from_scaling]
    train_mask = pd.to_datetime(df['ds']) <= pd.to_datetime(train_end_date)
    
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, continuous_cols])
    df[continuous_cols] = scaler.transform(df[continuous_cols])
    return df.dropna()

# ─── MASTER EXECUTION ──────────────────────────────────────────────────────────

def generate_architectures(master_path: str, output_dir: str):
    master_df_raw = pd.read_csv(master_path, index_col=0, parse_dates=True)
    master_df_raw.index = pd.to_datetime(master_df_raw.index) 
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    master_df_full = _clean_sparse_kaggle_cols(master_df_raw)
    master_df_full = master_df_full.dropna(subset=['total load actual'])

    leakage_cols = [c for c in master_df_full.columns if c.startswith('generation') or 'forecast' in c or 'price' in c]
    master_df_clean = master_df_full.drop(columns=leakage_cols, errors='ignore')

    print("Building Core Matrices (Strictly Leakage-Free)...")
    build_prophet_features(master_df_clean).to_parquet(out_path / "df_prophet.parquet")
    build_intraday_tree_features(master_df_clean).to_parquet(out_path / "df_intraday_trees.parquet")
    build_univariate_dl_features(master_df_clean).to_parquet(out_path / "df_univariate_deeplearning.parquet")
    
    df_mlp, mlp_y_scaler = build_mlp_features(master_df_clean)
    df_mlp.to_parquet(out_path / "df_mlp.parquet")
    joblib.dump(mlp_y_scaler, out_path / "mlp_y_scaler.pkl")
    
    # Kept "improved" in the output filename so your existing checkpoints load perfectly!
    build_tree_features(master_df_clean).to_parquet(out_path / "df_improved_trees.parquet")
    build_dl_features(master_df_clean).to_parquet(out_path / "df_improved_deeplearning.parquet")

    print("\n--- Generating KAGGLE SANDBOX A (Risky/Forecasts) Matrices ---")
    df_sb_a_trees = process_sandbox_a_columns(build_tree_features(master_df_full))
    df_sb_a_trees.to_parquet(out_path / "df_sandbox_a_trees.parquet")
    
    df_sb_a_dl = process_sandbox_a_columns(build_dl_features(master_df_full, '2017-12-31 23:00:00'))
    scale_dl_sandbox_features(df_sb_a_dl, '2017-12-31 23:00:00').to_parquet(out_path / "df_sandbox_a_deeplearning.parquet")

    print("--- Generating KAGGLE SANDBOX B (Cheating/God Mode) Matrices ---")
    df_sb_b_trees = process_sandbox_b_columns(build_tree_features(master_df_full))
    df_sb_b_trees.to_parquet(out_path / "df_sandbox_b_trees.parquet")
    
    df_sb_b_dl = process_sandbox_b_columns(build_dl_features(master_df_full, '2017-12-31 23:00:00'))
    scale_dl_sandbox_features(df_sb_b_dl, '2017-12-31 23:00:00').to_parquet(out_path / "df_sandbox_b_deeplearning.parquet")

    print(f"\nAll model architectures successfully saved to {output_dir}")

if __name__ == "__main__":
    MASTER_PATH = "data/processed/master_df.csv"
    OUTPUT_DIR = "data/model_inputs"
    generate_architectures(MASTER_PATH, OUTPUT_DIR)