# src/feature_engineering.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

import holidays # Ensure this is imported at the top of the script

def build_prophet_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """Prepares additive curve-fitting format natively suited for Prophet (Target + Holidays only)."""
    df = master_df.copy()
    
    # Prophet requires 'ds' and 'y'
    df = df.reset_index().rename(columns={
        'index': 'ds', 'time': 'ds', 'dt_iso': 'ds',
        'total load actual': 'y'
    })
    
    # ── Add Spanish National Holidays & "Puentes" ──
    es_holidays = holidays.Spain(years=df['ds'].dt.year.unique().tolist())
    holiday_series = df['ds'].map(lambda x: x in es_holidays)
    df['is_holiday'] = holiday_series.astype(int)
    
    # Bridge Day Logic (Puente)
    dayofweek = df['ds'].dt.dayofweek
    holiday_dates = pd.Series(holiday_series, index=df.index)
    is_holiday_tomorrow = holiday_dates.shift(-24).fillna(False)
    is_holiday_yesterday = holiday_dates.shift(24).fillna(False)
    
    is_puente = (((dayofweek == 0) & is_holiday_tomorrow) | 
                 ((dayofweek == 4) & is_holiday_yesterday))
    df['is_bridge_day'] = is_puente.astype(int)
    
    # Isolate strictly to what Prophet needs
    return df[['ds', 'y', 'is_holiday', 'is_bridge_day']].dropna()

def build_tree_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """Prepares flat format for LightGBM/XGBoost with lags and cyclical encoding."""
    df = master_df.copy()
    
    # Cyclical Calendar
    hour = df.index.hour
    dayofweek = df.index.dayofweek
    dayofyear = df.index.dayofyear
    
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['dow_sin'] = np.sin(2 * np.pi * dayofweek / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * dayofweek / 7.0)
    df['doy_sin'] = np.sin(2 * np.pi * dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * dayofyear / 365.25)
    df['is_weekend'] = (dayofweek >= 5).astype(int)
    
    # Autoregressive Lags (Memory)
    df['load_t_24'] = df['total load actual'].shift(24)
    df['load_t_168'] = df['total load actual'].shift(168)
    
    # Drop rows introduced by the 168-hour shift
    return df.dropna()

import pandas as pd
import numpy as np
import holidays

def build_improved_tree_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """Prepares flat format for LightGBM/XGBoost with advanced lags, rolling stats, and holidays."""
    df = master_df.copy()
    
    # ── 1. Cyclical Calendar ──
    hour = df.index.hour
    dayofweek = df.index.dayofweek
    dayofyear = df.index.dayofyear
    
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['dow_sin'] = np.sin(2 * np.pi * dayofweek / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * dayofweek / 7.0)
    df['doy_sin'] = np.sin(2 * np.pi * dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * dayofyear / 365.25)
    df['is_weekend'] = (dayofweek >= 5).astype(int)
    
    # ── 2. Spanish National Holidays & "Puentes" ──
    # Instantiate the Spanish holiday calendar for the years in our dataset
    es_holidays = holidays.Spain(years=df.index.year.unique().tolist())
    
    # Map index to check if it's a holiday
    holiday_series = df.index.map(lambda x: x in es_holidays)
    df['is_holiday'] = holiday_series.astype(int)
    
    # Bridge Day Logic (Puente):
    # A Monday (dow=0) is a puente if Tuesday is a holiday.
    # A Friday (dow=4) is a puente if Thursday is a holiday.
    holiday_dates = pd.Series(holiday_series, index=df.index)
    is_holiday_tomorrow = holiday_dates.shift(-24).fillna(False)
    is_holiday_yesterday = holiday_dates.shift(24).fillna(False)
    
    is_puente = (((dayofweek == 0) & is_holiday_tomorrow) | 
                 ((dayofweek == 4) & is_holiday_yesterday))
    df['is_bridge_day'] = is_puente.astype(int)

    # ── 3. Autoregressive Lags (Memory & Differentiation Clusters) ──
    # The 24-hour cluster (Yesterday's exact curve slope)
    df['load_t_23'] = df['total load actual'].shift(23)
    df['load_t_24'] = df['total load actual'].shift(24) # The Day-Ahead Anchor
    df['load_t_25'] = df['total load actual'].shift(25)
    
    # The 168-hour cluster (Last week's exact curve slope)
    df['load_t_167'] = df['total load actual'].shift(167)
    df['load_t_168'] = df['total load actual'].shift(168) # The Weekly Anchor
    df['load_t_169'] = df['total load actual'].shift(169)
    
    # ── 4. Rolling Window Statistics (Strictly Leakage-Free) ──
    # CRITICAL: We cannot run rolling stats on the target column directly, or we will leak today's data into tomorrow's forecast.
    # We must run the rolling window ON the t-24 series. 
    # This represents the "daily summary" of the previous known 24 hours.
    prev_day_series = df['total load actual'].shift(24)
    
    df['rolling_mean_24h'] = prev_day_series.rolling(window=24).mean()
    df['rolling_std_24h']  = prev_day_series.rolling(window=24).std() # Volatility
    df['rolling_max_24h']  = prev_day_series.rolling(window=24).max()
    df['rolling_min_24h']  = prev_day_series.rolling(window=24).min()
    
    # Drop rows introduced by the longest shifts and rolling windows
    return df.dropna()

"""
def build_prophet_features(master_df: pd.DataFrame) -> pd.DataFrame:
    # Prepares additive curve-fitting format for Prophet.
    df = master_df.copy()
    
    # Prophet requires 'ds' and 'y'
    df = df.reset_index().rename(columns={
        'index': 'ds', # or 'time' depending on how it loaded
        'time': 'ds',
        'dt_iso': 'ds',
        'total load actual': 'y'
    })
    
    # Tree features (lags, sine/cosine) break Prophet's internal Fourier series.
    # The master_df does not have them yet, so we just return the raw DataFrame 
    # ready for add_regressor() calls.
    return df
    """

def build_mlp_features(master_df: pd.DataFrame, train_end_date: str = '2017-12-31') -> tuple[pd.DataFrame, StandardScaler]:
    """Prepares strictly scaled tabular format for MLP/TabNet, scaling BOTH features and target."""
    df = build_tree_features(master_df)
    
    target_col = 'total load actual'
    features = [c for c in df.columns if c != target_col]
    
    train_mask = df.index <= pd.to_datetime(train_end_date)
    
    # 1. Scale Features (X)
    x_scaler = StandardScaler()
    x_scaler.fit(df.loc[train_mask, features])
    df[features] = x_scaler.transform(df[features])
    
    # 2. Scale Target (y)
    y_scaler = StandardScaler()
    y_scaler.fit(df.loc[train_mask, [target_col]])
    df[target_col] = y_scaler.transform(df[[target_col]])
    
    return df, y_scaler

def build_dl_features(master_df: pd.DataFrame, train_end_date: str = '2017-12-31') -> pd.DataFrame:
    """Prepares 3D Tensor Dataloader format for TFT / N-HiTS."""
    df = master_df.copy()
    
    # Raw categories for internal embedding layers
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    
    # Identify continuous columns to standardize (Weather only)
    continuous_cols = [c for c in df.columns if any(x in c for x in ['HDD', 'CDD', 'wind_speed', 'humidity', 'cloud_cover_pct'])]
    
    # To prevent look-ahead bias, fit StandardScaler ONLY on the Train set
    train_mask = df.index <= pd.to_datetime(train_end_date)
    
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, continuous_cols])
    
    # Apply transformation to the entire dataset safely
    df[continuous_cols] = scaler.transform(df[continuous_cols])
    
    # Ensure a unique ID column exists for NeuralForecast/PyTorch Forecasting structures
    df = df.reset_index().rename(columns={'index': 'ds', 'time': 'ds'})
    df['unique_id'] = 'Spain_Grid'
    
    # Leave target raw! Darts Scaler handles this dynamically.
    df = df.rename(columns={'total load actual': 'y'})
    
    return df

def build_improved_dl_features(master_df: pd.DataFrame, train_end_date: str = '2017-12-31') -> pd.DataFrame:
    """Prepares 3D Tensor Dataloader format for TFT / N-HiTS."""
    df = master_df.copy()
    
    # 1. Cyclical Calendar (Required for smooth gradients)
    hour = df.index.hour
    dayofweek = df.index.dayofweek
    dayofyear = df.index.dayofyear
    
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['dow_sin'] = np.sin(2 * np.pi * dayofweek / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * dayofweek / 7.0)
    df['doy_sin'] = np.sin(2 * np.pi * dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * dayofyear / 365.25)
    
    # 2. Future Covariates (Holidays & Puentes)
    es_holidays = holidays.Spain(years=df.index.year.unique().tolist())
    holiday_series = df.index.map(lambda x: x in es_holidays)
    df['is_holiday'] = holiday_series.astype(int)
    
    holiday_dates = pd.Series(holiday_series, index=df.index)
    is_holiday_tomorrow = holiday_dates.shift(-24).fillna(False)
    is_holiday_yesterday = holiday_dates.shift(24).fillna(False)
    
    df['is_bridge_day'] = (((dayofweek == 0) & is_holiday_tomorrow) | 
                           ((dayofweek == 4) & is_holiday_yesterday)).astype(int)
    
    # 3. Scale only raw continuous weather columns 
    # (Sine/Cosine are already -1 to 1; Binary flags are 0 to 1)
    continuous_cols = [c for c in df.columns if any(x in c for x in ['HDD', 'CDD', 'wind_speed', 'humidity', 'cloud_cover_pct'])]
    
    train_mask = df.index <= pd.to_datetime(train_end_date)
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, continuous_cols])
    df[continuous_cols] = scaler.transform(df[continuous_cols])
    
    df = df.reset_index().rename(columns={'index': 'ds', 'time': 'ds'})
    df['unique_id'] = 'Spain_Grid'
    df = df.rename(columns={'total load actual': 'y'})
    
    # Drop rows at the beginning missing the shift data for puentes
    return df.dropna()

def process_kaggle_sandbox_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to apply Kaggle sandbox rules (Leakage + Day-Ahead Forecasts)."""
    df = df.copy()
    
    # 1. Drop completely empty/useless columns & the TSO benchmark
    cols_to_drop = [
        'generation fossil coal-derived gas', 'generation fossil oil shale', 
        'generation fossil peat', 'generation geothermal', 
        'generation hydro pumped storage aggregated', 'generation marine', 
        'generation wind offshore', 'forecast wind offshore day ahead', 
        'total load forecast' # The benchmark to beat
    ]
    # Drop only if they exist to avoid KeyError
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # 2. Handle Sparse Data (Brown Coal & Pumped Storage Consumption)
    sparse_cols = ['generation fossil brown coal / lignite', 'generation hydro pumped storage consumption']
    for col in sparse_cols:
        if col in df.columns:
            # Create binary flag
            df[f'{col}_is_active'] = (df[col] > 0).astype(int)
            # Apply log(1+x) to the original to handle massive discrepancies
            df[col] = np.log1p(df[col].fillna(0))
            
    return df

def build_sandbox_tree_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """Prepares LightGBM format including all Kaggle Sandbox leaked/forecast columns."""
    # Start with our best tree features (Calendar, Holidays, Lags, Rolling)
    df = build_improved_tree_features(master_df)
    
    # Process the extra Kaggle columns
    df = process_kaggle_sandbox_columns(df)
    
    return df.dropna()

def build_sandbox_dl_features(master_df: pd.DataFrame, train_end_date: str = '2017-12-31') -> pd.DataFrame:
    """Prepares Deep Learning format including all Kaggle Sandbox leaked/forecast columns."""
    # Start with the baseline cyclical and holiday features
    # NOTE: This function returns a df where the index is reset and the dates are in 'ds'
    df = build_improved_dl_features(master_df, train_end_date)
    
    # Process the extra Kaggle columns
    df = process_kaggle_sandbox_columns(df)
    
    # Identify ALL continuous columns that need Z-Score scaling now
    # We exclude our target ('y'), IDs, binary/cyclical flags, and the 'ds' column itself
    exclude_from_scaling = ['ds', 'unique_id', 'y', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
                            'doy_sin', 'doy_cos', 'is_holiday', 'is_bridge_day']
    exclude_from_scaling += [c for c in df.columns if '_is_active' in c]
    
    continuous_cols = [c for c in df.columns if c not in exclude_from_scaling]
    
    # ── THE FIX ──
    # Since build_improved_dl_features already reset the index, the dates are now in the 'ds' column.
    # Convert 'ds' back to datetime just for the mask comparison.
    train_mask = pd.to_datetime(df['ds']) <= pd.to_datetime(train_end_date)
    
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, continuous_cols])
    
    # Apply transformation
    df[continuous_cols] = scaler.transform(df[continuous_cols])
    
    return df.dropna()

def build_univariate_dl_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares a clean univariate format for N-HiTS.
    Removes all weather, generation, and calendar features.
    """
    df = master_df.copy()
    
    # Keep only the target
    df = df[['total load actual']]
    
    # Standard Darts formatting
    df = df.reset_index().rename(columns={'index': 'ds', 'time': 'ds', 'total load actual': 'y'})
    df['unique_id'] = 'Spain_Grid'
    
    return df

def generate_architectures(master_path: str, output_dir: str):
    """Executes the branching logic to create all required formats."""
    # Ensure datetime index is parsed
    master_df = pd.read_csv(master_path, index_col=0, parse_dates=True)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Building Prophet matrix...")
    df_prophet = build_prophet_features(master_df)
    df_prophet.to_parquet(out_path / "df_prophet.parquet")

    print("Building LightGBM/XGBoost matrix...")
    df_trees = build_tree_features(master_df)
    df_trees.to_parquet(out_path / "df_trees.parquet")
    
    print("Building LightGBM/XGBoost matrix...")
    df_trees = build_improved_tree_features(master_df)
    df_trees.to_parquet(out_path / "df_improved_trees.parquet")

    print("Building MLP/TabNet (Tabular Scaled) matrix...")
    df_mlp, mlp_y_scaler = build_mlp_features(master_df, train_end_date='2017-12-31 23:00:00')
    df_mlp.to_parquet(out_path / "df_mlp.parquet")
    # Export the target scaler for inference un-scaling
    joblib.dump(mlp_y_scaler, out_path / "mlp_y_scaler.pkl")
    
    print("Building Deep Learning (TFT/N-HiTS) matrix...")
    # Prevents look-ahead bias by standardizing only on pre-2018 data
    df_dl = build_dl_features(master_df, train_end_date='2017-12-31 23:00:00')
    df_dl.to_parquet(out_path / "df_deeplearning.parquet")

    print("Building Deep Learning (TFT/N-HiTS) matrix...")
    # Prevents look-ahead bias by standardizing only on pre-2018 data
    df_dl = build_improved_dl_features(master_df, train_end_date='2017-12-31 23:00:00')
    df_dl.to_parquet(out_path / "df_improved_deeplearning.parquet")

    print("\n--- Generating KAGGLE SANDBOX (Leakage) Matrices ---")
    print("Building Sandbox LightGBM matrix...")
    df_sandbox_trees = build_sandbox_tree_features(master_df)
    df_sandbox_trees.to_parquet(out_path / "df_sandbox_trees.parquet")
    
    print("Building Sandbox Deep Learning matrix...")
    df_sandbox_dl = build_sandbox_dl_features(master_df, train_end_date='2017-12-31 23:00:00')
    df_sandbox_dl.to_parquet(out_path / "df_sandbox_deeplearning.parquet")

    print("Building Univariate N-HiTS matrix...")
    df_univariate = build_univariate_dl_features(master_df)
    df_univariate.to_parquet(out_path / "df_univariate_deeplearning.parquet")
    
    print(f"All model architectures successfully saved to {output_dir}")

if __name__ == "__main__":
    MASTER_PATH = "data/processed/master_df.csv"
    OUTPUT_DIR = "data/model_inputs"
    
    generate_architectures(MASTER_PATH, OUTPUT_DIR)