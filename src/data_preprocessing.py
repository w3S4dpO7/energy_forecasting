# src/data_processing.py
import pandas as pd
import numpy as np
from pathlib import Path

def process_energy_base(energy_path: str) -> pd.DataFrame:
    """Extracts target and enforces chronologically sorted index."""
    df = pd.read_csv(energy_path)
    
    # Standardize index
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Europe/Madrid').dt.tz_localize(None)
    df = df.set_index('time').sort_index()
    
    # Keep only the target, isolate from all leakage (generation/forecast/price)
    target_df = df[['total load actual']].copy()
    
    return target_df

def process_weather_pipeline(weather_path: str) -> pd.DataFrame:
    """Executes the Honors-Level Weather Pipeline (Thermodynamics, Flags, Pivot)."""
    df = pd.read_csv(weather_path)
    
    # Standardize time format to match energy
    df['dt_iso'] = pd.to_datetime(df['dt_iso'], utc=True).dt.tz_convert('Europe/Madrid').dt.tz_localize(None)
    
    # Deduplicate early
    df = df.drop_duplicates(subset=['dt_iso', 'city_name'], keep='first')
    
    # 1. Thermodynamics (Base 18 C)
    df['temp_c'] = df['temp'] - 273.15
    df['HDD'] = np.maximum(18.0 - df['temp_c'], 0)
    df['CDD'] = np.maximum(df['temp_c'] - 18.0, 0)
    
    # 2. Behavioral Flags
    desc = df['weather_description'].str.lower()
    df['is_rain'] = desc.str.contains('rain|drizzle|shower', na=False).astype(int)
    df['is_snow'] = desc.str.contains('snow|sleet', na=False).astype(int)
    df['is_extreme'] = desc.str.contains('thunderstorm|squall|heavy|very heavy', na=False).astype(int)
    df['is_obscured'] = desc.str.contains('fog|mist|haze|dust|smoke|sand', na=False).astype(int)
    
    # 3. Clean Continuous Outliers
    df['wind_speed'] = df['wind_speed'].clip(upper=25)
    df['humidity'] = df['humidity'].clip(upper=100)
    df['cloud_cover_pct'] = df['clouds_all'] / 100.0
    
    # 4. Pivot Cities and Aggregate Flags
    continuous_cols = ['HDD', 'CDD', 'wind_speed', 'humidity', 'cloud_cover_pct']
    flag_cols = ['is_rain', 'is_snow', 'is_extreme', 'is_obscured']
    
    # Pivot continuous
    df_continuous = df.pivot(index='dt_iso', columns='city_name', values=continuous_cols)
    # Flatten hierarchical columns (e.g., 'HDD_Madrid')
    df_continuous.columns = [f"{col[0]}_{col[1].replace(' ', '_')}" for col in df_continuous.columns]
    
    # Aggregate flags nationally (max logic: if 1 anywhere, then 1)
    df_flags = df.groupby('dt_iso')[flag_cols].max()
    
    weather_final = pd.concat([df_continuous, df_flags], axis=1)
    return weather_final

def build_master_dataset(energy_path: str, weather_path: str, output_path: str):
    """Merges processed data and saves the master structural file."""
    print("Processing energy base...")
    energy_clean = process_energy_base(energy_path)
    
    print("Processing weather pipeline...")
    weather_clean = process_weather_pipeline(weather_path)
    
    print("Merging master dataset...")
    master_df = energy_clean.join(weather_clean, how='inner')
    master_df = master_df.dropna(subset=['total load actual'])
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(output_path)
    print(f"Master dataset saved to {output_path} with shape {master_df.shape}")

if __name__ == "__main__":
    # Adjust paths according to your repository structure
    RAW_ENERGY = "archive/energy_dataset.csv"
    RAW_WEATHER = "archive/weather_features.csv"
    OUTPUT_MASTER = "data/processed/master_df.csv"
    
    build_master_dataset(RAW_ENERGY, RAW_WEATHER, OUTPUT_MASTER)