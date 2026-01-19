import pandas as pd
import os
import sys
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandera as pa
from pandera import Column, Check
from typing import Union, IO, List, Optional

# --- Configuration & Constants ---
@dataclass
class PipelineConfig:
    FILE_PATH: str = r'c:\Users\Elijay\Downloads\iot_dataset.csv'
    TEMP_THRESHOLD_HIGH: float = 30.0
    BATTERY_THRESHOLD_LOW: float = 15.0
    LOG_FILE: str = 'pipeline.log'

CONF = PipelineConfig()

# --- Schema Definition ---
SENSOR_SCHEMA = pa.DataFrameSchema({
    "timestamp": Column(pd.Timestamp),
    "device_id": Column(str),
    "temperature_celsius": Column(float, [Check.ge(-20), Check.le(50)]),
    "humidity_percent": Column(float, [Check.ge(0), Check.le(100)]),
    "pressure_hpa": Column(float),
    "battery_level_percent": Column(float, [Check.ge(0), Check.le(100)]),
}, coerce=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONF.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_data(file_path: Union[str, IO]) -> pd.DataFrame:
    """
    Step 1: Extract
    Loads data from CSV file (path or buffer) with error handling.
    """
    if isinstance(file_path, str):
        if not os.path.exists(file_path):
            logging.error(f"File not found at: {file_path}")
            logging.error("Please ensure the dataset is in the correct directory.")
            sys.exit(1)
    
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Failed to read CSV: {e}")
        sys.exit(1)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Clean
    Handles missing values and ensures correct data types.
    """
    logging.info("Starting data cleaning...")
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Identify missing values
    logging.info("Missing values before cleaning:\n" + str(df[['temperature_celsius', 'humidity_percent']].isnull().sum()))

    # Sort by device and time to ensure interpolation makes sense
    df = df.sort_values(by=['device_id', 'timestamp'])
    
    # Interpolate missing sensor readings (linear method for time series)
    # This fills gaps based on the previous and next known values
    # Justification: Interpolation is chosen over mean/median because sensor data 
    # is continuous time-series data; local trends are more accurate than global averages.
    cols_to_interpolate = ['temperature_celsius', 'humidity_percent', 'pressure_hpa']
    for col in cols_to_interpolate:
        if col in df.columns:
            df[col] = df.groupby('device_id')[col].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )

    # Drop rows where critical data might still be missing
    initial_count = len(df)
    df.dropna(inplace=True)
    dropped_count = initial_count - len(df)
    
    if dropped_count > 0:
        print(f"[Info] Dropped {dropped_count} rows that could not be recovered.")
        
    # Validate Schema with Pandera
    # This ensures that the cleaned data strictly adheres to our quality contract
    logging.info("Validating data schema with Pandera...")
    try:
        df = SENSOR_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        logging.error(f"Schema validation failed: {err}")
        raise
        
    return df

def transform_data(df):
    """
    Step 3: Transform
    Adds derived metrics and aggregations.
    """
    print("[Info] Transforming data...")
    
    # Unit Conversion
    df['temperature_fahrenheit'] = (df['temperature_celsius'] * 9/5) + 32
    
    # Extract hour of day
    df['hour_of_day'] = df['timestamp'].dt.hour
    
    # Calculate 3-point rolling average for temperature per device
    df['temp_rolling_avg_3'] = df.groupby('device_id')['temperature_celsius'].transform(
        lambda x: x.rolling(window=3).mean()
    )
    
    # Calculate 'valid_until' column for validity windows
    # This represents the timestamp when the current reading is superseded by the next one
    df['valid_until'] = df.groupby('device_id')['timestamp'].shift(-1)
    
    return df

def analyze_anomalies(df: pd.DataFrame, temp_threshold: Optional[float] = None, battery_threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Step 4: Analyze
    Detects anomalies based on defined thresholds.
    """
    logging.info("Running anomaly detection...")
    
    # Determine thresholds (use arguments if provided, else default to config)
    limit_temp = temp_threshold if temp_threshold is not None else CONF.TEMP_THRESHOLD_HIGH
    limit_battery = battery_threshold if battery_threshold is not None else CONF.BATTERY_THRESHOLD_LOW

    # Define conditions
    high_temp = df['temperature_celsius'] > limit_temp
    low_battery = df['battery_level_percent'] < limit_battery
    
    # Create 'anomaly_type' column
    df['anomaly_type'] = 'Normal'
    df.loc[high_temp, 'anomaly_type'] = 'High Temperature'
    df.loc[low_battery, 'anomaly_type'] = 'Low Battery'
    df.loc[high_temp & low_battery, 'anomaly_type'] = 'High Temp & Low Battery'
    
    # Create 'is_anomaly' boolean column
    df['is_anomaly'] = (high_temp | low_battery)
    
    anomalies = df[df['is_anomaly']]
    
    if not anomalies.empty:
        logging.warning(f"Detected {len(anomalies)} anomalies!")
        logging.warning("\n" + str(anomalies[['timestamp', 'device_id', 'temperature_celsius', 'battery_level_percent', 'anomaly_type']]))
    else:
        logging.info("No anomalies detected. Systems nominal.")
        
    return anomalies

def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5: Aggregate
    Calculates daily averages and minimums.
    """
    logging.info("Aggregating daily statistics...")
    
    daily_stats = df.groupby(['device_id', df['timestamp'].dt.date]).agg({
        'temperature_celsius': 'mean',
        'humidity_percent': 'mean',
        'battery_level_percent': 'min'
    }).reset_index()
    
    daily_stats.rename(columns={'timestamp': 'date', 'temperature_celsius': 'daily_avg_temp', 'humidity_percent': 'daily_avg_humidity', 'battery_level_percent': 'daily_min_battery'}, inplace=True)
    
    logging.info("Daily Aggregated Statistics:\n" + str(daily_stats))
    
    return daily_stats

def visualize_data(daily_stats: pd.DataFrame, save_to_disk: bool = True) -> List[plt.Figure]:
    """
    Step 6: Visualize
    Plots daily average temperature per device.
    """
    logging.info("Generating visualization...")
    figs = []
    
    fig1 = plt.figure(figsize=(10, 6))
    
    for device_id in daily_stats['device_id'].unique():
        subset = daily_stats[daily_stats['device_id'] == device_id]
        plt.plot(subset['date'], subset['daily_avg_temp'], marker='o', label=device_id)
        
    # Add horizontal red line for high temperature threshold
    plt.axhline(y=CONF.TEMP_THRESHOLD_HIGH, color='r', linestyle='--', label=f'High Temp Threshold ({CONF.TEMP_THRESHOLD_HIGH}°C)')
        
    plt.title('Daily Average Temperature by Device')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_to_disk:
        output_file = 'daily_temperature_plot.png'
        plt.savefig(output_file)
        logging.info(f"Visualization saved to {output_file}")
    figs.append(fig1)

    # Plot 2: Daily Minimum Battery Level
    fig2 = plt.figure(figsize=(10, 6))
    
    for device_id in daily_stats['device_id'].unique():
        subset = daily_stats[daily_stats['device_id'] == device_id]
        plt.plot(subset['date'], subset['daily_min_battery'], marker='s', linestyle='--', label=device_id)
        
    # Add horizontal red line for low battery threshold
    plt.axhline(y=CONF.BATTERY_THRESHOLD_LOW, color='r', linestyle=':', label=f'Low Battery Threshold ({CONF.BATTERY_THRESHOLD_LOW}%)')
        
    plt.title('Daily Minimum Battery Level by Device')
    plt.xlabel('Date')
    plt.ylabel('Battery Level (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_to_disk:
        output_file_battery = 'daily_battery_plot.png'
        plt.savefig(output_file_battery)
        logging.info(f"Battery visualization saved to {output_file_battery}")
    figs.append(fig2)
    
    return figs

def load_data_to_storage(df: pd.DataFrame):
    """
    Step 7: Load
    Saves the processed data to partitioned Parquet files.
    """
    logging.info("Loading data to storage (Parquet)...")
    
    # Create a date column for partitioning
    df_storage = df.copy()
    df_storage['date'] = df_storage['timestamp'].dt.date.astype(str)
    
    output_dir = 'processed_data'
    
    try:
        df_storage.to_parquet(output_dir, partition_cols=['date'], index=False)
        logging.info(f"Data saved to '{output_dir}' directory (Partitioned by Date)")
    except Exception as e:
        logging.error(f"Failed to save Parquet file: {e}")

def main():
    # 1. Extract
    df_raw = load_data(CONF.FILE_PATH)
    
    # 2. Clean
    df_clean = clean_data(df_raw)
    
    # 3. Transform
    df_transformed = transform_data(df_clean)
    
    # 4. Analyze / Load
    analyze_anomalies(df_transformed)
    
    # 5. Aggregate
    daily_stats = aggregate_data(df_transformed)
    
    # 6. Visualize
    visualize_data(daily_stats)
    
    # 7. Load
    load_data_to_storage(df_transformed)
    
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()