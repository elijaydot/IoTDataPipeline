import pandas as pd
import os
import sys
import logging
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from dataclasses import dataclass
import pandera as pa
from pandera import Column, Check
from typing import Union, IO, List, Optional

# --- Configuration & Constants (from iot_data_pipeline.py) ---
@dataclass
class PipelineConfig:
    FILE_PATH: str = r'c:\Users\Elijay\Downloads\iot_dataset.csv' # Kept for reference, but dashboard uses uploader
    TEMP_THRESHOLD_HIGH: float = 30.0
    BATTERY_THRESHOLD_LOW: float = 15.0
    LOG_FILE: str = 'pipeline.log'

CONF = PipelineConfig()

# --- Schema Definition (from iot_data_pipeline.py) ---
SENSOR_SCHEMA = pa.DataFrameSchema({
    "timestamp": Column(pd.Timestamp),
    "device_id": Column(str),
    "temperature_celsius": Column(float, [Check.ge(-20), Check.le(50)]),
    "humidity_percent": Column(float, [Check.ge(0), Check.le(100)]),
    "pressure_hpa": Column(float),
    "battery_level_percent": Column(float, [Check.ge(0), Check.le(100)]),
}, coerce=True)

# --- Logging Setup (from iot_data_pipeline.py) ---
# Note: Streamlit has its own logging, but this can be useful for debugging file-based operations.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONF.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Pipeline Functions (from iot_data_pipeline.py) ---

@st.cache_data
def load_data(file_path: Union[str, IO]) -> pd.DataFrame:
    """
    Step 1: Extract
    Loads data from CSV file (path or buffer) with error handling.
    (Used by original CLI script, not directly by dashboard uploader)
    """
    if isinstance(file_path, str):
        if not os.path.exists(file_path):
            msg = f"File not found at: {file_path}"
            logging.error(msg)
            raise FileNotFoundError(msg)
    
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        msg = f"Failed to read CSV: {e}"
        logging.error(msg)
        raise ValueError(msg) from e

@st.cache_data
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
    target_cols = [c for c in cols_to_interpolate if c in df.columns]
    if target_cols:
        df[target_cols] = df.groupby('device_id')[target_cols].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )

    # Drop rows where critical data might still be missing
    initial_count = len(df)
    df.dropna(inplace=True)
    dropped_count = initial_count - len(df)
    
    if dropped_count > 0:
        # Using st.info for dashboard visibility if possible, else print
        try:
            st.info(f"Dropped {dropped_count} rows that could not be recovered after interpolation.")
        except Exception:
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
    logging.info("Transforming data...")
    
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
    # Check for sustained high temperature using rolling average (window=3)
    if 'temp_rolling_avg_3' in df.columns:
        high_temp = df['temp_rolling_avg_3'] > limit_temp
    else:
        high_temp = df.groupby('device_id')['temperature_celsius'].transform(
            lambda x: x.rolling(window=3).mean()
        ) > limit_temp
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
    
    logging.info("Daily Aggregated Statistics calculated.")
    
    return daily_stats

# --- Unused CLI-specific functions from iot_data_pipeline.py ---
# These are kept for completeness but are not called by the Streamlit app.

def visualize_data(daily_stats: pd.DataFrame, save_to_disk: bool = True) -> List[plt.Figure]:
    """
    Step 6: Visualize (CLI version)
    Plots daily average temperature per device using Matplotlib.
    """
    logging.info("Generating static visualization (Matplotlib)...")
    figs = []
    
    fig1 = plt.figure(figsize=(10, 6))
    
    for device_id in daily_stats['device_id'].unique():
        subset = daily_stats[daily_stats['device_id'] == device_id]
        plt.plot(subset['date'], subset['daily_avg_temp'], marker='o', label=device_id)
        
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

    fig2 = plt.figure(figsize=(10, 6))
    
    for device_id in daily_stats['device_id'].unique():
        subset = daily_stats[daily_stats['device_id'] == device_id]
        plt.plot(subset['date'], subset['daily_min_battery'], marker='s', linestyle='--', label=device_id)
        
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

def load_data_to_storage(df: pd.DataFrame, file_format: str = 'parquet', output_dir: str = 'processed_data'):
    """
    Step 7: Load (CLI version)
    Saves the processed data to partitioned Parquet files or CSV.
    """
    logging.info(f"Loading data to storage ({file_format})...")
    
    df_storage = df.copy()
    df_storage['date'] = df_storage['timestamp'].dt.date.astype(str)
    
    try:
        if file_format == 'parquet':
            df_storage.to_parquet(output_dir, partition_cols=['date'], index=False)
            logging.info(f"Data saved to '{output_dir}' directory (Partitioned by Date)")
        elif file_format == 'csv':
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, 'processed_data.csv')
            df_storage.to_csv(output_path, index=False)
            logging.info(f"Data saved to '{output_path}'")
        else:
            logging.error(f"Unsupported format: {file_format}")
    except Exception as e:
        logging.error(f"Failed to save {file_format} file: {e}")


# --- Streamlit Dashboard Code (from iot_dashboard.py) ---

def main():
    # --- Page Configuration ---
    st.set_page_config(page_title="IoT Sensor Dashboard", layout="wide")

    # --- Sidebar Configuration ---
    st.sidebar.header("⚙️ Configuration")
    temp_threshold = st.sidebar.slider("High Temp Threshold (°C)", 0.0, 100.0, CONF.TEMP_THRESHOLD_HIGH)
    battery_threshold = st.sidebar.slider("Low Battery Threshold (%)", 0.0, 100.0, CONF.BATTERY_THRESHOLD_LOW)

    if st.sidebar.button("Clear Cache & Rerun"):
        st.cache_data.clear()
        st.rerun()

    st.title("📊 IoT Sensor Data Pipeline")
    st.markdown("""
    Upload your raw IoT sensor logs (CSV) below. 
    The system will automatically clean the data, check for anomalies, and visualize trends.
    """)

    # --- File Upload Section ---
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        st.divider()
        
        # 1. Load Data
        st.subheader("1. Raw Data Inspection")
        try:
            # Read directly from the uploaded file buffer
            df_raw = load_data(uploaded_file)

            # Ensure timestamp is datetime for filtering
            if 'timestamp' in df_raw.columns:
                df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
                
                min_date = df_raw['timestamp'].min().date()
                max_date = df_raw['timestamp'].max().date()
                
                st.sidebar.header("📅 Date Filter")
                date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    df_raw = df_raw[(df_raw['timestamp'].dt.date >= start_date) & (df_raw['timestamp'].dt.date <= end_date)]

            st.write(f"**Loaded {len(df_raw)} rows.**")
            with st.expander("View Raw Data"):
                st.dataframe(df_raw.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()

        # 2. Run Pipeline
        with st.spinner('Running ETL Pipeline...'):
            try:
                # Reuse existing pipeline logic now in the same file
                df_clean = clean_data(df_raw)
                df_transformed = transform_data(df_clean)
                anomalies = analyze_anomalies(df_transformed, temp_threshold=temp_threshold, battery_threshold=battery_threshold)
                daily_stats = aggregate_data(df_transformed)
                
                st.success("Pipeline completed successfully!")
                
                # Allow downloading the processed data
                csv = df_transformed.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned Data (CSV)",
                    data=csv,
                    file_name='cleaned_iot_data.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.stop()

        # 3. Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("2. Anomaly Detection")
            if not anomalies.empty:
                st.error(f"⚠️ Found {len(anomalies)} Anomalies")
                st.dataframe(anomalies[['timestamp', 'device_id', 'temperature_celsius', 'battery_level_percent', 'anomaly_type']])
            else:
                st.success("✅ No anomalies detected. Systems nominal.")

        with col2:
            st.subheader("3. Daily Statistics")
            st.dataframe(daily_stats)

        # 4. Visualizations
        st.subheader("4. Visualizations")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Temperature Trends", "Battery Health", "Humidity Trends", "Anomaly Analysis", "Heatmap Analysis", "Rolling Avg Comparison"])
        
        with tab1:
            st.info("Interactive Daily Average Temperature")
            fig_temp = px.line(daily_stats, x='date', y='daily_avg_temp', color='device_id', markers=True)
            fig_temp.add_hline(y=temp_threshold, line_dash="dash", line_color="red", annotation_text="High Temp Limit")
            st.plotly_chart(fig_temp, use_container_width=True)
            
        with tab2:
            st.info("Interactive Daily Minimum Battery")
            fig_batt = px.line(daily_stats, x='date', y='daily_min_battery', color='device_id', markers=True)
            fig_batt.add_hline(y=battery_threshold, line_dash="dash", line_color="red", annotation_text="Low Battery Limit")
            st.plotly_chart(fig_batt, use_container_width=True)

        with tab3:
            st.info("Interactive Daily Average Humidity")
            fig_hum = px.line(daily_stats, x='date', y='daily_avg_humidity', color='device_id', markers=True)
            st.plotly_chart(fig_hum, use_container_width=True)

        with tab4:
            st.info("Detailed View: Anomalies Highlighted")
            # Plot raw data points, coloring by anomaly type
            fig_anom = px.scatter(
                df_transformed, 
                x='timestamp', 
                y='temperature_celsius', 
                color='anomaly_type',
                symbol='device_id',
                hover_data=['battery_level_percent', 'humidity_percent'],
                title="Temperature Readings & Anomalies"
            )
            fig_anom.add_hline(y=temp_threshold, line_dash="dash", line_color="red")
            st.plotly_chart(fig_anom, use_container_width=True)

        with tab5:
            st.info("Heatmap: Average Temperature by Hour of Day")
            fig_heat = px.density_heatmap(
                df_transformed,
                x="hour_of_day",
                y="device_id",
                z="temperature_celsius",
                histfunc="avg",
                color_continuous_scale="Viridis",
                labels={'hour_of_day': 'Hour of Day', 'device_id': 'Device', 'temperature_celsius': 'Avg Temp (°C)'}
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        with tab6:
            st.info("Comparison: Raw Temperature vs. 3-Point Rolling Average")
            
            # Select device to reduce clutter
            device_list = df_transformed['device_id'].unique()
            selected_device = st.selectbox("Select Device for Comparison", device_list)
            
            subset_df = df_transformed[df_transformed['device_id'] == selected_device]
            
            fig_roll = px.line(
                subset_df, 
                x='timestamp', 
                y=['temperature_celsius', 'temp_rolling_avg_3'],
                title=f"Raw vs. Rolling Average Temperature for {selected_device}",
                labels={'value': 'Temperature (°C)', 'variable': 'Metric'}
            )
            st.plotly_chart(fig_roll, use_container_width=True)

if __name__ == "__main__":
    main()