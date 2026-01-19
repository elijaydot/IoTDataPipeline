import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from iot_data_pipeline import clean_data, transform_data, analyze_anomalies, aggregate_data, CONF

# --- Page Configuration ---
st.set_page_config(page_title="IoT Sensor Dashboard", layout="wide")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")
temp_threshold = st.sidebar.slider("High Temp Threshold (°C)", 0.0, 100.0, CONF.TEMP_THRESHOLD_HIGH)
battery_threshold = st.sidebar.slider("Low Battery Threshold (%)", 0.0, 100.0, CONF.BATTERY_THRESHOLD_LOW)

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
        df_raw = pd.read_csv(uploaded_file)
        st.write(f"**Loaded {len(df_raw)} rows.**")
        with st.expander("View Raw Data"):
            st.dataframe(df_raw.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # 2. Run Pipeline
    with st.spinner('Running ETL Pipeline...'):
        try:
            # Reuse existing pipeline logic
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
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Temperature Trends", "Battery Health", "Humidity Trends", "Anomaly Analysis", "Heatmap Analysis"])
    
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