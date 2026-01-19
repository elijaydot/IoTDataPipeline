-- IoT Data Pipeline Schema
-- Database: PostgreSQL
-- Description: Star Schema for IoT Sensor Data with Partitioning and Validity Windows

-- 1. Dimension Table: Devices
-- Stores metadata about the sensors.
-- Includes columns for handling different reporting frequencies.
CREATE TABLE dim_devices (
    device_key SERIAL PRIMARY KEY,
    device_id VARCHAR(50) UNIQUE NOT NULL,
    model VARCHAR(50),
    location VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata for frequency handling
    expected_interval_seconds INTEGER, -- e.g., 60 for 1 min, 3600 for 1 hour
    sensor_category VARCHAR(50)        -- e.g., 'High-Frequency', 'Periodic', 'On-Event'
);

-- 2. Fact Table: Sensor Readings
-- Stores the high-volume measurement data.
-- Partitioned by timestamp to handle high volume and different frequencies efficiently.
CREATE TABLE fact_sensor_readings (
    reading_id BIGSERIAL, 
    device_key INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Metrics
    temperature_celsius DECIMAL(5,2),
    temperature_fahrenheit DECIMAL(5,2),
    humidity_percent DECIMAL(5,2),
    pressure_hpa DECIMAL(7,2),
    battery_level DECIMAL(5,2),
    
    -- Derived Features
    hour_of_day INTEGER,
    temp_rolling_avg_3 DECIMAL(5,2),
    
    -- Anomaly Flags
    is_anomaly BOOLEAN DEFAULT FALSE,
    anomaly_type VARCHAR(50),
    
    -- Validity Window (for sparse data analysis)
    -- Represents the time until this reading is superseded by the next one
    valid_until TIMESTAMP,

    -- Constraints
    CONSTRAINT fk_device
      FOREIGN KEY(device_key) 
      REFERENCES dim_devices(device_key)
) PARTITION BY RANGE (timestamp);

-- 3. Create Partitions (Example for 2023)
-- In a production environment, these are often managed by a maintenance script.
CREATE TABLE fact_sensor_readings_y2023 PARTITION OF fact_sensor_readings
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

-- Catch-all partition for other dates
CREATE TABLE fact_sensor_readings_default PARTITION OF fact_sensor_readings DEFAULT;

-- 4. Indexes for Performance
CREATE INDEX idx_readings_timestamp ON fact_sensor_readings(timestamp);
CREATE INDEX idx_readings_device ON fact_sensor_readings(device_key);
CREATE INDEX idx_readings_device_validity ON fact_sensor_readings(device_key, timestamp, valid_until);