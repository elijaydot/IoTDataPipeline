import unittest
from unittest.mock import MagicMock
import sys
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
import pandera as pa

# --- Mock Streamlit ---
# We must mock streamlit before importing iot_dashboard_app because it uses 
# streamlit decorators (@st.cache_data) at the module level.
mock_st = MagicMock()
mock_st.cache_data = lambda func: func  # Mock cache_data as a pass-through decorator
sys.modules["streamlit"] = mock_st

# Now import the module under test
from iot_dashboard_app import clean_data, analyze_anomalies, load_data_to_storage, load_data, CONF

class TestIoTDashboardApp(unittest.TestCase):
    
    def setUp(self):
        # Create sample data for testing
        self.raw_data = pd.DataFrame({
            'timestamp': ['2023-01-01 10:00:00', '2023-01-01 10:10:00', '2023-01-01 10:20:00'],
            'device_id': ['sensor_test', 'sensor_test', 'sensor_test'],
            'temperature_celsius': [20.0, np.nan, 22.0], # Missing value for interpolation
            'humidity_percent': [50.0, 51.0, 52.0],
            'pressure_hpa': [1010.0, 1011.0, 1012.0],
            'battery_level_percent': [90, 85, 80]
        })
        
        # Create a temporary directory for file output tests
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)

    def test_clean_data_interpolation(self):
        """Test if missing temperature is interpolated correctly."""
        cleaned_df = clean_data(self.raw_data.copy())
        
        # The middle value (NaN) should be interpolated between 20.0 and 22.0 -> 21.0
        interpolated_value = cleaned_df.iloc[1]['temperature_celsius']
        self.assertEqual(interpolated_value, 21.0, "Interpolation logic failed")

    def test_clean_data_validation(self):
        """Test if out-of-range values raise a SchemaError."""
        bad_data = self.raw_data.copy()
        bad_data.loc[0, 'temperature_celsius'] = 1000.0 # Impossible temp
        
        # Expect Pandera to raise a SchemaError because of the invalid temperature
        with self.assertRaises(pa.errors.SchemaErrors):
            clean_data(bad_data)

    def test_anomaly_detection(self):
        """Test if anomalies are correctly flagged."""
        # Create data that triggers thresholds.
        data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2023-01-01 10:00:00', '2023-01-01 10:10:00', '2023-01-01 10:20:00', # s1: Sustained High Temp
                '2023-01-01 10:00:00' # s2: Low Battery
            ]),
            'device_id': ['s1', 's1', 's1', 's2'],
            'temperature_celsius': [35.0, 35.0, 35.0, 20.0], 
            'battery_level_percent': [90, 90, 90, 10]
        })
        
        anomalies = analyze_anomalies(data)
        
        # Check for High Temperature Anomaly
        high_temp_anomalies = anomalies[anomalies['anomaly_type'].str.contains('High Temperature')]
        self.assertTrue(len(high_temp_anomalies) > 0, "Failed to detect sustained high temperature")
        
        # Check for Low Battery Anomaly
        low_batt_anomalies = anomalies[anomalies['anomaly_type'].str.contains('Low Battery')]
        self.assertTrue(len(low_batt_anomalies) > 0, "Failed to detect low battery")

    def test_load_data_file_not_found(self):
        """Test that load_data raises FileNotFoundError for non-existent files."""
        with self.assertRaises(FileNotFoundError):
            load_data("non_existent_file.csv")

    def test_load_data_to_storage_csv(self):
        """Test saving data to CSV format."""
        load_data_to_storage(self.raw_data, file_format='csv', output_dir=self.test_dir)
        
        expected_file = os.path.join(self.test_dir, 'processed_data.csv')
        self.assertTrue(os.path.exists(expected_file), "CSV file was not created")

if __name__ == '__main__':
    unittest.main()