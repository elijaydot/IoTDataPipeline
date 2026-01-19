import unittest
import pandas as pd
import numpy as np
import pandera as pa
from iot_data_pipeline import clean_data, analyze_anomalies, CONF

class TestIoTPipeline(unittest.TestCase):
    
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
        # Create data that triggers thresholds
        # High Temp: 35 > 30
        # Low Battery: 10 < 15
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:10:00']),
            'device_id': ['s1', 's2'],
            'temperature_celsius': [35.0, 20.0], 
            'battery_level_percent': [90, 10]
        })
        
        anomalies = analyze_anomalies(data)
        
        # Both rows should be anomalies
        self.assertEqual(len(anomalies), 2)
        self.assertTrue('High Temperature' in anomalies.iloc[0]['anomaly_type'])
        self.assertTrue('Low Battery' in anomalies.iloc[1]['anomaly_type'])

if __name__ == '__main__':
    unittest.main()