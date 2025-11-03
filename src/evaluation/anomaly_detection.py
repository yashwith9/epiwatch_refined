"""
Anomaly Detection System for Outbreak Alerts
Detects unusual spikes in outbreak signals by region and time
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import json


class AnomalyDetector:
    """
    Time-series anomaly detection for epidemic signals
    """
    
    def __init__(self, method='zscore', threshold=3.0, window_size=7):
        """
        Args:
            method: Detection method ('zscore', 'isolation_forest', 'moving_average')
            threshold: Threshold for anomaly detection
            window_size: Window size for moving average (in days)
        """
        self.method = method
        self.threshold = threshold
        self.window_size = window_size
        
    def detect_zscore(self, data):
        """
        Z-score based anomaly detection
        Detects values that are more than 'threshold' standard deviations from mean
        """
        if len(data) < 2:
            return np.zeros(len(data), dtype=bool)
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return np.zeros(len(data), dtype=bool)
        
        z_scores = np.abs((data - mean) / std)
        anomalies = z_scores > self.threshold
        
        return anomalies
    
    def detect_isolation_forest(self, data):
        """
        Isolation Forest based anomaly detection
        Machine learning approach for outlier detection
        """
        if len(data) < 10:
            return np.zeros(len(data), dtype=bool)
        
        data_reshaped = data.reshape(-1, 1)
        
        clf = IsolationForest(contamination=0.1, random_state=42)
        predictions = clf.fit_predict(data_reshaped)
        
        anomalies = predictions == -1
        return anomalies
    
    def detect_moving_average(self, data):
        """
        Moving average based anomaly detection
        Detects values significantly different from recent average
        """
        if len(data) < self.window_size:
            return np.zeros(len(data), dtype=bool)
        
        anomalies = np.zeros(len(data), dtype=bool)
        
        for i in range(self.window_size, len(data)):
            window = data[i-self.window_size:i]
            mean = np.mean(window)
            std = np.std(window)
            
            if std > 0:
                z_score = abs((data[i] - mean) / std)
                anomalies[i] = z_score > self.threshold
        
        return anomalies
    
    def detect(self, data):
        """
        Detect anomalies using selected method
        
        Args:
            data: Time-series data (numpy array or list)
            
        Returns:
            Boolean array indicating anomalies
        """
        data = np.array(data)
        
        if self.method == 'zscore':
            return self.detect_zscore(data)
        elif self.method == 'isolation_forest':
            return self.detect_isolation_forest(data)
        elif self.method == 'moving_average':
            return self.detect_moving_average(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class OutbreakAlertSystem:
    """
    Complete alert system for epidemic detection
    Aggregates predictions by region, time, and disease
    """
    
    def __init__(self, anomaly_detector=None):
        """
        Args:
            anomaly_detector: AnomalyDetector instance
        """
        self.anomaly_detector = anomaly_detector or AnomalyDetector()
        self.alerts = []
        
    def aggregate_predictions(self, predictions_df):
        """
        Aggregate model predictions by region, date, and disease
        
        Args:
            predictions_df: DataFrame with columns:
                - date: Date of prediction
                - region: Geographic region
                - disease: Disease name
                - text: Original text
                - prediction: Binary prediction (0 or 1)
                - probability: Prediction probability
        
        Returns:
            Aggregated DataFrame with counts per region/date/disease
        """
        aggregated = predictions_df.groupby(['date', 'region', 'disease']).agg({
            'prediction': 'sum',  # Count of outbreak signals
            'probability': 'mean'  # Average probability
        }).reset_index()
        
        aggregated.columns = ['date', 'region', 'disease', 'signal_count', 'avg_probability']
        
        return aggregated
    
    def detect_outbreaks(self, aggregated_df):
        """
        Detect outbreak anomalies from aggregated data
        
        Args:
            aggregated_df: Aggregated DataFrame from aggregate_predictions
            
        Returns:
            DataFrame with detected outbreaks and risk levels
        """
        outbreaks = []
        
        # Group by region and disease
        for (region, disease), group in aggregated_df.groupby(['region', 'disease']):
            # Sort by date
            group = group.sort_values('date')
            
            # Get signal counts over time
            signal_counts = group['signal_count'].values
            
            # Detect anomalies
            anomalies = self.anomaly_detector.detect(signal_counts)
            
            # Process anomalies
            for i, is_anomaly in enumerate(anomalies):
                if is_anomaly:
                    date = group.iloc[i]['date']
                    count = group.iloc[i]['signal_count']
                    prob = group.iloc[i]['avg_probability']
                    
                    # Calculate risk level
                    risk_level = self._calculate_risk_level(count, prob)
                    
                    outbreak = {
                        'date': date,
                        'region': region,
                        'disease': disease,
                        'signal_count': int(count),
                        'avg_probability': float(prob),
                        'risk_level': risk_level,
                        'alert_generated': datetime.now().isoformat()
                    }
                    
                    outbreaks.append(outbreak)
        
        return pd.DataFrame(outbreaks)
    
    def _calculate_risk_level(self, count, probability):
        """
        Calculate risk level based on signal count and probability
        
        Returns: 'high', 'moderate', or 'low'
        """
        score = count * probability
        
        if score >= 100:
            return 'high'
        elif score >= 50:
            return 'moderate'
        else:
            return 'low'
    
    def generate_alert_message(self, outbreak):
        """
        Generate human-readable alert message
        
        Args:
            outbreak: Dictionary with outbreak information
            
        Returns:
            Alert message string
        """
        templates = {
            'high': "⚠️ CRITICAL: {disease} outbreak detected in {region}. {count} signals with {prob:.1%} confidence. Immediate action recommended.",
            'moderate': "⚡ WARNING: Potential {disease} outbreak in {region}. {count} signals detected with {prob:.1%} confidence. Monitor situation closely.",
            'low': "ℹ️ ADVISORY: Increased {disease} activity in {region}. {count} signals detected. Continue routine monitoring."
        }
        
        template = templates[outbreak['risk_level']]
        
        return template.format(
            disease=outbreak['disease'],
            region=outbreak['region'],
            count=outbreak['signal_count'],
            prob=outbreak['avg_probability']
        )
    
    def process_and_alert(self, predictions_df):
        """
        Complete pipeline: aggregate -> detect -> generate alerts
        
        Args:
            predictions_df: Raw predictions DataFrame
            
        Returns:
            List of alert dictionaries
        """
        # Aggregate predictions
        aggregated = self.aggregate_predictions(predictions_df)
        
        # Detect outbreaks
        outbreaks_df = self.detect_outbreaks(aggregated)
        
        # Generate alerts
        alerts = []
        for _, outbreak in outbreaks_df.iterrows():
            alert = outbreak.to_dict()
            alert['message'] = self.generate_alert_message(alert)
            alert['id'] = len(alerts) + 1
            alerts.append(alert)
        
        self.alerts = alerts
        return alerts
    
    def get_alerts_for_mobile(self, limit=None, risk_levels=None):
        """
        Format alerts for mobile application
        
        Args:
            limit: Maximum number of alerts to return
            risk_levels: List of risk levels to filter ['high', 'moderate', 'low']
            
        Returns:
            List of formatted alerts for mobile app
        """
        filtered_alerts = self.alerts
        
        # Filter by risk level
        if risk_levels:
            filtered_alerts = [a for a in filtered_alerts if a['risk_level'] in risk_levels]
        
        # Sort by risk level and date (most recent first)
        risk_order = {'high': 0, 'moderate': 1, 'low': 2}
        filtered_alerts.sort(key=lambda x: (risk_order[x['risk_level']], x['date']), reverse=True)
        
        # Limit results
        if limit:
            filtered_alerts = filtered_alerts[:limit]
        
        # Format for mobile
        mobile_alerts = []
        for alert in filtered_alerts:
            mobile_alert = {
                'id': alert['id'],
                'title': f"{alert['disease']} Alert",
                'location': alert['region'],
                'risk_level': alert['risk_level'],
                'case_count': alert['signal_count'],
                'date': alert['date'],
                'summary': alert['message'],
                'color': self._get_risk_color(alert['risk_level'])
            }
            mobile_alerts.append(mobile_alert)
        
        return mobile_alerts
    
    def _get_risk_color(self, risk_level):
        """Get color code for risk level"""
        colors = {
            'high': '#FF4444',
            'moderate': '#FFA500',
            'low': '#4CAF50'
        }
        return colors.get(risk_level, '#808080')
    
    def get_map_data(self):
        """
        Generate geospatial data for map visualization
        
        Returns:
            List of region data with coordinates and risk levels
        """
        if not self.alerts:
            return []
        
        # Group alerts by region
        region_data = {}
        for alert in self.alerts:
            region = alert['region']
            if region not in region_data:
                region_data[region] = {
                    'region': region,
                    'alerts': [],
                    'max_risk': 'low'
                }
            
            region_data[region]['alerts'].append(alert)
            
            # Update max risk level
            current_risk = region_data[region]['max_risk']
            new_risk = alert['risk_level']
            
            risk_order = {'high': 2, 'moderate': 1, 'low': 0}
            if risk_order[new_risk] > risk_order[current_risk]:
                region_data[region]['max_risk'] = new_risk
        
        return list(region_data.values())
    
    def get_trend_data(self, days=7):
        """
        Generate 7-day trend data for mobile app
        
        Args:
            days: Number of days to include
            
        Returns:
            Dictionary with disease trends over time
        """
        if not self.alerts:
            return {}
        
        # Get date range
        dates = [alert['date'] for alert in self.alerts]
        if not dates:
            return {}
        
        max_date = max(dates)
        min_date = max_date - timedelta(days=days-1)
        
        # Aggregate by disease and date
        trends = {}
        for alert in self.alerts:
            if alert['date'] < min_date:
                continue
            
            disease = alert['disease']
            if disease not in trends:
                trends[disease] = {date: 0 for date in pd.date_range(min_date, max_date)}
            
            trends[disease][alert['date']] += alert['signal_count']
        
        # Format for mobile
        mobile_trends = {}
        for disease, data in trends.items():
            mobile_trends[disease] = {
                'name': disease,
                'data': [{'date': str(date), 'count': count} 
                        for date, count in sorted(data.items())]
            }
        
        return mobile_trends
    
    def save_alerts(self, filepath='outputs/alerts/current_alerts.json'):
        """Save alerts to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.alerts, f, indent=4, default=str)
        print(f"✓ Alerts saved: {filepath}")


if __name__ == "__main__":
    print("Outbreak Alert System for EpiWatch")
    print("=" * 60)
    
    # Example usage
    detector = AnomalyDetector(method='zscore', threshold=2.5)
    alert_system = OutbreakAlertSystem(detector)
    
    # Example data
    example_data = np.array([10, 12, 11, 13, 15, 45, 50, 14, 12, 11])
    anomalies = detector.detect(example_data)
    
    print(f"\nData: {example_data}")
    print(f"Anomalies: {anomalies}")
    print(f"Anomaly indices: {np.where(anomalies)[0]}")
