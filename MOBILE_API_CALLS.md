# 📱 Mobile App API Calls - Copy & Paste Ready

## 🔧 **API Configuration**

First, add this at the top of your mobile app:

```javascript
// Replace with your computer's IP address
const API_BASE_URL = 'http://192.168.1.8:8000';
```

---

## 🏠 **1. DASHBOARD TAB - Replace Hardcoded Stats**

### **OLD CODE (Remove this):**
```javascript
// Remove these hardcoded values:
const totalCases = 8081;
const countries = 8;
const criticalAlerts = 2;
const regions = 6;
```

### **NEW CODE (Add this):**
```javascript
// Dashboard API Call
const fetchDashboardStats = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/dashboard/stats`);
    const data = await response.json();
    
    // data contains:
    // {
    //   "total_cases": 8081,
    //   "countries_affected": 8,
    //   "critical_alerts": 2,
    //   "regions_monitored": 6,
    //   "network_status": "Global Network • 8 Outbreaks Tracked",
    //   "outbreaks_tracked": 8
    // }
    
    return data;
  } catch (error) {
    console.error('Dashboard API failed:', error);
    // Return fallback data
    return {
      total_cases: 8081,
      countries_affected: 8,
      critical_alerts: 2,
      regions_monitored: 6
    };
  }
};

// Use it in your component:
const [dashboardData, setDashboardData] = useState({});

useEffect(() => {
  const loadDashboard = async () => {
    const data = await fetchDashboardStats();
    setDashboardData(data);
  };
  
  loadDashboard();
  // Auto-refresh every 30 seconds
  const interval = setInterval(loadDashboard, 30000);
  return () => clearInterval(interval);
}, []);

// Update your UI elements:
<Text>{dashboardData.total_cases?.toLocaleString() || '8,081'}</Text>
<Text>{dashboardData.countries_affected || '8'}</Text>
<Text>{dashboardData.critical_alerts || '2'}</Text>
<Text>{dashboardData.regions_monitored || '6'}</Text>
```

---

## 🔔 **2. ALERTS TAB - Replace Static Alerts**

### **NEW CODE (Add this):**
```javascript
// Alerts API Call
const fetchAlerts = async (filter = 'all') => {
  try {
    const endpoint = filter === 'all' ? '/alerts' : `/alerts/filter/${filter}`;
    const response = await fetch(`${API_BASE_URL}${endpoint}`);
    const data = await response.json();
    
    // data contains:
    // {
    //   "alerts": [
    //     {
    //       "id": "ALERT_001",
    //       "title": "Dengue Fever Surge Detected",
    //       "description": "Unusual spike in dengue cases...",
    //       "disease": "Dengue Fever",
    //       "location": "Mumbai",
    //       "country": "India",
    //       "region": "Asia",
    //       "risk_level": "high",
    //       "timestamp": "47 minutes ago",
    //       "cases": 2847
    //     }
    //   ],
    //   "total_alerts": 3,
    //   "critical_count": 2,
    //   "moderate_count": 1
    // }
    
    return data;
  } catch (error) {
    console.error('Alerts API failed:', error);
    return { alerts: [], total_alerts: 0, critical_count: 0, moderate_count: 0 };
  }
};

// Use it in your alerts component:
const [alerts, setAlerts] = useState([]);
const [currentFilter, setCurrentFilter] = useState('all');

const loadAlerts = async (filter = 'all') => {
  const data = await fetchAlerts(filter);
  setAlerts(data.alerts);
  setCurrentFilter(filter);
};

// Filter button handlers:
const onAllPressed = () => loadAlerts('all');
const onCriticalPressed = () => loadAlerts('high');
const onModeratePressed = () => loadAlerts('moderate');

// Investigate alert function:
const investigateAlert = async (alertId) => {
  try {
    await fetch(`${API_BASE_URL}/alerts/investigate/${alertId}`, { method: 'POST' });
    alert(`Alert ${alertId} marked for investigation`);
  } catch (error) {
    alert('Failed to update alert');
  }
};
```

---

## 🗺️ **3. MAP TAB - Replace Static Map Data**

### **NEW CODE (Add this):**
```javascript
// Map Data API Call
const fetchMapData = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/map/data`);
    const data = await response.json();
    
    // data contains:
    // {
    //   "active_outbreaks": [
    //     {
    //       "disease": "Dengue Fever",
    //       "location": "Mumbai",
    //       "country": "India",
    //       "region": "Asia",
    //       "cases": 2847,
    //       "risk_level": "critical",
    //       "coordinates": {"lat": 19.0760, "lng": 72.8777}
    //     }
    //   ],
    //   "risk_zones": [...],
    //   "global_status": "8 active • 8 countries"
    // }
    
    return data;
  } catch (error) {
    console.error('Map API failed:', error);
    return { active_outbreaks: [], risk_zones: [], global_status: "Offline" };
  }
};

// Use it in your map component:
const [mapData, setMapData] = useState({ active_outbreaks: [] });

useEffect(() => {
  const loadMapData = async () => {
    const data = await fetchMapData();
    setMapData(data);
    
    // Add markers to your map
    data.active_outbreaks.forEach(outbreak => {
      addMapMarker({
        latitude: outbreak.coordinates.lat,
        longitude: outbreak.coordinates.lng,
        title: outbreak.disease,
        subtitle: `${outbreak.location}, ${outbreak.country}`,
        pinColor: getRiskColor(outbreak.risk_level)
      });
    });
  };
  
  loadMapData();
  const interval = setInterval(loadMapData, 60000); // Update every minute
  return () => clearInterval(interval);
}, []);

// Helper function for risk colors:
const getRiskColor = (riskLevel) => {
  switch(riskLevel) {
    case 'critical': return '#FF4444'; // Red
    case 'moderate': return '#FF8800'; // Orange
    case 'low': return '#44FF44';      // Green
    default: return '#888888';         // Gray
  }
};
```

---

## 📈 **4. TRENDS TAB - Replace Static Chart Data**

### **NEW CODE (Add this):**
```javascript
// Trends Data API Call
const fetchTrendsData = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/trends`);
    const data = await response.json();
    
    // data contains:
    // {
    //   "weekly_data": [...],
    //   "disease_breakdown": [
    //     {"disease": "Dengue", "cases": 287, "trend_direction": "up", "percentage_change": 15.2},
    //     {"disease": "Malaria", "cases": 134, "trend_direction": "down", "percentage_change": -8.1},
    //     {"disease": "Cholera", "cases": 67, "trend_direction": "stable", "percentage_change": 2.3},
    //     {"disease": "Yellow", "cases": 73, "trend_direction": "up", "percentage_change": 12.7},
    //     {"disease": "Measles", "cases": 68, "trend_direction": "down", "percentage_change": -5.4},
    //     {"disease": "Influenza", "cases": 103, "trend_direction": "up", "percentage_change": 9.8}
    //   ],
    //   "total_cases_trend": [150, 180, 210, 240, 280, 320]
    // }
    
    return data;
  } catch (error) {
    console.error('Trends API failed:', error);
    return { disease_breakdown: [], total_cases_trend: [] };
  }
};

// Use it in your trends component:
const [trendsData, setTrendsData] = useState({ disease_breakdown: [] });

useEffect(() => {
  const loadTrends = async () => {
    const data = await fetchTrendsData();
    setTrendsData(data);
    
    // Update your chart with data.total_cases_trend
    updateLineChart(data.total_cases_trend);
  };
  
  loadTrends();
  const interval = setInterval(loadTrends, 30000);
  return () => clearInterval(interval);
}, []);

// Update your disease cards:
{trendsData.disease_breakdown.map((disease, index) => (
  <View key={index} style={styles.diseaseCard}>
    <Text style={styles.diseaseName}>{disease.disease}</Text>
    <Text style={styles.casesCount}>{disease.cases}</Text>
    <Text style={[styles.trend, styles[`trend_${disease.trend_direction}`]]}>
      {disease.trend_direction === 'up' ? '↗️' : disease.trend_direction === 'down' ? '↘️' : '➡️'}
    </Text>
  </View>
))}
```

---

## 🔍 **5. EPIDEMIC PREDICTION (Bonus Feature)**

### **NEW CODE (Add this):**
```javascript
// Predict Epidemic from Text
const predictEpidemic = async (text, location = null) => {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: text,
        location: location
      })
    });
    
    const data = await response.json();
    
    // data contains:
    // {
    //   "prediction": 1,           // 0 or 1 (epidemic detected)
    //   "confidence": 0.87,        // 0.0 to 1.0
    //   "risk_level": "high",      // "low", "moderate", "high"
    //   "processing_time_ms": 5.2, // Ultra fast!
    //   "model_version": "Custom LSTM+Attention v1.0"
    // }
    
    return data;
  } catch (error) {
    console.error('Prediction API failed:', error);
    return { prediction: 0, confidence: 0, risk_level: 'low' };
  }
};

// Use it for text analysis:
const analyzeText = async () => {
  const result = await predictEpidemic(
    "Outbreak of dengue fever reported in Mumbai with rising cases",
    "Mumbai, India"
  );
  
  if (result.prediction === 1) {
    alert(`⚠️ Epidemic Detected! Risk: ${result.risk_level} (${(result.confidence * 100).toFixed(1)}%)`);
  } else {
    alert(`✅ No epidemic detected (${(result.confidence * 100).toFixed(1)}% confidence)`);
  }
};
```

---

## 🔄 **6. COMPLETE INTEGRATION EXAMPLE**

### **Complete Dashboard Component:**
```javascript
import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, RefreshControl, ScrollView } from 'react-native';

const API_BASE_URL = 'http://192.168.1.8:8000'; // Replace with your IP

const Dashboard = () => {
  const [data, setData] = useState({
    total_cases: 8081,
    countries_affected: 8,
    critical_alerts: 2,
    regions_monitored: 6
  });
  const [refreshing, setRefreshing] = useState(false);
  const [isLive, setIsLive] = useState(false);

  const fetchData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/dashboard/stats`);
      const apiData = await response.json();
      setData(apiData);
      setIsLive(true);
      console.log('✅ Live data loaded');
    } catch (error) {
      setIsLive(false);
      console.log('📱 Using offline data');
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchData();
    setRefreshing(false);
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <ScrollView
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {/* Status Indicator */}
      <View style={styles.statusBar}>
        <Text style={[styles.status, { color: isLive ? '#44FF44' : '#FF8800' }]}>
          {isLive ? '🟢 Live Data' : '🟡 Offline Mode'}
        </Text>
      </View>

      {/* Stats Grid */}
      <View style={styles.statsGrid}>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>{data.total_cases?.toLocaleString() || '8,081'}</Text>
          <Text style={styles.statLabel}>Cases</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>{data.countries_affected || '8'}</Text>
          <Text style={styles.statLabel}>Countries</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>{data.critical_alerts || '2'}</Text>
          <Text style={styles.statLabel}>Critical</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>{data.regions_monitored || '6'}</Text>
          <Text style={styles.statLabel}>Regions</Text>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0e27',
  },
  statusBar: {
    padding: 10,
    alignItems: 'center',
  },
  status: {
    fontSize: 14,
    fontWeight: 'bold',
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-around',
    padding: 20,
  },
  statCard: {
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    padding: 20,
    margin: 8,
    width: '40%',
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  statLabel: {
    fontSize: 14,
    color: '#888888',
    marginTop: 4,
  },
});

export default Dashboard;
```

---

## 🎯 **SUMMARY: What to Replace**

1. **Replace hardcoded numbers** with `fetchDashboardStats()`
2. **Replace static alerts** with `fetchAlerts(filter)`
3. **Replace static map markers** with `fetchMapData()`
4. **Replace chart data** with `fetchTrendsData()`
5. **Add real-time updates** every 30 seconds
6. **Add offline fallback** for when API fails

**Copy these functions into your mobile app and replace your existing hardcoded data! 🚀**