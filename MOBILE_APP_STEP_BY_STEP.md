# 📱 Mobile App Integration - Step by Step Guide

## 🎯 What You Need to Do with Your Mobile App

Based on your app screenshots, here's exactly what you need to implement:

---

## 📋 **STEP 1: Update Your App's API Configuration**

### In your mobile app code, add these API endpoints:

```javascript
// API Configuration
const API_CONFIG = {
  BASE_URL: 'http://localhost:8000',  // Change to your server IP for mobile testing
  ENDPOINTS: {
    DASHBOARD: '/dashboard/stats',
    ALERTS: '/alerts',
    ALERTS_FILTER: '/alerts/filter',
    TRENDS: '/trends', 
    MAP_DATA: '/map/data',
    PREDICT: '/predict'
  }
};

// For mobile device testing, use your computer's IP:
// const API_CONFIG = { BASE_URL: 'http://192.168.1.100:8000' };
```

---

## 🏠 **STEP 2: Dashboard Tab Implementation**

### Update your dashboard to fetch real data:

```javascript
// Dashboard.js or Dashboard.dart (React Native/Flutter)
async function updateDashboard() {
  try {
    const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.DASHBOARD}`);
    const data = await response.json();
    
    // Update your UI elements with real data
    updateDashboardCards({
      totalCases: data.total_cases,        // 8,081
      countries: data.countries_affected,   // 8
      critical: data.critical_alerts,      // 2
      regions: data.regions_monitored      // 6
    });
    
    // Update network status
    updateNetworkStatus(data.network_status); // "Global Network • 8 Outbreaks Tracked"
    
  } catch (error) {
    console.error('Dashboard update failed:', error);
    // Show cached data or error message
  }
}

// Call this when dashboard tab loads
updateDashboard();
```

### For React Native:
```jsx
import React, { useState, useEffect } from 'react';

const Dashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  
  useEffect(() => {
    fetchDashboardData();
  }, []);
  
  const fetchDashboardData = async () => {
    try {
      const response = await fetch('http://localhost:8000/dashboard/stats');
      const data = await response.json();
      setDashboardData(data);
    } catch (error) {
      console.error('Error:', error);
    }
  };
  
  return (
    <View>
      <Text>{dashboardData?.total_cases || '8,081'}</Text>
      <Text>{dashboardData?.countries_affected || '8'}</Text>
      {/* Update your existing UI components */}
    </View>
  );
};
```

---

## 📈 **STEP 3: Trends Tab Implementation**

### Update your trends chart with real data:

```javascript
// Trends.js
async function updateTrends() {
  try {
    const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.TRENDS}`);
    const data = await response.json();
    
    // Update disease breakdown cards
    updateDiseaseCards([
      { name: 'Dengue', cases: data.disease_breakdown[0].cases, trend: data.disease_breakdown[0].trend_direction },
      { name: 'Malaria', cases: data.disease_breakdown[1].cases, trend: data.disease_breakdown[1].trend_direction },
      { name: 'Cholera', cases: data.disease_breakdown[2].cases, trend: data.disease_breakdown[2].trend_direction },
      { name: 'Yellow', cases: data.disease_breakdown[3].cases, trend: data.disease_breakdown[3].trend_direction },
      { name: 'Measles', cases: data.disease_breakdown[4].cases, trend: data.disease_breakdown[4].trend_direction },
      { name: 'Influenza', cases: data.disease_breakdown[5].cases, trend: data.disease_breakdown[5].trend_direction }
    ]);
    
    // Update line chart
    updateLineChart(data.total_cases_trend); // [150, 180, 210, 240, 280, 320]
    
  } catch (error) {
    console.error('Trends update failed:', error);
  }
}

function updateDiseaseCards(diseases) {
  diseases.forEach((disease, index) => {
    // Update your existing disease cards
    document.getElementById(`disease-${index}-cases`).textContent = disease.cases;
    document.getElementById(`disease-${index}-trend`).className = `trend-${disease.trend}`;
  });
}
```

---

## 🗺️ **STEP 4: Map Tab Implementation**

### Update your map with real outbreak locations:

```javascript
// Map.js
async function updateMap() {
  try {
    const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.MAP_DATA}`);
    const data = await response.json();
    
    // Clear existing markers
    map.clearMarkers();
    
    // Add outbreak markers
    data.active_outbreaks.forEach(outbreak => {
      const marker = {
        lat: outbreak.coordinates.lat,
        lng: outbreak.coordinates.lng,
        title: outbreak.disease,
        subtitle: `${outbreak.location}, ${outbreak.country}`,
        cases: outbreak.cases,
        riskLevel: outbreak.risk_level,
        color: getRiskColor(outbreak.risk_level)
      };
      
      map.addMarker(marker);
    });
    
    // Update global status
    updateGlobalStatus(data.global_status); // "8 active • 8 countries"
    
    // Update outbreak list
    updateOutbreakList(data.active_outbreaks);
    
  } catch (error) {
    console.error('Map update failed:', error);
  }
}

function getRiskColor(riskLevel) {
  switch(riskLevel) {
    case 'critical': return '#FF4444'; // Red dots
    case 'moderate': return '#FF8800'; // Orange dots  
    case 'low': return '#44FF44';      // Green dots
    default: return '#888888';
  }
}

function updateOutbreakList(outbreaks) {
  const listContainer = document.getElementById('outbreak-list');
  listContainer.innerHTML = '';
  
  outbreaks.forEach(outbreak => {
    const item = createOutbreakListItem({
      disease: outbreak.disease,
      location: `${outbreak.location}, ${outbreak.country}`,
      region: outbreak.region,
      cases: outbreak.cases,
      riskLevel: outbreak.risk_level
    });
    listContainer.appendChild(item);
  });
}
```

---

## 🔔 **STEP 5: Alerts Tab Implementation**

### Update your alerts with real data and filtering:

```javascript
// Alerts.js
let currentFilter = 'all';

async function updateAlerts(filter = 'all') {
  try {
    const endpoint = filter === 'all' ? 
      `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.ALERTS}` :
      `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.ALERTS_FILTER}/${filter}`;
      
    const response = await fetch(endpoint);
    const data = await response.json();
    
    // Update filter buttons
    updateFilterButtons(filter, data);
    
    // Update alerts list
    updateAlertsList(data.alerts);
    
    currentFilter = filter;
    
  } catch (error) {
    console.error('Alerts update failed:', error);
  }
}

function updateFilterButtons(activeFilter, data) {
  // Update button states
  document.getElementById('all-btn').className = activeFilter === 'all' ? 'active' : '';
  document.getElementById('critical-btn').className = activeFilter === 'high' ? 'active' : '';
  document.getElementById('moderate-btn').className = activeFilter === 'moderate' ? 'active' : '';
  
  // Update counts
  document.getElementById('critical-count').textContent = data.critical_count;
  document.getElementById('moderate-count').textContent = data.moderate_count;
}

function updateAlertsList(alerts) {
  const alertsContainer = document.getElementById('alerts-container');
  alertsContainer.innerHTML = '';
  
  alerts.forEach(alert => {
    const alertElement = createAlertCard({
      id: alert.id,
      title: alert.title,
      description: alert.description,
      location: `${alert.location}, ${alert.country}`,
      region: alert.region,
      riskLevel: alert.risk_level,
      timestamp: alert.timestamp,
      cases: alert.cases
    });
    
    alertsContainer.appendChild(alertElement);
  });
}

// Filter button handlers
document.getElementById('all-btn').onclick = () => updateAlerts('all');
document.getElementById('critical-btn').onclick = () => updateAlerts('high');
document.getElementById('moderate-btn').onclick = () => updateAlerts('moderate');

// Investigate button handler
function investigateAlert(alertId) {
  fetch(`${API_CONFIG.BASE_URL}/alerts/investigate/${alertId}`, { method: 'POST' })
    .then(response => response.json())
    .then(data => {
      showNotification(`Alert ${alertId} marked for investigation`);
      // Update UI to show investigation status
    });
}
```

---

## 🔄 **STEP 6: Real-time Updates**

### Add automatic refresh for live data:

```javascript
// App.js - Main app file
class SentinelApp {
  constructor() {
    this.refreshInterval = null;
    this.currentTab = 'dashboard';
  }
  
  startRealTimeUpdates() {
    // Update every 30 seconds
    this.refreshInterval = setInterval(() => {
      this.refreshCurrentTab();
    }, 30000);
  }
  
  refreshCurrentTab() {
    switch(this.currentTab) {
      case 'dashboard':
        updateDashboard();
        break;
      case 'trends':
        updateTrends();
        break;
      case 'map':
        updateMap();
        break;
      case 'alerts':
        updateAlerts(currentFilter);
        break;
    }
  }
  
  onTabChange(tabName) {
    this.currentTab = tabName;
    this.refreshCurrentTab();
  }
  
  stopRealTimeUpdates() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
  }
}

// Initialize app
const app = new SentinelApp();
app.startRealTimeUpdates();
```

---

## 🛠️ **STEP 7: Error Handling & Offline Support**

### Add robust error handling:

```javascript
// Utils.js
class APIClient {
  static async fetchWithFallback(url, fallbackData) {
    try {
      const response = await fetch(url, { timeout: 5000 });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.warn(`API call failed: ${url}`, error);
      // Return cached data or default values
      return fallbackData;
    }
  }
  
  static async getDashboardStats() {
    return this.fetchWithFallback(
      `${API_CONFIG.BASE_URL}/dashboard/stats`,
      { total_cases: 8081, countries_affected: 8, critical_alerts: 2, regions_monitored: 6 }
    );
  }
  
  static async getAlerts(filter = 'all') {
    const endpoint = filter === 'all' ? '/alerts' : `/alerts/filter/${filter}`;
    return this.fetchWithFallback(
      `${API_CONFIG.BASE_URL}${endpoint}`,
      { alerts: [], total_alerts: 0, critical_count: 0, moderate_count: 0 }
    );
  }
}
```

---

## 📱 **STEP 8: Mobile Device Testing**

### For testing on mobile device:

1. **Find your computer's IP address:**
   ```bash
   # Windows
   ipconfig
   
   # Mac/Linux  
   ifconfig
   ```

2. **Update API URL in your mobile app:**
   ```javascript
   const API_CONFIG = {
     BASE_URL: 'http://192.168.1.100:8000', // Replace with your computer's IP
   };
   ```

3. **Make sure API server is accessible:**
   ```bash
   # Start API server
   python epidemic_api.py
   
   # Test from mobile browser
   # Open: http://192.168.1.100:8000
   ```

---

## 🚀 **STEP 9: Production Deployment**

### When ready for production:

1. **Deploy API to cloud server** (AWS, Google Cloud, etc.)
2. **Update mobile app API URL:**
   ```javascript
   const API_CONFIG = {
     BASE_URL: 'https://your-api-domain.com',
   };
   ```

3. **Add authentication if needed:**
   ```javascript
   const headers = {
     'Authorization': 'Bearer your-api-key',
     'Content-Type': 'application/json'
   };
   ```

---

## ✅ **STEP 10: Testing Checklist**

Before releasing, test these scenarios:

- [ ] Dashboard loads with correct numbers
- [ ] Trends chart updates with real data  
- [ ] Map shows outbreak markers correctly
- [ ] Alerts filter by risk level works
- [ ] Real-time updates work (30-second refresh)
- [ ] Offline mode shows cached data
- [ ] Error handling works when API is down
- [ ] Mobile device can connect to API

---

## 🎯 **Summary: What You Need to Do**

1. **Replace hardcoded data** in your mobile app with API calls
2. **Update 4 main functions**: `updateDashboard()`, `updateTrends()`, `updateMap()`, `updateAlerts()`
3. **Add real-time refresh** every 30 seconds
4. **Test on mobile device** using your computer's IP address
5. **Deploy API to production** when ready

**Your mobile app will now show live epidemic data powered by AI! 🎉**