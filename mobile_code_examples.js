// 📱 MOBILE APP CODE EXAMPLES
// Copy these functions into your mobile app

// ===================================
// REACT NATIVE EXAMPLES
// ===================================

// Dashboard Component (React Native)
import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, RefreshControl, ScrollView } from 'react-native';

const Dashboard = () => {
  const [dashboardData, setDashboardData] = useState({
    total_cases: 8081,
    countries_affected: 8,
    critical_alerts: 2,
    regions_monitored: 6
  });
  const [refreshing, setRefreshing] = useState(false);

  const API_BASE_URL = 'http://192.168.1.100:8000'; // Replace with your IP

  const fetchDashboardData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/dashboard/stats`);
      const data = await response.json();
      setDashboardData(data);
    } catch (error) {
      console.error('Dashboard fetch error:', error);
      // Keep existing data on error
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchDashboardData();
    setRefreshing(false);
  };

  useEffect(() => {
    fetchDashboardData();
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <ScrollView
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      <View style={styles.statsContainer}>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>{dashboardData.total_cases.toLocaleString()}</Text>
          <Text style={styles.statLabel}>Cases</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>{dashboardData.countries_affected}</Text>
          <Text style={styles.statLabel}>Countries</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>{dashboardData.critical_alerts}</Text>
          <Text style={styles.statLabel}>Critical</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>{dashboardData.regions_monitored}</Text>
          <Text style={styles.statLabel}>Regions</Text>
        </View>
      </View>
    </ScrollView>
  );
};

// Alerts Component (React Native)
const Alerts = () => {
  const [alerts, setAlerts] = useState([]);
  const [filter, setFilter] = useState('all');
  const [loading, setLoading] = useState(false);

  const fetchAlerts = async (filterType = 'all') => {
    setLoading(true);
    try {
      const endpoint = filterType === 'all' ? '/alerts' : `/alerts/filter/${filterType}`;
      const response = await fetch(`${API_BASE_URL}${endpoint}`);
      const data = await response.json();
      setAlerts(data.alerts);
      setFilter(filterType);
    } catch (error) {
      console.error('Alerts fetch error:', error);
    }
    setLoading(false);
  };

  const investigateAlert = async (alertId) => {
    try {
      await fetch(`${API_BASE_URL}/alerts/investigate/${alertId}`, { method: 'POST' });
      Alert.alert('Success', `Alert ${alertId} marked for investigation`);
    } catch (error) {
      Alert.alert('Error', 'Failed to update alert status');
    }
  };

  return (
    <View style={styles.container}>
      {/* Filter Buttons */}
      <View style={styles.filterContainer}>
        <TouchableOpacity 
          style={[styles.filterButton, filter === 'all' && styles.activeFilter]}
          onPress={() => fetchAlerts('all')}
        >
          <Text style={styles.filterText}>All</Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[styles.filterButton, filter === 'high' && styles.activeFilter]}
          onPress={() => fetchAlerts('high')}
        >
          <Text style={styles.filterText}>Critical</Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[styles.filterButton, filter === 'moderate' && styles.activeFilter]}
          onPress={() => fetchAlerts('moderate')}
        >
          <Text style={styles.filterText}>Moderate</Text>
        </TouchableOpacity>
      </View>

      {/* Alerts List */}
      <FlatList
        data={alerts}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <View style={[styles.alertCard, styles[`${item.risk_level}Alert`]]}>
            <Text style={styles.alertTitle}>{item.title}</Text>
            <Text style={styles.alertDescription}>{item.description}</Text>
            <Text style={styles.alertLocation}>{item.location}, {item.country}</Text>
            <Text style={styles.alertTime}>{item.timestamp}</Text>
            <TouchableOpacity 
              style={styles.investigateButton}
              onPress={() => investigateAlert(item.id)}
            >
              <Text style={styles.buttonText}>Investigate Alert</Text>
            </TouchableOpacity>
          </View>
        )}
        refreshing={loading}
        onRefresh={() => fetchAlerts(filter)}
      />
    </View>
  );
};

// ===================================
// FLUTTER EXAMPLES (Dart)
// ===================================

/*
// Dashboard Widget (Flutter)
class Dashboard extends StatefulWidget {
  @override
  _DashboardState createState() => _DashboardState();
}

class _DashboardState extends State<Dashboard> {
  Map<String, dynamic> dashboardData = {
    'total_cases': 8081,
    'countries_affected': 8,
    'critical_alerts': 2,
    'regions_monitored': 6
  };

  final String apiBaseUrl = 'http://192.168.1.100:8000';

  @override
  void initState() {
    super.initState();
    fetchDashboardData();
    // Auto-refresh every 30 seconds
    Timer.periodic(Duration(seconds: 30), (timer) {
      fetchDashboardData();
    });
  }

  Future<void> fetchDashboardData() async {
    try {
      final response = await http.get(Uri.parse('$apiBaseUrl/dashboard/stats'));
      if (response.statusCode == 200) {
        setState(() {
          dashboardData = json.decode(response.body);
        });
      }
    } catch (e) {
      print('Dashboard fetch error: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: RefreshIndicator(
        onRefresh: fetchDashboardData,
        child: GridView.count(
          crossAxisCount: 2,
          children: [
            _buildStatCard('Cases', dashboardData['total_cases'].toString()),
            _buildStatCard('Countries', dashboardData['countries_affected'].toString()),
            _buildStatCard('Critical', dashboardData['critical_alerts'].toString()),
            _buildStatCard('Regions', dashboardData['regions_monitored'].toString()),
          ],
        ),
      ),
    );
  }

  Widget _buildStatCard(String label, String value) {
    return Card(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(value, style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
          Text(label, style: TextStyle(fontSize: 16)),
        ],
      ),
    );
  }
}
*/

// ===================================
// VANILLA JAVASCRIPT (Web/Hybrid)
// ===================================

// Dashboard Functions (Vanilla JS)
const API_BASE_URL = 'http://localhost:8000';

async function updateDashboard() {
  try {
    const response = await fetch(`${API_BASE_URL}/dashboard/stats`);
    const data = await response.json();
    
    // Update DOM elements
    document.getElementById('total-cases').textContent = data.total_cases.toLocaleString();
    document.getElementById('countries-count').textContent = data.countries_affected;
    document.getElementById('critical-alerts').textContent = data.critical_alerts;
    document.getElementById('regions-count').textContent = data.regions_monitored;
    document.getElementById('network-status').textContent = data.network_status;
    
  } catch (error) {
    console.error('Dashboard update failed:', error);
    showErrorMessage('Failed to update dashboard data');
  }
}

async function updateTrends() {
  try {
    const response = await fetch(`${API_BASE_URL}/trends`);
    const data = await response.json();
    
    // Update disease cards
    const diseases = ['dengue', 'malaria', 'cholera', 'yellow', 'measles', 'influenza'];
    data.disease_breakdown.forEach((disease, index) => {
      if (index < diseases.length) {
        document.getElementById(`${diseases[index]}-cases`).textContent = disease.cases;
        document.getElementById(`${diseases[index]}-trend`).className = `trend-${disease.trend_direction}`;
      }
    });
    
    // Update line chart (assuming you have a chart library)
    updateLineChart(data.total_cases_trend);
    
  } catch (error) {
    console.error('Trends update failed:', error);
  }
}

async function updateAlerts(filter = 'all') {
  try {
    const endpoint = filter === 'all' ? '/alerts' : `/alerts/filter/${filter}`;
    const response = await fetch(`${API_BASE_URL}${endpoint}`);
    const data = await response.json();
    
    // Update filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`${filter}-btn`).classList.add('active');
    
    // Update alerts list
    const alertsContainer = document.getElementById('alerts-container');
    alertsContainer.innerHTML = '';
    
    data.alerts.forEach(alert => {
      const alertElement = createAlertElement(alert);
      alertsContainer.appendChild(alertElement);
    });
    
  } catch (error) {
    console.error('Alerts update failed:', error);
  }
}

function createAlertElement(alert) {
  const div = document.createElement('div');
  div.className = `alert-card ${alert.risk_level}-alert`;
  div.innerHTML = `
    <div class="alert-header">
      <h3>${alert.title}</h3>
      <span class="risk-badge ${alert.risk_level}">${alert.risk_level}</span>
    </div>
    <p class="alert-description">${alert.description}</p>
    <div class="alert-meta">
      <span class="location">${alert.location}, ${alert.country}</span>
      <span class="time">${alert.timestamp}</span>
    </div>
    <button class="investigate-btn" onclick="investigateAlert('${alert.id}')">
      Investigate Alert
    </button>
  `;
  return div;
}

async function investigateAlert(alertId) {
  try {
    const response = await fetch(`${API_BASE_URL}/alerts/investigate/${alertId}`, {
      method: 'POST'
    });
    const result = await response.json();
    showNotification(`Alert ${alertId} marked for investigation`);
  } catch (error) {
    showNotification('Failed to update alert status', 'error');
  }
}

// Auto-refresh functionality
let refreshInterval;

function startAutoRefresh() {
  refreshInterval = setInterval(() => {
    const currentTab = getCurrentTab();
    switch(currentTab) {
      case 'dashboard':
        updateDashboard();
        break;
      case 'trends':
        updateTrends();
        break;
      case 'alerts':
        updateAlerts(getCurrentFilter());
        break;
      case 'map':
        updateMap();
        break;
    }
  }, 30000); // Refresh every 30 seconds
}

function stopAutoRefresh() {
  if (refreshInterval) {
    clearInterval(refreshInterval);
  }
}

// Utility functions
function showNotification(message, type = 'success') {
  // Implement your notification system
  console.log(`${type.toUpperCase()}: ${message}`);
}

function showErrorMessage(message) {
  // Implement your error display
  console.error(message);
}

function getCurrentTab() {
  // Return current active tab
  return document.querySelector('.tab.active')?.dataset.tab || 'dashboard';
}

function getCurrentFilter() {
  // Return current alert filter
  return document.querySelector('.filter-btn.active')?.dataset.filter || 'all';
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
  updateDashboard();
  startAutoRefresh();
});

// ===================================
// STYLES (CSS)
// ===================================

const styles = `
.alert-card {
  background: #1a1a2e;
  border-radius: 12px;
  padding: 16px;
  margin: 8px 0;
  border-left: 4px solid;
}

.high-alert {
  border-left-color: #ff4444;
}

.moderate-alert {
  border-left-color: #ff8800;
}

.low-alert {
  border-left-color: #44ff44;
}

.filter-btn {
  background: #16213e;
  color: #ffffff;
  border: none;
  padding: 8px 16px;
  border-radius: 20px;
  margin: 4px;
}

.filter-btn.active {
  background: #0066ff;
}

.risk-badge {
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: bold;
}

.risk-badge.high {
  background: #ff4444;
  color: white;
}

.risk-badge.moderate {
  background: #ff8800;
  color: white;
}

.investigate-btn {
  background: #0066ff;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 6px;
  margin-top: 8px;
}
`;

// Export for use in mobile app
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    updateDashboard,
    updateTrends,
    updateAlerts,
    investigateAlert,
    startAutoRefresh,
    stopAutoRefresh
  };
}