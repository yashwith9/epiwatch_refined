# 📱 How to Connect API to Mobile App - Super Simple Guide

## 🎯 **3 Easy Steps to Connect Your API**

---

## **STEP 1: Find Your Computer's IP Address**

### On Windows:
1. Press `Windows + R`
2. Type `cmd` and press Enter
3. Type `ipconfig` and press Enter
4. Look for "IPv4 Address" - it will be something like `192.168.1.100`

### On Mac:
1. Open Terminal
2. Type `ifconfig` and press Enter
3. Look for "inet" under your WiFi connection

### Example Output:
```
IPv4 Address: 192.168.1.100  ← This is what you need!
```

---

## **STEP 2: Make Sure API is Running**

1. **Start your API server:**
   ```bash
   python epidemic_api.py
   ```

2. **You should see:**
   ```
   🚀 Starting Sentinel AI Epidemic Detection API
   INFO: Uvicorn running on http://0.0.0.0:8000
   ```

3. **Test it works:**
   - Open browser
   - Go to: `http://localhost:8000`
   - You should see: `{"message":"Sentinel AI - Epidemic Detection API"}`

---

## **STEP 3: Update Your Mobile App Code**

### **Option A: React Native App**

Find your mobile app code and add this:

```javascript
// At the top of your main component file
const API_BASE_URL = 'http://192.168.1.100:8000'; // Replace with YOUR IP

// Replace your hardcoded dashboard data with this:
const [dashboardData, setDashboardData] = useState({
  total_cases: 8081,
  countries_affected: 8,
  critical_alerts: 2,
  regions_monitored: 6
});

// Add this function to fetch real data:
const fetchDashboardData = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/dashboard/stats`);
    const data = await response.json();
    setDashboardData(data);
    console.log('✅ Dashboard data updated:', data);
  } catch (error) {
    console.error('❌ API connection failed:', error);
    // Keep using hardcoded data if API fails
  }
};

// Call this when your component loads:
useEffect(() => {
  fetchDashboardData();
  // Auto-refresh every 30 seconds
  const interval = setInterval(fetchDashboardData, 30000);
  return () => clearInterval(interval);
}, []);
```

### **Option B: Flutter App**

Add this to your Flutter app:

```dart
class _DashboardState extends State<Dashboard> {
  final String apiBaseUrl = 'http://192.168.1.100:8000'; // Replace with YOUR IP
  
  Map<String, dynamic> dashboardData = {
    'total_cases': 8081,
    'countries_affected': 8,
    'critical_alerts': 2,
    'regions_monitored': 6
  };

  Future<void> fetchDashboardData() async {
    try {
      final response = await http.get(Uri.parse('$apiBaseUrl/dashboard/stats'));
      if (response.statusCode == 200) {
        setState(() {
          dashboardData = json.decode(response.body);
        });
        print('✅ Dashboard data updated: $dashboardData');
      }
    } catch (e) {
      print('❌ API connection failed: $e');
      // Keep using hardcoded data if API fails
    }
  }

  @override
  void initState() {
    super.initState();
    fetchDashboardData();
    // Auto-refresh every 30 seconds
    Timer.periodic(Duration(seconds: 30), (timer) => fetchDashboardData());
  }
}
```

### **Option C: JavaScript/HTML App**

Add this to your web app:

```javascript
const API_BASE_URL = 'http://192.168.1.100:8000'; // Replace with YOUR IP

async function updateDashboard() {
  try {
    const response = await fetch(`${API_BASE_URL}/dashboard/stats`);
    const data = await response.json();
    
    // Update your HTML elements
    document.getElementById('total-cases').textContent = data.total_cases;
    document.getElementById('countries').textContent = data.countries_affected;
    document.getElementById('critical-alerts').textContent = data.critical_alerts;
    document.getElementById('regions').textContent = data.regions_monitored;
    
    console.log('✅ Dashboard updated:', data);
  } catch (error) {
    console.error('❌ API connection failed:', error);
  }
}

// Call this when page loads
updateDashboard();
// Auto-refresh every 30 seconds
setInterval(updateDashboard, 30000);
```

---

## **🧪 TESTING: Make Sure It Works**

### **Test 1: Check API from Mobile Browser**
1. On your phone, open browser
2. Go to: `http://192.168.1.100:8000` (use YOUR IP)
3. You should see the API response

### **Test 2: Check Your Mobile App**
1. Run your mobile app
2. Check the console/logs for:
   - ✅ `Dashboard data updated:` - SUCCESS!
   - ❌ `API connection failed:` - Need to fix connection

### **Test 3: Verify Data Updates**
1. Your app should show the same numbers as the API
2. API shows: `{"total_cases": 8081}` 
3. Your app should show: `8,081` cases

---

## **🔧 TROUBLESHOOTING**

### **Problem: "Network request failed"**
**Solution:** Make sure both devices are on same WiFi network

### **Problem: "Connection refused"**
**Solutions:**
1. Check API is running: `python epidemic_api.py`
2. Check firewall isn't blocking port 8000
3. Try different IP address from `ipconfig`

### **Problem: "CORS error"**
**Solution:** API already has CORS enabled, but try restarting API server

### **Problem: App shows old data**
**Solution:** Check console logs - API might be failing silently

---

## **📱 COMPLETE EXAMPLE: Dashboard Connection**

Here's a complete working example:

```javascript
// Complete Dashboard Component
import React, { useState, useEffect } from 'react';

const Dashboard = () => {
  // Replace 192.168.1.100 with YOUR computer's IP
  const API_BASE_URL = 'http://192.168.1.100:8000';
  
  const [data, setData] = useState({
    total_cases: 8081,      // Default values
    countries_affected: 8,
    critical_alerts: 2,
    regions_monitored: 6
  });
  
  const [isConnected, setIsConnected] = useState(false);

  const fetchData = async () => {
    try {
      console.log('🔄 Fetching data from API...');
      const response = await fetch(`${API_BASE_URL}/dashboard/stats`);
      const apiData = await response.json();
      
      setData(apiData);
      setIsConnected(true);
      console.log('✅ API connected! Data:', apiData);
      
    } catch (error) {
      setIsConnected(false);
      console.error('❌ API connection failed:', error);
      console.log('📱 Using offline data');
    }
  };

  useEffect(() => {
    fetchData(); // Fetch immediately
    const interval = setInterval(fetchData, 30000); // Every 30 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      {/* Connection Status */}
      <div style={{color: isConnected ? 'green' : 'orange'}}>
        {isConnected ? '🟢 Live Data' : '🟡 Offline Mode'}
      </div>
      
      {/* Your existing dashboard UI */}
      <div className="stats-grid">
        <div className="stat-card">
          <h2>{data.total_cases.toLocaleString()}</h2>
          <p>Cases</p>
        </div>
        <div className="stat-card">
          <h2>{data.countries_affected}</h2>
          <p>Countries</p>
        </div>
        <div className="stat-card">
          <h2>{data.critical_alerts}</h2>
          <p>Critical</p>
        </div>
        <div className="stat-card">
          <h2>{data.regions_monitored}</h2>
          <p>Regions</p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
```

---

## **🎉 SUCCESS! Your App is Now Connected**

When working correctly, you'll see:
- ✅ Your mobile app shows live data from the API
- ✅ Numbers update automatically every 30 seconds
- ✅ Console shows "API connected!" messages
- ✅ Green "Live Data" indicator

**Your mobile app is now powered by AI epidemic detection! 🚀**

---

## **🚀 NEXT STEPS**

1. **Connect other tabs** using the same pattern:
   - Alerts: `${API_BASE_URL}/alerts`
   - Trends: `${API_BASE_URL}/trends`
   - Map: `${API_BASE_URL}/map/data`

2. **Deploy to production** when ready:
   - Host API on cloud server
   - Update `API_BASE_URL` to production URL

3. **Add more features**:
   - Real-time notifications
   - Offline data caching
   - User authentication

**You're all set! 🎯**