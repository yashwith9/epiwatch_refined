import requests
import json

# Test prediction endpoint
data = {
    "text": "Outbreak of dengue fever reported in Mumbai with rising cases",
    "location": "Mumbai, India"
}

response = requests.post("http://localhost:8000/predict", json=data)
print("Prediction Test:")
print(json.dumps(response.json(), indent=2))

# Test dashboard
response = requests.get("http://localhost:8000/dashboard/stats")
print("\nDashboard Stats:")
print(json.dumps(response.json(), indent=2))

# Test alerts
response = requests.get("http://localhost:8000/alerts")
print("\nAlerts:")
alerts = response.json()
print(f"Total alerts: {alerts['total_alerts']}")
print(f"Critical: {alerts['critical_count']}")
for alert in alerts['alerts'][:2]:  # Show first 2 alerts
    print(f"- {alert['title']} ({alert['risk_level']})")