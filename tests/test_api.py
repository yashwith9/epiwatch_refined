"""
Test script for Epidemic Detection API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_api():
    """Test all API endpoints"""
    print("🧪 TESTING SENTINEL AI API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response: {response.json()}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Single prediction
    print("\n2. Testing Single Prediction...")
    try:
        data = {
            "text": "Outbreak of dengue fever reported in Mumbai with rising cases",
            "location": "Mumbai, India"
        }
        response = requests.post(f"{BASE_URL}/predict", json=data)
        print(f"✓ Status: {response.status_code}")
        result = response.json()
        print(f"✓ Prediction: {result['prediction']}")
        print(f"✓ Confidence: {result['confidence']:.3f}")
        print(f"✓ Risk Level: {result['risk_level']}")
        print(f"✓ Processing Time: {result['processing_time_ms']:.2f}ms")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Dashboard stats
    print("\n3. Testing Dashboard Stats...")
    try:
        response = requests.get(f"{BASE_URL}/dashboard/stats")
        print(f"✓ Status: {response.status_code}")
        stats = response.json()
        print(f"✓ Total Cases: {stats['total_cases']}")
        print(f"✓ Countries: {stats['countries_affected']}")
        print(f"✓ Critical Alerts: {stats['critical_alerts']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 4: Trends data
    print("\n4. Testing Trends Data...")
    try:
        response = requests.get(f"{BASE_URL}/trends")
        print(f"✓ Status: {response.status_code}")
        trends = response.json()
        print(f"✓ Weekly Data Points: {len(trends['weekly_data'])}")
        print(f"✓ Disease Breakdown: {len(trends['disease_breakdown'])}")
        print(f"✓ Top Disease: {trends['disease_breakdown'][0]['disease']} ({trends['disease_breakdown'][0]['cases']} cases)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 5: Map data
    print("\n5. Testing Map Data...")
    try:
        response = requests.get(f"{BASE_URL}/map/data")
        print(f"✓ Status: {response.status_code}")
        map_data = response.json()
        print(f"✓ Active Outbreaks: {len(map_data['active_outbreaks'])}")
        print(f"✓ Risk Zones: {len(map_data['risk_zones'])}")
        print(f"✓ Global Status: {map_data['global_status']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 6: Alerts
    print("\n6. Testing Alerts...")
    try:
        response = requests.get(f"{BASE_URL}/alerts")
        print(f"✓ Status: {response.status_code}")
        alerts = response.json()
        print(f"✓ Total Alerts: {alerts['total_alerts']}")
        print(f"✓ Critical: {alerts['critical_count']}")
        print(f"✓ Moderate: {alerts['moderate_count']}")
        if alerts['alerts']:
            print(f"✓ Latest Alert: {alerts['alerts'][0]['title']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 7: Model info
    print("\n7. Testing Model Info...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"✓ Status: {response.status_code}")
        model_info = response.json()
        print(f"✓ Model: {model_info['model_name']}")
        print(f"✓ Accuracy: {model_info['performance']['accuracy']}")
        print(f"✓ F1-Score: {model_info['performance']['f1_score']}")
        print(f"✓ Inference Speed: {model_info['performance']['inference_speed_ms']}ms")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API TESTING COMPLETE!")
    print("=" * 50)

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n🔄 TESTING BATCH PREDICTION")
    print("-" * 30)
    
    try:
        data = {
            "texts": [
                "Dengue outbreak reported in Mumbai",
                "Weather is nice today",
                "COVID-19 cases rising in Delhi",
                "Stock market update",
                "Malaria spreading in rural areas"
            ]
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/predict/batch", json=data)
        end_time = time.time()
        
        print(f"✓ Status: {response.status_code}")
        result = response.json()
        print(f"✓ Processed: {result['total_processed']} texts")
        print(f"✓ Average Confidence: {result['average_confidence']:.3f}")
        print(f"✓ Total Time: {(end_time - start_time) * 1000:.2f}ms")
        
        # Show individual predictions
        for i, pred in enumerate(result['predictions']):
            print(f"  Text {i+1}: Prediction={pred['prediction']}, Confidence={pred['confidence']:.3f}, Risk={pred['risk_level']}")
            
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    print("Starting API tests...")
    print("Make sure the API is running on http://localhost:8000")
    print()
    
    # Wait a moment for user to start API
    input("Press Enter when API is running...")
    
    test_api()
    test_batch_prediction()