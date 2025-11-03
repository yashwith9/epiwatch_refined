"""
Test Mobile Connection to API
Run this to verify your mobile app can connect to the API
"""

import socket
import requests
import json
import time

def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a remote server to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "Unable to determine IP"

def test_api_connection(ip_address, port=8000):
    """Test if API is accessible from mobile devices"""
    base_url = f"http://{ip_address}:{port}"
    
    print("🧪 TESTING MOBILE API CONNECTION")
    print("=" * 50)
    print(f"📱 Mobile devices should connect to: {base_url}")
    print("=" * 50)
    
    # Test endpoints that mobile app will use
    endpoints = [
        "/",
        "/dashboard/stats", 
        "/alerts",
        "/trends",
        "/map/data"
    ]
    
    results = {}
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        try:
            print(f"\n🔍 Testing: {endpoint}")
            start_time = time.time()
            response = requests.get(url, timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                print(f"✅ SUCCESS - {response.status_code} ({response_time:.1f}ms)")
                results[endpoint] = "✅ Working"
                
                # Show sample data for key endpoints
                if endpoint == "/dashboard/stats":
                    data = response.json()
                    print(f"   📊 Cases: {data['total_cases']}, Countries: {data['countries_affected']}")
                elif endpoint == "/alerts":
                    data = response.json()
                    print(f"   🚨 Total Alerts: {data['total_alerts']}, Critical: {data['critical_count']}")
                    
            else:
                print(f"❌ FAILED - HTTP {response.status_code}")
                results[endpoint] = f"❌ HTTP {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            print(f"❌ CONNECTION REFUSED - API not running?")
            results[endpoint] = "❌ Connection refused"
        except requests.exceptions.Timeout:
            print(f"❌ TIMEOUT - API too slow")
            results[endpoint] = "❌ Timeout"
        except Exception as e:
            print(f"❌ ERROR - {str(e)}")
            results[endpoint] = f"❌ {str(e)}"
    
    return results

def generate_mobile_config(ip_address, port=8000):
    """Generate mobile app configuration"""
    config = {
        "API_BASE_URL": f"http://{ip_address}:{port}",
        "endpoints": {
            "dashboard": "/dashboard/stats",
            "alerts": "/alerts",
            "trends": "/trends", 
            "map": "/map/data",
            "predict": "/predict"
        },
        "refresh_interval": 30000,
        "timeout": 5000
    }
    
    return config

def main():
    print("🏥 SENTINEL AI - MOBILE CONNECTION TEST")
    print("=" * 60)
    
    # Get local IP
    local_ip = get_local_ip()
    print(f"🖥️  Your computer's IP address: {local_ip}")
    
    if local_ip == "Unable to determine IP":
        print("❌ Could not determine IP address")
        print("💡 Manually check with: ipconfig (Windows) or ifconfig (Mac/Linux)")
        return
    
    # Test API connection
    results = test_api_connection(local_ip)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 CONNECTION TEST SUMMARY")
    print("=" * 60)
    
    working_count = sum(1 for result in results.values() if "✅" in result)
    total_count = len(results)
    
    for endpoint, status in results.items():
        print(f"{endpoint:<20} {status}")
    
    print(f"\n📊 Results: {working_count}/{total_count} endpoints working")
    
    if working_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your mobile app can connect to the API")
        
        # Generate mobile config
        config = generate_mobile_config(local_ip)
        
        print(f"\n📱 MOBILE APP CONFIGURATION:")
        print("=" * 40)
        print("Copy this into your mobile app:")
        print()
        print("JavaScript/React Native:")
        print(f"const API_BASE_URL = '{config['API_BASE_URL']}';")
        print()
        print("Flutter/Dart:")
        print(f"final String apiBaseUrl = '{config['API_BASE_URL']}';")
        print()
        
        # Save config to file
        with open("mobile_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("💾 Configuration saved to: mobile_config.json")
        
    else:
        print("\n❌ SOME TESTS FAILED")
        print("🔧 Troubleshooting:")
        print("   1. Make sure API is running: python epidemic_api.py")
        print("   2. Check firewall settings")
        print("   3. Ensure both devices on same WiFi network")
    
    print(f"\n🌐 Test your connection from mobile browser:")
    print(f"   Open: http://{local_ip}:8000")
    print("   You should see the API welcome message")

if __name__ == "__main__":
    main()