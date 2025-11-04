#!/usr/bin/env python3
"""
Data Consistency Verification Script
Tests all API endpoints to ensure data matches across tabs
"""

import json
import requests
from typing import Dict, List

class DataConsistencyTester:
    """Verify data consistency across API endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            "status": "PENDING",
            "tests": {},
            "mismatches": []
        }
    
    def run_all_tests(self):
        """Run all consistency tests"""
        print("=" * 70)
        print("🔍 EpiWatch API - Data Consistency Verification")
        print("=" * 70)
        
        try:
            # Get all data
            alerts = self._get_alerts()
            map_data = self._get_map_data()
            regions = self._get_regions()
            diseases = self._get_diseases()
            stats = self._get_stats()
            
            # Run consistency tests
            self._test_alert_count(alerts)
            self._test_map_consistency(alerts, map_data)
            self._test_region_consistency(alerts, regions)
            self._test_disease_consistency(alerts, diseases)
            self._test_stats_consistency(alerts, stats)
            
            # Print results
            self._print_results()
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            self.results["status"] = "FAILED"
    
    def _get_alerts(self) -> List[Dict]:
        """Get alerts from API"""
        print("\n📡 Fetching /alerts...")
        response = requests.get(f"{self.base_url}/alerts")
        response.raise_for_status()
        alerts = response.json()
        print(f"   ✓ Got {len(alerts)} alerts")
        return alerts
    
    def _get_map_data(self) -> List[Dict]:
        """Get map data from API"""
        print("📡 Fetching /map...")
        response = requests.get(f"{self.base_url}/map")
        response.raise_for_status()
        map_data = response.json()
        print(f"   ✓ Got {len(map_data)} map regions")
        return map_data
    
    def _get_regions(self) -> Dict:
        """Get regions from API"""
        print("📡 Fetching /regions...")
        response = requests.get(f"{self.base_url}/regions")
        response.raise_for_status()
        data = response.json()
        regions = data.get('regions', [])
        print(f"   ✓ Got {len(regions)} regions")
        return regions
    
    def _get_diseases(self) -> Dict:
        """Get diseases from API"""
        print("📡 Fetching /diseases...")
        response = requests.get(f"{self.base_url}/diseases")
        response.raise_for_status()
        data = response.json()
        diseases = data.get('diseases', [])
        print(f"   ✓ Got {len(diseases)} diseases")
        return diseases
    
    def _get_stats(self) -> Dict:
        """Get stats from API"""
        print("📡 Fetching /stats...")
        response = requests.get(f"{self.base_url}/stats")
        response.raise_for_status()
        stats = response.json()
        print(f"   ✓ Got stats data")
        return stats
    
    def _test_alert_count(self, alerts: List[Dict]):
        """Test 1: Alert count validation"""
        print("\n🧪 Test 1: Alert Count Validation")
        
        test_name = "alert_count"
        expected = 6
        actual = len(alerts)
        
        passed = actual == expected
        self.results["tests"][test_name] = {
            "expected": expected,
            "actual": actual,
            "passed": passed
        }
        
        if passed:
            print(f"   ✅ PASS: {actual} alerts found (expected {expected})")
        else:
            print(f"   ❌ FAIL: {actual} alerts found (expected {expected})")
            self.results["mismatches"].append(f"Alert count mismatch: {actual} vs {expected}")
    
    def _test_map_consistency(self, alerts: List[Dict], map_data: List[Dict]):
        """Test 2: Map data matches alerts"""
        print("\n🧪 Test 2: Map Data Consistency")
        
        alert_locations = {alert['location'] for alert in alerts}
        map_regions = {region['region'] for region in map_data}
        
        test_name = "map_consistency"
        
        # Check if all alert locations are in map
        missing_in_map = alert_locations - map_regions
        
        passed = len(missing_in_map) == 0
        self.results["tests"][test_name] = {
            "alert_locations": list(alert_locations),
            "map_regions": list(map_regions),
            "missing_in_map": list(missing_in_map),
            "passed": passed
        }
        
        if passed:
            print(f"   ✅ PASS: All {len(alert_locations)} alert locations are in map")
        else:
            print(f"   ❌ FAIL: Missing in map: {missing_in_map}")
            self.results["mismatches"].append(f"Map missing regions: {missing_in_map}")
    
    def _test_region_consistency(self, alerts: List[Dict], regions: List[Dict]):
        """Test 3: Region data matches alerts"""
        print("\n🧪 Test 3: Region Data Consistency")
        
        alert_by_location = {a['location']: a['case_count'] for a in alerts}
        region_by_name = {r['name']: r['cases'] for r in regions}
        
        test_name = "region_consistency"
        mismatches = []
        
        for location, case_count in alert_by_location.items():
            if location in region_by_name:
                if region_by_name[location] != case_count:
                    mismatches.append(f"{location}: alert={case_count}, region={region_by_name[location]}")
        
        passed = len(mismatches) == 0
        self.results["tests"][test_name] = {
            "passed": passed,
            "mismatches": mismatches
        }
        
        if passed:
            print(f"   ✅ PASS: Region case counts match alerts")
        else:
            print(f"   ❌ FAIL: Case count mismatches:")
            for mismatch in mismatches:
                print(f"      - {mismatch}")
            self.results["mismatches"].extend(mismatches)
    
    def _test_disease_consistency(self, alerts: List[Dict], diseases: List[Dict]):
        """Test 4: Disease data matches alerts"""
        print("\n🧪 Test 4: Disease Data Consistency")
        
        # Build disease count from alerts
        disease_count_from_alerts = {}
        for alert in alerts:
            # Extract disease name from title (e.g., "Dengue Fever Alert" -> "Dengue")
            title = alert['title'].split()[0]  # Get first word as disease name
            if title not in disease_count_from_alerts:
                disease_count_from_alerts[title] = alert['case_count']
        
        disease_by_name = {d['name']: d['cases'] for d in diseases}
        
        test_name = "disease_consistency"
        mismatches = []
        
        for disease, case_count in disease_count_from_alerts.items():
            if disease in disease_by_name:
                if disease_by_name[disease] != case_count:
                    mismatches.append(f"{disease}: alert={case_count}, disease={disease_by_name[disease]}")
        
        passed = len(mismatches) == 0
        self.results["tests"][test_name] = {
            "passed": passed,
            "mismatches": mismatches
        }
        
        if passed:
            print(f"   ✅ PASS: Disease case counts match alerts")
        else:
            print(f"   ❌ FAIL: Case count mismatches:")
            for mismatch in mismatches:
                print(f"      - {mismatch}")
            self.results["mismatches"].extend(mismatches)
    
    def _test_stats_consistency(self, alerts: List[Dict], stats: Dict):
        """Test 5: Stats data matches alerts"""
        print("\n🧪 Test 5: Stats Data Consistency")
        
        expected_total = sum(a['case_count'] for a in alerts)
        expected_alerts = len(alerts)
        expected_high_risk = len([a for a in alerts if a['risk_level'] == 'high'])
        
        actual_total = stats['total_cases']
        actual_alerts = stats['active_alerts']
        actual_high_risk = stats['critical_alerts']
        
        test_name = "stats_consistency"
        mismatches = []
        
        if actual_total != expected_total:
            mismatches.append(f"Total cases: expected={expected_total}, actual={actual_total}")
        if actual_alerts != expected_alerts:
            mismatches.append(f"Active alerts: expected={expected_alerts}, actual={actual_alerts}")
        if actual_high_risk != expected_high_risk:
            mismatches.append(f"Critical alerts: expected={expected_high_risk}, actual={actual_high_risk}")
        
        passed = len(mismatches) == 0
        self.results["tests"][test_name] = {
            "total_cases": {"expected": expected_total, "actual": actual_total},
            "active_alerts": {"expected": expected_alerts, "actual": actual_alerts},
            "critical_alerts": {"expected": expected_high_risk, "actual": actual_high_risk},
            "passed": passed
        }
        
        if passed:
            print(f"   ✅ PASS: Stats data matches alerts")
            print(f"      - Total cases: {actual_total}")
            print(f"      - Active alerts: {actual_alerts}")
            print(f"      - Critical alerts: {actual_high_risk}")
        else:
            print(f"   ❌ FAIL: Stats mismatches:")
            for mismatch in mismatches:
                print(f"      - {mismatch}")
            self.results["mismatches"].extend(mismatches)
    
    def _print_results(self):
        """Print summary of test results"""
        print("\n" + "=" * 70)
        print("📊 CONSISTENCY TEST SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for t in self.results["tests"].values() if t.get("passed", False))
        failed_tests = total_tests - passed_tests
        
        print(f"\n📈 Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        
        if self.results["mismatches"]:
            print(f"\n⚠️  Mismatches Found ({len(self.results['mismatches'])}):")
            for mismatch in self.results["mismatches"]:
                print(f"   - {mismatch}")
            self.results["status"] = "FAILED"
        else:
            print(f"\n✅ All tests passed! Data is consistent across all endpoints.")
            self.results["status"] = "PASSED"
        
        print("\n" + "=" * 70)
        print("\n📋 Detailed Results:")
        print(json.dumps(self.results, indent=2))
        
        # Save results to file
        with open("data_consistency_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to: data_consistency_results.json")


if __name__ == "__main__":
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    tester = DataConsistencyTester(base_url)
    tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if tester.results["status"] == "PASSED" else 1)
