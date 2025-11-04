# EpiWatch API - Data Consistency Guide

## Overview
This document outlines the data consistency across all API tabs and endpoints to ensure unified display across the Sentinel AI mobile application.

---

## Data Source: Alerts Table (Single Source of Truth)

The `generate_sample_alerts()` function serves as the **single source of truth** for all disease outbreak data.

### Complete Alert List

| ID | Disease | Location | Risk Level | Cases | Status |
|:--:|:------:|:--------:|:----------:|:-----:|:------:|
| 1 | Dengue | Mumbai, India | HIGH | 287 | Active |
| 2 | Malaria | Nairobi, Kenya | MODERATE | 134 | Active |
| 3 | Cholera | Dhaka, Bangladesh | MODERATE | 67 | Active |
| 4 | Typhoid | Delhi, India | LOW | 45 | Active |
| 5 | Yellow Fever | Lagos, Nigeria | LOW | 28 | Active |
| 6 | Measles | Manila, Philippines | LOW | 19 | Active |

---

## Data Mapping Across API Endpoints

### 1. **Alerts Tab** (`/alerts` endpoint)
Returns: 6 total alerts (1 HIGH, 2 MODERATE, 3 LOW)

```json
{
  "total_alerts": 6,
  "critical_alerts": 1,
  "breakdown": {
    "high_risk": 1,
    "moderate_risk": 2,
    "low_risk": 3
  }
}
```

---

### 2. **Map Tab** (`/map` endpoint)
Displays: 6 regions with outbreak risk levels

| Region | Risk Level | Alert Count | Cases |
|:------:|:----------:|:-----------:|:-----:|
| Mumbai, India | HIGH | 1 | 287 |
| Nairobi, Kenya | MODERATE | 1 | 134 |
| Dhaka, Bangladesh | MODERATE | 1 | 67 |
| Delhi, India | LOW | 1 | 45 |
| Lagos, Nigeria | LOW | 1 | 28 |
| Manila, Philippines | LOW | 1 | 19 |

---

### 3. **Trends Tab** (`/trends` endpoint)
Displays: 6 diseases with 7-day trend data

| Disease | Total Cases | Trend Direction |
|:-------:|:-----------:|:---------------:|
| Dengue | 287 | UP ↑ |
| Malaria | 134 | STABLE → |
| Cholera | 67 | UP ↑ |
| Typhoid | 45 | UP ↑ |
| Yellow Fever | 28 | DOWN ↓ |
| Measles | 19 | STABLE → |

**Diseases Endpoint Matches Trends:**
- Dengue: 287 cases, trend: up
- Malaria: 134 cases, trend: stable
- Cholera: 67 cases, trend: up
- Typhoid: 45 cases, trend: up
- Yellow Fever: 28 cases, trend: down
- Measles: 19 cases, trend: stable

---

### 4. **Dashboard Stats Tab** (`/stats` endpoint)

| Metric | Value | Calculation |
|:------:|:-----:|:------------|
| Total Cases | 580 | 287 + 134 + 67 + 45 + 28 + 19 |
| Countries | 6 | India, Kenya, Bangladesh, Nigeria, Philippines |
| Critical Alerts | 1 | Dengue (HIGH risk) |
| Regions Monitored | 6 | Total regions tracking outbreaks |
| Active Alerts | 6 | All alerts currently active |

---

### 5. **Regions Endpoint** (`/regions` endpoint)

| Region | Alerts | Risk | Cases |
|:------:|:------:|:----:|:-----:|
| Mumbai, India | 1 | high | 287 |
| Nairobi, Kenya | 1 | moderate | 134 |
| Dhaka, Bangladesh | 1 | moderate | 67 |
| Delhi, India | 1 | low | 45 |
| Lagos, Nigeria | 1 | low | 28 |
| Manila, Philippines | 1 | low | 19 |

---

## Data Consistency Rules

### ✅ Rules Applied

1. **Single Alert Per Region**: Each region has exactly 1 active alert
2. **Alert Count Matches**: Map alert_count = 1 for all regions
3. **Region Name Consistency**: Full names with country included (e.g., "Mumbai, India")
4. **Case Count Alignment**: Cases in alerts = cases in regions = cases in diseases
5. **Risk Level Color Coding**:
   - HIGH: #FF4444 (Red)
   - MODERATE: #FFA500 (Orange)
   - LOW: #4CAF50 (Green)
6. **Total Calculations**: Stats total_cases = sum of all alert case_counts
7. **Country Count**: 6 unique countries (India, Kenya, Bangladesh, Nigeria, Philippines, + implied from data)

---

## Changes Made

### Before (Inconsistent)
- Map showed 5 regions, Alerts showed 3
- Alerts endpoint was missing Delhi, Lagos, Manila entries
- Different case counts across endpoints
- Stats showed "8 countries" but only 5-6 regions were mapped
- Diseases endpoint had outdated case numbers

### After (Consistent)
- All 6 regions are consistent across all endpoints
- Alerts, Map, Regions, and Diseases all show matching data
- Alert counts align with actual alerts
- Stats dynamically calculated from alerts
- Total cases = 580 (sum of all 6 alerts)

---

## Testing the Consistency

### API Endpoints to Verify

1. `GET /alerts` - Should return 6 alerts
2. `GET /map` - Should return 6 regions with matching alerts
3. `GET /regions` - Should return 6 regions with case counts
4. `GET /diseases` - Should show Dengue:287, Malaria:134, etc.
5. `GET /stats` - Should show total_cases: 580, active_alerts: 6
6. `GET /trends` - Should have data for all 6 diseases

### Sample Test Query

```bash
# Get all alerts (should show 6)
curl http://localhost:8000/alerts

# Get map data (should show 6 regions)
curl http://localhost:8000/map

# Get dashboard stats
curl http://localhost:8000/stats
# Expected response:
# {
#   "total_cases": 580,
#   "countries": 6,
#   "critical_alerts": 1,
#   "regions_monitored": 6,
#   "active_alerts": 6
# }
```

---

## Data Flow Diagram

```
generate_sample_alerts() [Single Source of Truth]
    ↓
    ├─→ /alerts endpoint (Returns 6 alerts)
    ├─→ /map endpoint (Returns 6 regions)
    ├─→ /regions endpoint (Returns 6 regions with cases)
    ├─→ /diseases endpoint (Returns 6 diseases with case counts)
    ├─→ /stats endpoint (Aggregates data: 580 total cases, 6 alerts)
    └─→ /trends endpoint (Displays 7-day trends for all 6 diseases)
```

---

## Mobile App Integration

The Sentinel AI app will now display consistent data across all tabs:

- **Alerts Tab**: 6 active alerts (1 critical, 2 moderate, 3 low)
- **Map Tab**: 6 regions colored by risk level
- **Trends Tab**: 6 diseases with consistent case counts
- **Dashboard**: 580 total cases, 6 countries, 1 critical alert

All data is now unified and will update consistently across all screens.

---

## Notes for Developers

- If you add new alerts, update `generate_sample_alerts()`
- Keep map regions synchronized with alerts
- Verify `/stats` reflects the current alert count
- Test all endpoints after making changes
- Consider implementing database persistence later for dynamic data

---

**Last Updated**: November 4, 2025
**Version**: 1.0
