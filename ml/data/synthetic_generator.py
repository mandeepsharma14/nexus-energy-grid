"""
NexusGrid Synthetic Data Generator
Generates realistic energy datasets for all demo organizations.

© 2026 Mandeep Sharma. All rights reserved.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import random

np.random.seed(42)
random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# ORGANIZATION PROFILES
# ─────────────────────────────────────────────
ORGANIZATIONS = {
    "arcelor_steel": {
        "name": "ArcelorSteel Manufacturing",
        "type": "industrial",
        "base_load_kw": 6000,
        "peak_multiplier": 1.45,
        "num_facilities": 5,
        "tariff": "industrial_demand",
        "cost_per_kwh": 0.0674,
        "demand_charge_per_kw": 12.80,
    },
    "megamart_retail": {
        "name": "MegaMart Retail Chain",
        "type": "retail",
        "base_load_kw": 3800,
        "peak_multiplier": 1.30,
        "num_facilities": 12,
        "tariff": "commercial_tou",
        "cost_per_kwh": 0.0820,
        "demand_charge_per_kw": 9.40,
    },
    "primeplex_cre": {
        "name": "PrimePlex Commercial RE",
        "type": "commercial",
        "base_load_kw": 2100,
        "peak_multiplier": 1.25,
        "num_facilities": 8,
        "tariff": "commercial_flat",
        "cost_per_kwh": 0.0910,
        "demand_charge_per_kw": 8.20,
    },
    "swiftlogix": {
        "name": "SwiftLogix Warehouses",
        "type": "logistics",
        "base_load_kw": 2800,
        "peak_multiplier": 1.35,
        "num_facilities": 6,
        "tariff": "industrial_tou",
        "cost_per_kwh": 0.0750,
        "demand_charge_per_kw": 11.20,
    },
    "vertex_dc": {
        "name": "Vertex Data Centers",
        "type": "data_center",
        "base_load_kw": 11000,
        "peak_multiplier": 1.15,
        "num_facilities": 3,
        "tariff": "large_power",
        "cost_per_kwh": 0.0580,
        "demand_charge_per_kw": 14.60,
    },
}

FACILITIES = {
    "arcelor_steel": [
        {"id": "F001", "name": "Plant Alpha", "city": "Columbus", "state": "OH", "sqft": 420000, "base_frac": 0.38},
        {"id": "F002", "name": "Plant Beta",  "city": "Pittsburgh", "state": "PA", "sqft": 310000, "base_frac": 0.29},
        {"id": "F003", "name": "Plant Gamma", "city": "Detroit",    "state": "MI", "sqft": 250000, "base_frac": 0.21},
        {"id": "F004", "name": "HQ Office",   "city": "Chicago",    "state": "IL", "sqft": 85000,  "base_frac": 0.07},
        {"id": "F005", "name": "R&D Center",  "city": "Indianapolis","state": "IN","sqft": 60000,  "base_frac": 0.05},
    ],
}


# ─────────────────────────────────────────────
# LOAD PROFILE SHAPE (24h, normalized 0-1)
# ─────────────────────────────────────────────
def industrial_load_shape():
    """Returns a 24-element array of fractional load (0-1)."""
    base = [
        0.42, 0.40, 0.38, 0.37, 0.38, 0.45,  # 0-5 AM
        0.68, 0.88, 0.96, 0.99, 1.00, 0.98,  # 6-11 AM
        0.97, 0.96, 1.00, 0.99, 0.97, 0.94,  # 12-17 PM
        0.88, 0.82, 0.74, 0.66, 0.58, 0.50,  # 18-23 PM
    ]
    return np.array(base)

def retail_load_shape():
    base = [
        0.25, 0.22, 0.20, 0.20, 0.22, 0.30,
        0.48, 0.72, 0.88, 0.96, 1.00, 0.99,
        0.98, 0.97, 0.98, 0.99, 0.97, 0.94,
        0.88, 0.78, 0.60, 0.45, 0.35, 0.28,
    ]
    return np.array(base)

LOAD_SHAPES = {
    "industrial": industrial_load_shape(),
    "retail": retail_load_shape(),
    "commercial": retail_load_shape() * 0.9,
    "logistics": industrial_load_shape() * 0.85,
    "data_center": np.full(24, 0.92),  # Flat load, high utilization
}


# ─────────────────────────────────────────────
# SEASONALITY + WEATHER FACTOR
# ─────────────────────────────────────────────
def monthly_seasonality(org_type: str) -> np.ndarray:
    """Returns 12-month seasonal multipliers (Jan=0)."""
    if org_type in ("industrial", "logistics"):
        # Industry: slightly up in winter (heating) and summer (cooling)
        return np.array([1.05, 1.00, 0.96, 0.94, 0.98, 1.04, 1.10, 1.12, 1.04, 0.96, 0.98, 1.02])
    elif org_type == "retail":
        # Retail: high Nov-Jan (holiday season)
        return np.array([1.08, 0.95, 0.92, 0.90, 0.94, 0.98, 1.04, 1.06, 0.98, 0.96, 1.10, 1.18])
    elif org_type == "data_center":
        # Data center: high summer (cooling load)
        return np.array([0.96, 0.94, 0.95, 0.97, 1.02, 1.06, 1.10, 1.10, 1.04, 0.98, 0.96, 0.96])
    else:
        return np.array([1.02, 0.98, 0.96, 0.95, 0.98, 1.03, 1.06, 1.07, 1.01, 0.97, 0.99, 1.01])


# ─────────────────────────────────────────────
# INTERVAL DATA GENERATOR (15-min)
# ─────────────────────────────────────────────
def generate_interval_data(
    org_id: str,
    facility: dict,
    org: dict,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Generate 15-minute interval energy readings with noise, seasonality, anomalies."""
    dates = pd.date_range(start=start_date, end=end_date, freq="15min")
    n = len(dates)
    shape = LOAD_SHAPES.get(org["type"], LOAD_SHAPES["industrial"])
    season = monthly_seasonality(org["type"])

    base_kw = org["base_load_kw"] * facility["base_frac"]

    kw_values = []
    for ts in dates:
        hour_frac = shape[ts.hour]
        month_frac = season[ts.month - 1]

        # Weekend factor
        weekend = 0.62 if ts.weekday() >= 5 else 1.0

        # Noise
        noise = np.random.normal(1.0, 0.03)

        kw = base_kw * hour_frac * month_frac * weekend * noise
        kw_values.append(max(kw, base_kw * 0.18))  # minimum floor

    kw_arr = np.array(kw_values)

    # ── Inject anomalies ──────────────────────────────────────
    num_anomalies = random.randint(3, 8)
    anomaly_indices = random.sample(range(n // 4, 3 * n // 4), num_anomalies)
    for idx in anomaly_indices:
        severity = random.choice([2.8, 3.4, 4.5, 0.15, 0.20])
        kw_arr[idx] = kw_arr[idx] * severity

    kwh = kw_arr * 0.25  # 15-min interval → kWh
    cost_per_kwh = org["cost_per_kwh"] * np.random.uniform(0.94, 1.06, n)
    cost = kwh * cost_per_kwh

    df = pd.DataFrame({
        "timestamp": dates,
        "facility_id": facility["id"],
        "facility_name": facility["name"],
        "kw": np.round(kw_arr, 2),
        "kwh": np.round(kwh, 4),
        "cost_usd": np.round(cost, 4),
        "is_anomaly": [i in anomaly_indices for i in range(n)],
    })
    return df


# ─────────────────────────────────────────────
# MONTHLY BILLING GENERATOR
# ─────────────────────────────────────────────
def generate_monthly_bills(
    org_id: str,
    interval_df: pd.DataFrame,
    org: dict,
) -> pd.DataFrame:
    """Aggregate interval data into monthly utility bills with demand charges."""
    interval_df["month"] = interval_df["timestamp"].dt.to_period("M")
    monthly = (
        interval_df.groupby(["facility_id", "facility_name", "month"])
        .agg(
            total_kwh=("kwh", "sum"),
            peak_kw=("kw", "max"),
            avg_kw=("kw", "mean"),
            energy_cost=("cost_usd", "sum"),
        )
        .reset_index()
    )
    monthly["demand_charge"] = monthly["peak_kw"] * org["demand_charge_per_kw"]
    monthly["distribution_charge"] = monthly["total_kwh"] * 0.0124
    monthly["taxes_fees"] = (monthly["energy_cost"] + monthly["demand_charge"]) * 0.078
    monthly["total_bill"] = (
        monthly["energy_cost"]
        + monthly["demand_charge"]
        + monthly["distribution_charge"]
        + monthly["taxes_fees"]
    )
    monthly["carbon_tco2"] = monthly["total_kwh"] * 0.000386  # EPA eGRID avg
    monthly["blended_rate"] = monthly["total_bill"] / monthly["total_kwh"]
    monthly["load_factor"] = (monthly["avg_kw"] / monthly["peak_kw"]).round(4)

    # Inject 2 billing errors
    if len(monthly) > 4:
        err_idx = random.sample(range(len(monthly)), 2)
        for i in err_idx:
            monthly.loc[i, "billing_error"] = True
            monthly.loc[i, "overcharge_usd"] = round(random.uniform(800, 4200), 2)
    monthly["billing_error"] = monthly.get("billing_error", False).fillna(False)
    monthly["overcharge_usd"] = monthly.get("overcharge_usd", 0.0).fillna(0.0)

    return monthly.round(4)


# ─────────────────────────────────────────────
# FORECAST DATA GENERATOR
# ─────────────────────────────────────────────
def generate_forecasts(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Generate AI forecast outputs for 7/30/90 day horizons."""
    rows = []
    last_month_kwh = monthly_df.groupby("month")["total_kwh"].sum().iloc[-1]

    for horizon in [7, 30, 90]:
        noise_scale = 0.03 if horizon == 7 else 0.06 if horizon == 30 else 0.11
        point = last_month_kwh * (horizon / 30) * np.random.normal(1.02, noise_scale)
        lower = point * 0.88
        upper = point * 1.12
        accuracy = 0.96 - (horizon / 30) * 0.025

        rows.append({
            "horizon_days": horizon,
            "forecast_kwh": round(point, 0),
            "lower_80pct": round(lower, 0),
            "upper_80pct": round(upper, 0),
            "model": "LSTM+Prophet_Ensemble",
            "accuracy_backtest": round(accuracy, 4),
            "forecast_generated_at": datetime.now().isoformat(),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# CARBON / EMISSIONS GENERATOR
# ─────────────────────────────────────────────
def generate_emissions(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Generate Scope 1, 2, 3 emissions data aligned to billing months."""
    monthly_totals = monthly_df.groupby("month")["total_kwh"].sum().reset_index()
    monthly_totals.columns = ["month", "total_kwh"]

    # eGRID emission factors (tCO2/MWh)
    egrid_factor = 0.386

    monthly_totals["scope2_tco2"] = (monthly_totals["total_kwh"] / 1000 * egrid_factor).round(2)
    monthly_totals["scope1_tco2"] = (monthly_totals["scope2_tco2"] * 0.42).round(2)  # natural gas boilers
    monthly_totals["scope3_tco2"] = (monthly_totals["scope2_tco2"] * 0.18).round(2)  # upstream
    monthly_totals["total_tco2"] = (
        monthly_totals["scope1_tco2"] + monthly_totals["scope2_tco2"] + monthly_totals["scope3_tco2"]
    ).round(2)
    monthly_totals["carbon_intensity"] = (
        monthly_totals["total_tco2"] / (monthly_totals["total_kwh"] / 1000)
    ).round(4)
    monthly_totals["renewable_pct"] = np.random.uniform(0.28, 0.42, len(monthly_totals)).round(4)

    return monthly_totals


# ─────────────────────────────────────────────
# RECOMMENDATIONS GENERATOR
# ─────────────────────────────────────────────
def generate_recommendations(org_id: str, org: dict) -> pd.DataFrame:
    templates = [
        {
            "type": "tariff_optimization",
            "title": "Switch to Time-of-Use Rate Schedule D",
            "description": "Load profile analysis shows 68% off-peak usage. Switching from flat rate reduces blended cost by $0.018/kWh.",
            "annual_savings_usd": 412000,
            "confidence": 0.94,
            "effort": "low",
            "payback_months": 1,
            "action": "Contact utility to change rate schedule. No capital required.",
        },
        {
            "type": "peak_shaving",
            "title": "Pre-cool HVAC during off-peak window (2–4 AM)",
            "description": "Demand charge of $12.80/kW accounts for 31% of bill. Shifting 480 kW reduces peak demand charge $6,144/month.",
            "annual_savings_usd": 73728,
            "confidence": 0.88,
            "effort": "medium",
            "payback_months": 8,
            "action": "Reprogram BMS setpoints. Requires BAS upgrade ~$18K.",
        },
        {
            "type": "procurement",
            "title": "Lock forward energy contracts — Q2 price spike forecast",
            "description": "LSTM model forecasts 14% price increase Q2. Lock 30% load at current $41.20/MWh vs projected $47/MWh.",
            "annual_savings_usd": 290000,
            "confidence": 0.81,
            "effort": "low",
            "payback_months": 1,
            "action": "Engage energy broker to execute 25 MW forward contract by April 5.",
        },
        {
            "type": "demand_response",
            "title": "Enroll in PJM Standard Demand Response Program",
            "description": "Grid operator DR events 4x/month. Curtailing 1,200 kW earns $14,400/event in incentive payments.",
            "annual_savings_usd": 172800,
            "confidence": 0.92,
            "effort": "medium",
            "payback_months": 3,
            "action": "Submit DR enrollment form to PJM by end of month.",
        },
        {
            "type": "load_shifting",
            "title": "Shift electric arc furnace to 11 PM – 3 AM slot",
            "description": "RL optimization agent identified $0.026/kWh differential. Carbon intensity also 18% lower in this window.",
            "annual_savings_usd": 504000,
            "confidence": 0.86,
            "effort": "high",
            "payback_months": 6,
            "action": "Requires production schedule coordination + BMS automation.",
        },
        {
            "type": "renewable",
            "title": "Add 8% renewable energy via PPA wind contract",
            "description": "Current renewable mix 38%. Adding 8% wind PPA at $28/MWh vs grid $41.20/MWh reduces cost and Scope 2 emissions.",
            "annual_savings_usd": 94500,
            "confidence": 0.90,
            "effort": "medium",
            "payback_months": 12,
            "action": "Issue RFP to wind developers. 15-year PPA at $28/MWh fixed.",
        },
    ]
    df = pd.DataFrame(templates)
    df["org_id"] = org_id
    df["status"] = "open"
    df["created_at"] = datetime.now().isoformat()
    return df


# ─────────────────────────────────────────────
# ALERTS GENERATOR
# ─────────────────────────────────────────────
def generate_alerts(org_id: str, interval_df: pd.DataFrame) -> pd.DataFrame:
    anomalies = interval_df[interval_df["is_anomaly"]]
    rows = []
    for _, row in anomalies.head(7).iterrows():
        severity = "critical" if row["kw"] > row["kw"] * 0.5 else "warning"
        rows.append({
            "org_id": org_id,
            "facility_id": row["facility_id"],
            "severity": severity,
            "type": "anomaly",
            "timestamp": row["timestamp"].isoformat(),
            "kw_reading": round(row["kw"], 2),
            "baseline_kw": round(row["kw"] / 3.5, 2),
            "anomaly_score": round(np.random.uniform(0.72, 0.98), 4),
            "estimated_cost_impact": round(row["cost_usd"] * 4, 2),
            "root_cause": random.choice([
                "HVAC control system fault",
                "Off-schedule equipment startup",
                "Demand ratchet breach",
                "Power factor degradation",
                "Unauthorized equipment",
            ]),
            "status": random.choice(["open", "open", "acknowledged"]),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# TARIFF SCHEDULE DATA
# ─────────────────────────────────────────────
def generate_tariff_schedules() -> pd.DataFrame:
    schedules = [
        {"id": "T001", "utility": "AEP Ohio", "name": "Industrial Demand (Schedule GS-3)",
         "type": "flat_demand", "base_energy_rate": 0.0674, "demand_charge_kw": 12.80,
         "off_peak_rate": None, "on_peak_rate": None, "on_peak_hours": "N/A"},
        {"id": "T002", "utility": "AEP Ohio", "name": "Time-of-Use Rate D",
         "type": "tou", "base_energy_rate": 0.0520, "demand_charge_kw": 10.40,
         "off_peak_rate": 0.0420, "on_peak_rate": 0.0980, "on_peak_hours": "9AM-9PM weekdays"},
        {"id": "T003", "utility": "ComEd", "name": "Commercial Flat Rate",
         "type": "flat", "base_energy_rate": 0.0910, "demand_charge_kw": 8.20,
         "off_peak_rate": None, "on_peak_rate": None, "on_peak_hours": "N/A"},
        {"id": "T004", "utility": "PPL Electric", "name": "Large Power Service",
         "type": "large_power", "base_energy_rate": 0.0580, "demand_charge_kw": 14.60,
         "off_peak_rate": 0.0480, "on_peak_rate": 0.1120, "on_peak_hours": "8AM-8PM weekdays"},
        {"id": "T005", "utility": "DTE Energy", "name": "Industrial TOU (Schedule D3)",
         "type": "tou_industrial", "base_energy_rate": 0.0750, "demand_charge_kw": 11.20,
         "off_peak_rate": 0.0610, "on_peak_rate": 0.1080, "on_peak_hours": "7AM-7PM weekdays"},
    ]
    return pd.DataFrame(schedules)


# ─────────────────────────────────────────────
# MARKET PRICE DATA (hourly, 30 days)
# ─────────────────────────────────────────────
def generate_market_prices() -> pd.DataFrame:
    dates = pd.date_range(start="2026-02-26", end="2026-03-28", freq="h")
    n = len(dates)
    base = 42.0
    trend = np.linspace(0, 4, n)  # slight upward trend
    hourly_shape = np.array([
        0.78, 0.74, 0.71, 0.70, 0.72, 0.80,
        0.95, 1.08, 1.18, 1.22, 1.20, 1.18,
        1.16, 1.14, 1.18, 1.20, 1.22, 1.24,
        1.20, 1.12, 1.02, 0.92, 0.84, 0.80,
    ])
    shape = np.tile(hourly_shape, n // 24 + 1)[:n]
    noise = np.random.normal(0, 2.4, n)
    prices = base * shape + trend + noise

    df = pd.DataFrame({
        "timestamp": dates,
        "pjm_day_ahead": np.round(prices, 2),
        "ercot_real_time": np.round(prices * 0.88 + np.random.normal(0, 1.8, n), 2),
        "nyiso_zone_a": np.round(prices * 1.12 + np.random.normal(0, 2.1, n), 2),
        "natural_gas_mmbtu": np.round(2.84 + np.random.normal(0, 0.08, n), 3),
        "carbon_credit_mt": np.round(31.20 + np.random.normal(0, 0.6, n), 2),
        "grid_carbon_intensity_tco2mwh": np.round(
            0.38 * shape + np.random.normal(0, 0.02, n), 4
        ),
    })
    df["pjm_day_ahead"] = df["pjm_day_ahead"].clip(lower=18)
    return df


# ─────────────────────────────────────────────
# MAIN — Run all generators
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("NexusGrid Synthetic Data Generator")
    print("© 2026 Mandeep Sharma. All rights reserved.")
    print("=" * 60)

    END_DATE = datetime(2026, 3, 28)
    START_DATE = END_DATE - timedelta(days=365)

    # Market prices
    print("\n[1/6] Generating market price data...")
    mkt = generate_market_prices()
    mkt.to_csv(f"{OUTPUT_DIR}/market_prices.csv", index=False)
    print(f"      → {len(mkt):,} rows saved to market_prices.csv")

    # Tariff schedules
    print("[2/6] Generating tariff schedules...")
    tariffs = generate_tariff_schedules()
    tariffs.to_csv(f"{OUTPUT_DIR}/tariff_schedules.csv", index=False)
    print(f"      → {len(tariffs)} tariff schedules saved")

    all_intervals = []
    all_bills = []
    all_emissions = []
    all_forecasts = []
    all_recommendations = []
    all_alerts = []

    # Per-org data
    print("[3/6] Generating per-organization data...")
    for org_id, org in ORGANIZATIONS.items():
        facilities = FACILITIES.get(org_id, [
            {"id": f"{org_id[:3].upper()}01", "name": f"Facility 1", "city": "Chicago",
             "state": "IL", "sqft": 120000, "base_frac": 0.5},
            {"id": f"{org_id[:3].upper()}02", "name": f"Facility 2", "city": "Dallas",
             "state": "TX", "sqft": 80000,  "base_frac": 0.5},
        ])

        org_intervals = []
        for fac in facilities:
            print(f"  {org['name']} — {fac['name']}...")
            df = generate_interval_data(org_id, fac, org, START_DATE, END_DATE)
            org_intervals.append(df)
            all_intervals.append(df)

        combined = pd.concat(org_intervals)
        bills = generate_monthly_bills(org_id, combined, org)
        bills["org_id"] = org_id
        all_bills.append(bills)

        emissions = generate_emissions(bills)
        emissions["org_id"] = org_id
        all_emissions.append(emissions)

        forecasts = generate_forecasts(bills)
        forecasts["org_id"] = org_id
        all_forecasts.append(forecasts)

        recs = generate_recommendations(org_id, org)
        all_recommendations.append(recs)

        alerts = generate_alerts(org_id, combined)
        all_alerts.append(alerts)

    # Save all CSVs
    print("\n[4/6] Saving all CSV files...")

    pd.concat(all_intervals).to_csv(f"{OUTPUT_DIR}/interval_data.csv", index=False)
    print(f"      → interval_data.csv  ({len(pd.concat(all_intervals)):,} rows)")

    pd.concat(all_bills).to_csv(f"{OUTPUT_DIR}/utility_bills.csv", index=False)
    print(f"      → utility_bills.csv  ({len(pd.concat(all_bills)):,} rows)")

    pd.concat(all_emissions).to_csv(f"{OUTPUT_DIR}/emissions.csv", index=False)
    print(f"      → emissions.csv      ({len(pd.concat(all_emissions)):,} rows)")

    pd.concat(all_forecasts).to_csv(f"{OUTPUT_DIR}/forecasts.csv", index=False)
    print(f"      → forecasts.csv      ({len(pd.concat(all_forecasts)):,} rows)")

    pd.concat(all_recommendations).to_csv(f"{OUTPUT_DIR}/recommendations.csv", index=False)
    print(f"      → recommendations.csv ({len(pd.concat(all_recommendations)):,} rows)")

    pd.concat(all_alerts).to_csv(f"{OUTPUT_DIR}/alerts.csv", index=False)
    print(f"      → alerts.csv         ({len(pd.concat(all_alerts)):,} rows)")

    # Save org manifest
    print("[5/6] Saving organization manifest...")
    with open(f"{OUTPUT_DIR}/organizations.json", "w") as f:
        json.dump(ORGANIZATIONS, f, indent=2)
    print("      → organizations.json")

    with open(f"{OUTPUT_DIR}/facilities.json", "w") as f:
        json.dump(FACILITIES, f, indent=2)
    print("      → facilities.json")

    print("\n[6/6] Done! All files written to:", OUTPUT_DIR)
    print("=" * 60)
    print("Summary:")
    print(f"  Organizations:   {len(ORGANIZATIONS)}")
    print(f"  Interval rows:   {len(pd.concat(all_intervals)):,}")
    print(f"  Bill records:    {len(pd.concat(all_bills)):,}")
    print(f"  Emission rows:   {len(pd.concat(all_emissions)):,}")
    print(f"  Recommendations: {len(pd.concat(all_recommendations)):,}")
    print(f"  Alerts:          {len(pd.concat(all_alerts)):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
