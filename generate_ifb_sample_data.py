"""
Generate Sample IFB Service Ecosystem Data

This script generates realistic sample data for IFB's service ecosystem
including service calls, spare parts, warranty claims, and franchise data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_RECORDS = 5000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Master data
LOCATIONS = [
    'Mumbai_Central', 'Mumbai_Andheri', 'Delhi_South', 'Delhi_North', 'Bangalore_Koramangala',
    'Bangalore_Whitefield', 'Chennai_Anna_Nagar', 'Chennai_Velachery', 'Kolkata_Salt_Lake',
    'Kolkata_Park_Street', 'Pune_Kothrud', 'Pune_Hinjewadi', 'Hyderabad_Gachibowli',
    'Hyderabad_Secunderabad', 'Ahmedabad_Satellite', 'Ahmedabad_Maninagar',
    'Jaipur_Malviya_Nagar', 'Lucknow_Gomti_Nagar', 'Surat_Adajan', 'Indore_Vijay_Nagar'
]

REGIONS = {
    'Mumbai_Central': 'West', 'Mumbai_Andheri': 'West',
    'Delhi_South': 'North', 'Delhi_North': 'North',
    'Bangalore_Koramangala': 'South', 'Bangalore_Whitefield': 'South',
    'Chennai_Anna_Nagar': 'South', 'Chennai_Velachery': 'South',
    'Kolkata_Salt_Lake': 'East', 'Kolkata_Park_Street': 'East',
    'Pune_Kothrud': 'West', 'Pune_Hinjewadi': 'West',
    'Hyderabad_Gachibowli': 'South', 'Hyderabad_Secunderabad': 'South',
    'Ahmedabad_Satellite': 'West', 'Ahmedabad_Maninagar': 'West',
    'Jaipur_Malviya_Nagar': 'North', 'Lucknow_Gomti_Nagar': 'North',
    'Surat_Adajan': 'West', 'Indore_Vijay_Nagar': 'Central'
}

FRANCHISE_MAPPING = {
    'Mumbai_Central': 'FR001_Mumbai_West',
    'Mumbai_Andheri': 'FR001_Mumbai_West',
    'Delhi_South': 'FR002_Delhi_NCR',
    'Delhi_North': 'FR002_Delhi_NCR',
    'Bangalore_Koramangala': 'FR003_Bangalore',
    'Bangalore_Whitefield': 'FR003_Bangalore',
    'Chennai_Anna_Nagar': 'FR004_Chennai',
    'Chennai_Velachery': 'FR004_Chennai',
    'Kolkata_Salt_Lake': 'FR005_Kolkata',
    'Kolkata_Park_Street': 'FR005_Kolkata',
    'Pune_Kothrud': 'FR006_Pune',
    'Pune_Hinjewadi': 'FR006_Pune',
    'Hyderabad_Gachibowli': 'FR007_Hyderabad',
    'Hyderabad_Secunderabad': 'FR007_Hyderabad',
    'Ahmedabad_Satellite': 'FR008_Gujarat',
    'Ahmedabad_Maninagar': 'FR008_Gujarat',
    'Jaipur_Malviya_Nagar': 'FR009_Rajasthan',
    'Lucknow_Gomti_Nagar': 'FR010_UP',
    'Surat_Adajan': 'FR008_Gujarat',
    'Indore_Vijay_Nagar': 'FR011_MP'
}

PRODUCT_CATEGORIES = [
    'Washing Machine', 'Refrigerator', 'Microwave Oven',
    'Dishwasher', 'Air Conditioner', 'Kitchen Appliances'
]

SERVICE_TYPES = [
    'Installation', 'Repair', 'Maintenance', 'Annual Maintenance Contract',
    'Diagnostic', 'Part Replacement', 'Cleaning'
]

PARTS = {
    'Washing Machine': ['Motor', 'Drum', 'Door Lock', 'Pump', 'Belt', 'Control Board', 'Inlet Valve'],
    'Refrigerator': ['Compressor', 'Thermostat', 'Door Seal', 'Evaporator Fan', 'Defrost Timer', 'Ice Maker'],
    'Microwave Oven': ['Magnetron', 'Diode', 'Capacitor', 'Turntable Motor', 'Door Switch'],
    'Dishwasher': ['Spray Arm', 'Pump', 'Heating Element', 'Door Latch', 'Filter'],
    'Air Conditioner': ['Compressor', 'Fan Motor', 'Thermostat', 'Capacitor', 'Filter', 'PCB'],
    'Kitchen Appliances': ['Heating Element', 'Motor', 'Timer', 'Gasket', 'Switch']
}

PART_COSTS = {
    'Motor': (1500, 3500), 'Drum': (2500, 5000), 'Door Lock': (300, 800), 'Pump': (800, 1500),
    'Belt': (200, 400), 'Control Board': (2000, 4000), 'Inlet Valve': (400, 800),
    'Compressor': (5000, 12000), 'Thermostat': (500, 1200), 'Door Seal': (300, 700),
    'Evaporator Fan': (800, 1500), 'Defrost Timer': (400, 900), 'Ice Maker': (1500, 3000),
    'Magnetron': (2000, 4000), 'Diode': (200, 500), 'Capacitor': (300, 700),
    'Turntable Motor': (400, 800), 'Door Switch': (150, 350), 'Spray Arm': (300, 600),
    'Heating Element': (600, 1200), 'Door Latch': (250, 550), 'Filter': (200, 500),
    'Fan Motor': (1200, 2500), 'PCB': (2500, 5000), 'Gasket': (200, 400),
    'Timer': (400, 800), 'Switch': (150, 350)
}

def generate_service_data():
    """Generate sample service ecosystem data"""

    data = []

    for i in range(NUM_RECORDS):
        # Basic service info
        service_id = f"SVC{str(i+10000).zfill(6)}"
        service_date = START_DATE + timedelta(days=random.randint(0, (END_DATE - START_DATE).days))

        # Location and franchise
        location = random.choice(LOCATIONS)
        region = REGIONS[location]
        franchise_id = FRANCHISE_MAPPING[location]
        branch = location

        # Product and service type
        product_category = random.choice(PRODUCT_CATEGORIES)
        service_type = random.choice(SERVICE_TYPES)

        # Technician (5-10 per location)
        tech_num = random.randint(1, 10)
        technician_id = f"TECH_{location[:3].upper()}{tech_num:03d}"

        # Customer
        customer_id = f"CUST{random.randint(100000, 999999)}"

        # Service duration (in hours)
        if service_type == 'Installation':
            duration = round(random.uniform(1.5, 4.0), 1)
        elif service_type == 'Repair':
            duration = round(random.uniform(1.0, 6.0), 1)
        elif service_type in ['Maintenance', 'Cleaning']:
            duration = round(random.uniform(0.5, 2.0), 1)
        else:
            duration = round(random.uniform(1.0, 3.0), 1)

        # Parts usage
        parts_needed = random.random() < 0.6  # 60% of services need parts

        if parts_needed and product_category in PARTS:
            available_parts = PARTS[product_category]
            num_parts = random.randint(1, min(3, len(available_parts)))
            parts_used_list = random.sample(available_parts, num_parts)
            part_name = parts_used_list[0]  # Main part
            parts_used = num_parts

            # Calculate parts cost
            parts_cost = sum([random.uniform(*PART_COSTS.get(part, (500, 1500))) for part in parts_used_list])
            parts_cost = round(parts_cost, 2)

            # Parts revenue (cost + 40-60% margin)
            parts_margin = random.uniform(1.4, 1.6)
            parts_revenue = round(parts_cost * parts_margin, 2)
        else:
            part_name = 'None'
            parts_used = 0
            parts_cost = 0
            parts_revenue = 0

        # Service cost and revenue
        base_service_cost = duration * random.uniform(200, 400)
        service_cost = round(base_service_cost, 2)

        # Service revenue (cost + margin)
        service_margin = random.uniform(1.3, 1.7)
        service_revenue = round(service_cost * service_margin, 2)

        # Warranty claim (15% of services)
        warranty_claim = 1 if random.random() < 0.15 else 0

        # If warranty claim, reduce revenue
        if warranty_claim:
            service_revenue = service_revenue * 0.3  # IFB absorbs 70% cost
            parts_revenue = parts_revenue * 0.4  # IFB absorbs 60% parts cost

        # Customer satisfaction (3.0 to 5.0)
        satisfaction = round(random.uniform(3.0, 5.0), 1)

        # Product age (affects warranty likelihood)
        product_age_months = random.randint(1, 120)  # 0-10 years

        # First call resolution
        fcr = 1 if random.random() < 0.75 else 0

        # Priority (1=Low, 2=Medium, 3=High, 4=Critical)
        priority = random.choices([1, 2, 3, 4], weights=[0.4, 0.35, 0.20, 0.05])[0]

        record = {
            'Service_ID': service_id,
            'Service_Date': service_date.strftime('%Y-%m-%d'),
            'Location': location,
            'Branch': branch,
            'Region': region,
            'Franchise_ID': franchise_id,
            'Product_Category': product_category,
            'Service_Type': service_type,
            'Technician_ID': technician_id,
            'Customer_ID': customer_id,
            'Service_Duration': duration,
            'Part_Name': part_name,
            'Parts_Used': parts_used,
            'Parts_Cost': parts_cost,
            'Parts_Revenue': parts_revenue,
            'Service_Cost': service_cost,
            'Service_Revenue': service_revenue,
            'Total_Revenue': service_revenue + parts_revenue,
            'Warranty_Claim': warranty_claim,
            'Customer_Satisfaction': satisfaction,
            'Product_Age_Months': product_age_months,
            'First_Call_Resolution': fcr,
            'Priority': priority
        }

        data.append(record)

    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating IFB Service Ecosystem Sample Data...")
    df = generate_service_data()

    # Save to CSV
    output_file = "data/ifb_service_data.csv"
    df.to_csv(output_file, index=False)

    print(f"✅ Generated {len(df)} records")
    print(f"📁 Saved to: {output_file}")
    print(f"\nDataset Summary:")
    print(f"- Date Range: {df['Service_Date'].min()} to {df['Service_Date'].max()}")
    print(f"- Locations: {df['Location'].nunique()}")
    print(f"- Franchises: {df['Franchise_ID'].nunique()}")
    print(f"- Total Service Revenue: ₹{df['Service_Revenue'].sum():,.2f}")
    print(f"- Total Parts Revenue: ₹{df['Parts_Revenue'].sum():,.2f}")
    print(f"- Warranty Claims: {df['Warranty_Claim'].sum()} ({df['Warranty_Claim'].sum()/len(df)*100:.1f}%)")
    print(f"- Avg Customer Satisfaction: {df['Customer_Satisfaction'].mean():.2f}/5.0")
    print("\n📊 Sample records:")
    print(df.head())
