#!/usr/bin/env python3
"""
Sample Data Generator for Testing the Visualization Engine
Creates realistic datasets for testing different visualization scenarios
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_sales_data(n_rows=1000):
    """Generate realistic sales data"""
    np.random.seed(42)
    random.seed(42)
    
    # Define categories
    regions = ['North', 'South', 'East', 'West', 'Central']
    products = ['Laptop', 'Phone', 'Tablet', 'Watch', 'Headphones', 'Camera']
    sales_reps = [f'Rep_{i:02d}' for i in range(1, 21)]
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(n_rows):
        # Generate correlated data
        base_price = np.random.choice([500, 800, 1200, 300, 150, 900])
        quantity = np.random.poisson(3) + 1
        
        # Add some correlation between price and quantity (higher price = lower quantity)
        if base_price > 800:
            quantity = max(1, quantity - np.random.poisson(1))
        
        discount = np.random.beta(2, 5) * 0.3  # Most discounts are small
        final_price = base_price * (1 - discount)
        revenue = final_price * quantity
        
        # Add seasonal effects
        date = start_date + timedelta(days=i // 3)
        month = date.month
        if month in [11, 12]:  # Holiday season
            revenue *= 1.2
        elif month in [1, 2]:  # Post-holiday slump
            revenue *= 0.8
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'region': np.random.choice(regions),
            'product': np.random.choice(products),
            'sales_rep': np.random.choice(sales_reps),
            'base_price': base_price,
            'discount_rate': discount,
            'final_price': final_price,
            'quantity': quantity,
            'revenue': revenue,
            'customer_satisfaction': np.random.normal(4.2, 0.8),  # Scale 1-5
            'processing_time_days': np.random.exponential(2) + 1
        })
    
    df = pd.DataFrame(data)
    
    # Add some missing values to make it realistic
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'customer_satisfaction'] = np.nan
    
    return df

def generate_employee_data(n_rows=500):
    """Generate employee performance data"""
    np.random.seed(123)
    
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
    positions = ['Junior', 'Senior', 'Lead', 'Manager', 'Director']
    
    data = []
    for i in range(n_rows):
        dept = np.random.choice(departments)
        position = np.random.choice(positions)
        
        # Correlate salary with position and department
        base_salary = {
            'Junior': 50000, 'Senior': 70000, 'Lead': 90000, 
            'Manager': 110000, 'Director': 150000
        }[position]
        
        dept_multiplier = {
            'Engineering': 1.2, 'Sales': 1.1, 'Marketing': 1.0,
            'HR': 0.9, 'Finance': 1.1, 'Operations': 0.95
        }[dept]
        
        salary = base_salary * dept_multiplier * np.random.normal(1, 0.15)
        
        # Performance correlates with salary but has variation
        performance = min(5.0, max(1.0, (salary / 100000) * 3 + np.random.normal(0, 0.5)))
        
        data.append({
            'employee_id': f'EMP_{i:04d}',
            'department': dept,
            'position': position,
            'years_experience': np.random.poisson(5) + 1,
            'salary': salary,
            'performance_rating': performance,
            'training_hours': np.random.poisson(40),
            'projects_completed': np.random.poisson(8),
            'overtime_hours': np.random.exponential(10),
            'satisfaction_score': np.random.normal(3.5, 1.0)
        })
    
    df = pd.DataFrame(data)
    df['satisfaction_score'] = df['satisfaction_score'].clip(1, 5)
    
    return df

def generate_website_analytics(n_rows=2000):
    """Generate website analytics data"""
    np.random.seed(456)
    
    traffic_sources = ['Organic', 'Paid Search', 'Social', 'Direct', 'Email', 'Referral']
    devices = ['Desktop', 'Mobile', 'Tablet']
    countries = ['USA', 'UK', 'Canada', 'Germany', 'France', 'Australia', 'Japan']
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(n_rows):
        date = start_date + timedelta(days=i // 10)
        source = np.random.choice(traffic_sources)
        device = np.random.choice(devices)
        
        # Mobile users have different behavior
        if device == 'Mobile':
            sessions = np.random.poisson(3) + 1
            bounce_rate = np.random.beta(3, 2) * 0.8  # Higher bounce rate
            page_views = sessions * np.random.poisson(2)
        else:
            sessions = np.random.poisson(5) + 1
            bounce_rate = np.random.beta(2, 3) * 0.6
            page_views = sessions * np.random.poisson(4)
        
        # Conversion rate varies by source
        conversion_multiplier = {
            'Organic': 1.0, 'Paid Search': 1.3, 'Social': 0.7,
            'Direct': 1.5, 'Email': 2.0, 'Referral': 0.9
        }[source]
        
        conversion_rate = np.random.beta(1, 20) * conversion_multiplier
        conversions = np.random.binomial(sessions, conversion_rate)
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'traffic_source': source,
            'device_type': device,
            'country': np.random.choice(countries),
            'sessions': sessions,
            'page_views': page_views,
            'bounce_rate': bounce_rate,
            'avg_session_duration': np.random.exponential(120) + 30,  # seconds
            'conversions': conversions,
            'conversion_rate': conversion_rate,
            'revenue': conversions * np.random.exponential(50) + 20
        })
    
    return pd.DataFrame(data)

def main():
    """Generate sample datasets for testing"""
    print("🎲 Generating sample datasets for testing...")
    
    # Create sample data directory
    os.makedirs('sample_data', exist_ok=True)
    
    # Generate different types of datasets
    datasets = {
        'sales_data.csv': generate_sales_data(1000),
        'employee_data.csv': generate_employee_data(500),
        'website_analytics.csv': generate_website_analytics(2000)
    }
    
    for filename, df in datasets.items():
        filepath = os.path.join('sample_data', filename)
        df.to_csv(filepath, index=False)
        print(f"✅ Generated {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Print first few rows and basic info
        print(f"   Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
        print(f"   Sample: {df.iloc[0].to_dict()}")
        print()
    
    print("🎉 Sample datasets generated successfully!")
    print("📁 Files saved to: sample_data/")
    print("\n💡 Test the visualization engine with:")
    print("   python viz_engine.py sample_data/sales_data.csv")
    print("   python viz_engine.py sample_data/employee_data.csv")
    print("   python viz_engine.py sample_data/website_analytics.csv")

if __name__ == "__main__":
    main()