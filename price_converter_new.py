#!/usr/bin/env python3
import json
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Currency conversion rates to USD (as of May 2025)
CONVERSION_RATES = {
    'USD': 1.0,        # US Dollar (base currency)
    'EUR': 1.08,       # Euro
    'GBP': 1.26,       # British Pound
    'JPY': 0.0067,     # Japanese Yen
    'CAD': 0.74,       # Canadian Dollar
    'AUD': 0.67,       # Australian Dollar
    'CHF': 1.12,       # Swiss Franc
    'CNY': 0.14,       # Chinese Yuan
    'MXN': 0.052,      # Mexican Peso
    'THB': 0.028,      # Thai Baht
}

def convert_to_usd(price: float, currency: str) -> float:
    """Convert price from given currency to USD"""
    if currency not in CONVERSION_RATES:
        print(f"Warning: Currency {currency} not found in conversion rates. Using 1:1 ratio.")
        return price
    
    return price * CONVERSION_RATES[currency]

def extract_price_value(price_str: str) -> tuple:
    """Extract numeric value and currency from price string"""
    currency_symbols = {
        '$': 'USD',
        '€': 'EUR',
        '£': 'GBP',
        '¥': 'JPY',
        'C$': 'CAD',
        'A$': 'AUD',
        'CHF': 'CHF',
        '฿': 'THB'
    }
    
    price_str = price_str.strip()
    
    # Check for currency symbols at the beginning
    for symbol, code in currency_symbols.items():
        if price_str.startswith(symbol):
            numeric_str = price_str[len(symbol):].strip().replace(',', '')
            return float(numeric_str), code
    
    # If no symbol found, assume it's formatted as "123.45 USD"
    parts = price_str.split()
    if len(parts) >= 2:
        try:
            price_value = float(parts[0].replace(',', ''))
            currency_code = parts[1].upper()
            return price_value, currency_code
        except ValueError:
            pass
    
    # Default fallback - assume USD and try to extract just the number
    try:
        price_value = float(price_str.replace(',', '').replace('$', ''))
        return price_value, 'USD'
    except ValueError:
        print(f"Warning: Could not parse price string: {price_str}. Setting to 0 USD.")
        return 0.0, 'USD'

def create_sample_data():
    """Create sample pricing data for St. Regis resorts"""
    # Base data structure with resorts and room types
    resorts = [
        {"name": "St. Regis Bora Bora", "room_type": "Overwater Villa", "base_price": 1200, "currency": "USD"},
        {"name": "St. Regis Bora Bora", "room_type": "Beach Villa", "base_price": 900, "currency": "USD"},
        {"name": "St. Regis Maldives", "room_type": "Overwater Villa", "base_price": 1100, "currency": "EUR"},
        {"name": "St. Regis Maldives", "room_type": "Garden Villa", "base_price": 850, "currency": "EUR"},
        {"name": "St. Regis Bahia Beach", "room_type": "Ocean View", "base_price": 750, "currency": "USD"},
        {"name": "St. Regis Mauritius", "room_type": "Junior Suite", "base_price": 700, "currency": "GBP"}
    ]
    
    # Create a dictionary to store prices by resort and date
    resort_price_data = {}
    
    # Generate price data for 7 days starting from June 1
    start_date = datetime(2025, 6, 1)
    
    for resort in resorts:
        resort_key = f"{resort['name']} - {resort['room_type']}"
        resort_price_data[resort_key] = {}
        
        # Convert base price to USD
        base_price_usd = convert_to_usd(resort['base_price'], resort['currency'])
        
        # Generate prices for each day with slight variations
        for day in range(7):
            current_date = start_date + timedelta(days=day)
            date_key = f"Day {day+1} ({current_date.strftime('%Y-%m-%d')})"
            
            # Add some variation based on the day
            if day % 2 == 0:  # Higher prices on even days
                daily_price = round(base_price_usd * 1.03, 2)
            elif day % 3 == 0:  # Lower prices on days divisible by 3
                daily_price = round(base_price_usd * 0.97, 2)
            else:
                daily_price = base_price_usd
                
            resort_price_data[resort_key][date_key] = daily_price
            
        # Calculate average price
        avg_price = sum(resort_price_data[resort_key].values()) / 7
        resort_price_data[resort_key]["Average Price"] = round(avg_price, 2)
    
    return resort_price_data

def format_as_usd(df):
    """Format all numeric values as USD currency"""
    formatted_df = df.copy()
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
    return formatted_df

def main():
    # Set pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 2)
    
    # File path for the JSON data
    json_file_path = 'st-regis.json'
    
    # Check if file exists and is not empty
    if not os.path.exists(json_file_path) or os.path.getsize(json_file_path) == 0:
        print(f"Creating sample data since {json_file_path} is empty or does not exist.")
        # Use sample data
        resort_price_data = create_sample_data()
    else:
        # TODO: Process actual JSON data file if it exists
        # For now, just use sample data
        print(f"Using sample data for demonstration.")
        resort_price_data = create_sample_data()
    
    # Convert the nested dictionary to a DataFrame
    df = pd.DataFrame.from_dict(resort_price_data, orient='index')
    
    # Move the Average Price column to the end
    cols = list(df.columns)
    avg_col = cols.pop(-1)  # Remove Average Price
    df = df[cols + [avg_col]]  # Add it back at the end
    
    # Create the formatted table with USD currency symbols
    formatted_df = format_as_usd(df)
    
    # Display the pricing table
    print("\n===== St. Regis Resort Pricing (USD) - Starting from June 1, 2025 =====\n")
    print(formatted_df)
    
    # Create a sorted version by average price
    avg_price_col = df.columns[-1]
    sorted_df = df.sort_values(by=avg_price_col)
    formatted_sorted = format_as_usd(sorted_df[[avg_price_col]])
    
    # Display the sorted table
    print("\n===== Resorts Sorted by Average Price (USD) =====\n")
    print(formatted_sorted)

if __name__ == "__main__":
    main()
