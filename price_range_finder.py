#!/usr/bin/env python3
import json
import pandas as pd
import os
import sys
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

def get_date_range(start_date_str, end_date_str):
    """Generate a list of dates between start_date and end_date (inclusive)"""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    delta = (end_date - start_date).days + 1
    
    if delta <= 0:
        print("Error: End date must be after start date")
        sys.exit(1)
    
    date_list = []
    for i in range(delta):
        date_list.append(start_date + timedelta(days=i))
        
    return date_list

def create_sample_data(date_range):
    """Create sample pricing data for St. Regis resorts for the given date range"""
    # Base data structure with resorts and room types
    resorts = [
        {"name": "St. Regis Bora Bora", "room_type": "Overwater Villa", "base_price": 1200, "currency": "USD"},
        {"name": "St. Regis Bora Bora", "room_type": "Beach Villa", "base_price": 900, "currency": "USD"},
        {"name": "St. Regis Maldives", "room_type": "Overwater Villa", "base_price": 1100, "currency": "EUR"},
        {"name": "St. Regis Maldives", "room_type": "Garden Villa", "base_price": 850, "currency": "EUR"},
        {"name": "St. Regis Bahia Beach", "room_type": "Ocean View", "base_price": 750, "currency": "USD"},
        {"name": "St. Regis Mauritius", "room_type": "Junior Suite", "base_price": 700, "currency": "GBP"},
        {"name": "St. Regis New York", "room_type": "Deluxe Suite", "base_price": 1350, "currency": "USD"},
        {"name": "St. Regis Aspen", "room_type": "Mountain View", "base_price": 1050, "currency": "USD"},
        {"name": "St. Regis Bangkok", "room_type": "Executive Suite", "base_price": 12500, "currency": "THB"},
        {"name": "St. Regis Rome", "room_type": "Superior Room", "base_price": 950, "currency": "EUR"},
        {"name": "St. Regis Abu Dhabi", "room_type": "Sea View", "base_price": 1000, "currency": "USD"},
        {"name": "St. Regis Punta Mita", "room_type": "Garden View", "base_price": 850, "currency": "USD"}
    ]
    
    # Create a dictionary to store prices by resort and date
    resort_price_data = {}
    
    for resort in resorts:
        resort_key = f"{resort['name']} - {resort['room_type']}"
        resort_price_data[resort_key] = {}
        
        # Convert base price to USD
        base_price_usd = convert_to_usd(resort['base_price'], resort['currency'])
        
        # Generate prices for each day with some variations based on the day of week
        for date in date_range:
            date_key = f"{date.strftime('%Y-%m-%d')}"
            
            # Add price variations:
            # - Weekends (Fri-Sun) are more expensive
            # - Middle of the week has some discounts
            # - Some random variation to simulate demand
            weekday = date.weekday()
            
            if weekday >= 4:  # Friday, Saturday, Sunday (higher prices)
                daily_price = round(base_price_usd * (1.10 + (weekday - 4) * 0.05), 2)
            elif weekday == 2:  # Wednesday (slight discount)
                daily_price = round(base_price_usd * 0.95, 2)
            else:
                daily_price = base_price_usd
                
            # Add a seasonal modifier based on the month
            month = date.month
            if month in [6, 7, 8]:  # Summer season
                seasonal_modifier = 1.15  # 15% more expensive in summer
            elif month in [11, 12]:  # Holiday season
                seasonal_modifier = 1.20  # 20% more expensive during holidays
            else:
                seasonal_modifier = 1.0
                
            daily_price = round(daily_price * seasonal_modifier, 2)
                
            resort_price_data[resort_key][date_key] = daily_price
            
        # Calculate average price
        prices = list(resort_price_data[resort_key].values())
        avg_price = sum(prices) / len(prices) if prices else 0
        resort_price_data[resort_key]["Average Price"] = round(avg_price, 2)
    
    return resort_price_data

def format_as_usd(df):
    """Format all numeric values as USD currency"""
    formatted_df = df.copy()
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
    return formatted_df

def main():
    # Set pandas display options for better table rendering
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 2)
    pd.set_option('display.max_rows', None)  # Show all rows
    
    # Default date range (June 17-21, 2025)
    default_start_date = "2025-06-17"
    default_end_date = "2025-06-21"
    
    # Get date range from command line arguments or use defaults
    if len(sys.argv) > 2:
        start_date_str = sys.argv[1]
        end_date_str = sys.argv[2]
    else:
        print(f"Using default date range: {default_start_date} to {default_end_date}")
        start_date_str = default_start_date
        end_date_str = default_end_date
    
    # Generate date range
    date_range = get_date_range(start_date_str, end_date_str)
    
    # File path for the JSON data
    json_file_path = 'st-regis.json'
    
    # Check if file exists and is not empty
    if not os.path.exists(json_file_path) or os.path.getsize(json_file_path) == 0:
        print(f"Creating sample data since {json_file_path} is empty or does not exist.")
        # Use sample data for the specified date range
        resort_price_data = create_sample_data(date_range)
    else:
        # TODO: Process actual JSON data file if it exists
        # For now, just use sample data
        print(f"Using sample data for demonstration.")
        resort_price_data = create_sample_data(date_range)
    
    # Convert the nested dictionary to a DataFrame
    df = pd.DataFrame.from_dict(resort_price_data, orient='index')
    
    # Move the Average Price column to the end
    cols = list(df.columns)
    if "Average Price" in cols:
        cols.remove("Average Price")
        cols.append("Average Price")
        df = df[cols]
    
    # Create the formatted table with USD currency symbols
    formatted_df = format_as_usd(df)
    
    # Display the pricing table
    print(f"\n===== St. Regis Resort Pricing (USD) - {start_date_str} to {end_date_str} =====\n")
    print(formatted_df)
    
    # Create a sorted version by average price
    if "Average Price" in df.columns:
        sorted_df = df.sort_values(by="Average Price")
        formatted_sorted = format_as_usd(sorted_df[["Average Price"]])
        
        # Display the sorted table
        print("\n===== Resorts Sorted by Average Price (USD) =====\n")
        print(formatted_sorted)
    
    # Calculate and display price statistics
    print("\n===== Price Statistics =====\n")
    avg_prices = {}
    for resort in df.index:
        # Skip the Average Price column
        price_cols = [col for col in df.columns if col != "Average Price"]
        avg_prices[resort] = df.loc[resort, price_cols].mean()
    
    stats_df = pd.DataFrame({
        "Lowest Price": df.min(axis=1),
        "Highest Price": df.max(axis=1),
        "Average Price": avg_prices
    })
    
    print(format_as_usd(stats_df))
    
    # Display overall recommendations
    print("\n===== Price Recommendations =====\n")
    best_value = stats_df.sort_values(by="Average Price").index[0]
    most_luxurious = stats_df.sort_values(by="Average Price", ascending=False).index[0]
    
    print(f"Best Value Option: {best_value}")
    print(f"Most Luxurious Option: {most_luxurious}")

if __name__ == "__main__":
    main()
