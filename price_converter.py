#!/usr/bin/env python3
import json
import pandas as pd
import os
import numpy as np
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
    # Add more currencies as needed
}

# Number of days to display in the pricing table
NUM_DAYS = 7

def convert_to_usd(price: float, currency: str) -> float:
    """Convert price from given currency to USD"""
    if currency not in CONVERSION_RATES:
        print(f"Warning: Currency {currency} not found in conversion rates. Using 1:1 ratio.")
        return price
    
    return price * CONVERSION_RATES[currency]

def extract_price_value(price_str: str) -> tuple:
    """Extract numeric value and currency from price string"""
    # Common currency symbols and their corresponding currency codes
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
            # Remove currency symbol and any whitespace, then convert to float
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

def generate_sample_data() -> List[Dict[str, Any]]:
    """Generate sample pricing data for multiple days starting from June 1, 2025"""
    # Base resorts and room types with their initial prices
    base_data = [
        {"resort_name": "St. Regis Bora Bora", "room_type": "Overwater Villa", "price": "$1,200"},
        {"resort_name": "St. Regis Bora Bora", "room_type": "Beach Villa", "price": "$900"},
        {"resort_name": "St. Regis Maldives", "room_type": "Overwater Villa", "price": "€1,100"},
        {"resort_name": "St. Regis Maldives", "room_type": "Garden Villa", "price": "€850"},
        {"resort_name": "St. Regis Bahia Beach", "room_type": "Ocean View", "price": "$750"},
        {"resort_name": "St. Regis Mauritius", "room_type": "Junior Suite", "price": "£700"}
    ]
    
    # Extended data with prices for each day
    extended_data = []
    
    # Start date (June 1, 2025)
    start_date = datetime(2025, 6, 1)
    
    # Generate price data for each resort, room type, and day
    for base_item in base_data:
        price_value, currency = extract_price_value(base_item["price"])
        
        for day in range(NUM_DAYS):
            # Calculate date for this item
            current_date = start_date + timedelta(days=day)
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Add some price variation between days (±3%)
            if day % 2 == 0:  # Even days have higher prices
                daily_price = price_value * 1.03
            elif day % 3 == 0:  # Every third day has lower prices
                daily_price = price_value * 0.97
            else:  # Other days use the base price
                daily_price = price_value
            
            # Format the price string with the appropriate currency symbol
            if currency == 'USD':
                price_str = f"${daily_price:,.2f}"
            elif currency == 'EUR':
                price_str = f"€{daily_price:,.2f}"
            elif currency == 'GBP':
                price_str = f"£{daily_price:,.2f}"
            else:
                price_str = f"{daily_price:,.2f} {currency}"
            
            # Create the data item for this resort, room type, and day
            item = {
                "resort_name": base_item["resort_name"],
                "room_type": base_item["room_type"],
                "price": price_str,
                "date": date_str
            }
            
            extended_data.append(item)
    
    return extended_data

def process_pricing_data(json_file_path: str) -> pd.DataFrame:
    """Process pricing data from a JSON file (or generate sample data) and convert to USD"""
    # Check if file exists and is not empty
    if not os.path.exists(json_file_path) or os.path.getsize(json_file_path) == 0:
        print(f"Creating sample data since {json_file_path} is empty or does not exist.")
        # Generate sample data for demonstration
        data = generate_sample_data()
    else:
        # Load the data from JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    
    # Prepare data for DataFrame
    processed_data = []
    
    for item in data:
        # Skip items without a price
        if 'price' not in item or not item['price']:
            continue
        
        # Get the price and currency
        price_value, currency = extract_price_value(item['price'])
        
        # Override currency if explicitly specified in the data
        if 'currency' in item:
            currency = item['currency']
        
        # Convert to USD
        price_usd = convert_to_usd(price_value, currency)
        
        # Add to processed data
        processed_item = {
            'Resort': item.get('resort_name', 'Unknown'),
            'Room Type': item.get('room_type', 'Unknown'),
            'Date': item.get('date', 'Unknown'),
            'Original Price': f"{price_value:.2f} {currency}",
            'Price (USD)': round(price_usd, 2)
        }
        processed_data.append(processed_item)
    
    # Create DataFrame
    df = pd.DataFrame(processed_data)
    
    if df.empty:
        print("Warning: No valid pricing data found. Creating sample data.")
        # If we ended up with an empty DataFrame, use sample data
        data = generate_sample_data()
        return process_pricing_data(json_file_path)
    
    return df

def create_resort_date_table(pricing_df: pd.DataFrame) -> pd.DataFrame:
    """Create a table with resorts as rows and dates as columns"""
    # Create a combined resort and room type field for the index
    pricing_df['Resort_Room'] = pricing_df['Resort'] + ' - ' + pricing_df['Room Type']
    
    # Convert the date to a proper datetime
    pricing_df['Date'] = pd.to_datetime(pricing_df['Date'])
    
    # Create a pivot table with resorts as rows and dates as columns
    pivot_df = pricing_df.pivot_table(
        index='Resort_Room',
        columns='Date',
        values='Price (USD)',
        aggfunc='first'
    )
    
    # Format the date columns to a readable format
    pivot_df.columns = [f"Day {(col.date() - datetime(2025, 6, 1).date()).days + 1} ({col.strftime('%Y-%m-%d')})" 
                        for col in pivot_df.columns]
    
    # Calculate the average price for each resort
    pivot_df['Average Price'] = pivot_df.mean(axis=1)
    
    return pivot_df

def format_as_usd(df: pd.DataFrame) -> pd.DataFrame:
    """Format all numeric columns as USD"""
    # Format all numeric columns as USD
    formatted_df = df.copy()
    
    for col in formatted_df.columns:
        if col != 'Average Price':
            formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
        else:
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
    
    # Process the pricing data
    pricing_df = process_pricing_data(json_file_path)
    
    # Create the resort-date table
    resort_date_table = create_resort_date_table(pricing_df)
    
    # Format the table values as USD
    formatted_table = format_as_usd(resort_date_table)
    
    # Display the pricing table
    print("\n===== St. Regis Resort Pricing (USD) - Starting from June 1, 2025 =====\n")
    print(formatted_table)
    
    # Create a sorted version by average price
    avg_price_sorted = resort_date_table.sort_values(by='Average Price')
    formatted_sorted = format_as_usd(avg_price_sorted[['Average Price']])
    
    # Display the sorted table
    print("\n===== Resorts Sorted by Average Price (USD) =====\n")
    print(formatted_sorted)

if __name__ == "__main__":
    main()
