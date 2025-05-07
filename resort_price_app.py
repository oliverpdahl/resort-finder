#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import json

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
    'AED': 0.27,       # UAE Dirham
    'SAR': 0.27,       # Saudi Riyal
    'MAD': 0.10,       # Moroccan Dirham
    'MYR': 0.23,       # Malaysian Ringgit
    'IDR': 0.000065,   # Indonesian Rupiah
    'XPF': 0.0091      # CFP Franc (French Polynesia)
}

def convert_to_usd(price: float, currency: str) -> float:
    """Convert price from given currency to USD"""
    if currency not in CONVERSION_RATES:
        st.warning(f"Currency {currency} not found in conversion rates. Using 1:1 ratio.")
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
    
    price_str = str(price_str).strip()
    
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
        st.warning(f"Could not parse price string: {price_str}. Setting to 0 USD.")
        return 0.0, 'USD'

def get_date_range(start_date, end_date):
    """Generate a list of dates between start_date and end_date (inclusive)"""
    delta = (end_date - start_date).days + 1
    
    if delta <= 0:
        st.error("End date must be after start date")
        return []
    
    date_list = []
    for i in range(delta):
        date_list.append(start_date + timedelta(days=i))
        
    return date_list

def process_json_data(json_data, date_range):
    """Process the real JSON data from st-regis.json for the given date range"""
    # Create a dictionary to store processed data
    resort_price_data = {}
    
    # Process each resort in the JSON data
    for resort in json_data:
        resort_name = resort.get('name', 'Unknown Resort')
        location = resort.get('location', 'Unknown Location')
        currency = resort.get('currency', 'USD')
        pricing = resort.get('pricing', {})
        
        # Create a key for this resort
        resort_key = f"{resort_name} - {location}"
        resort_price_data[resort_key] = {}
        
        # Calculate the Peak Price Average (avg of 3 highest prices in the month)
        # Convert all prices for the entire month to USD
        all_month_prices = []
        for day in range(1, 32):  # Cover all possible days in a month
            day_str = str(day)
            if day_str in pricing and pricing[day_str] is not None:
                price_in_local = float(pricing[day_str])
                price_in_usd = convert_to_usd(price_in_local, currency)
                all_month_prices.append(price_in_usd)
        
        # Get the 3 highest prices and calculate their average
        if len(all_month_prices) >= 3:
            top_3_prices = sorted(all_month_prices, reverse=True)[:3]
            peak_avg = sum(top_3_prices) / 3
        elif len(all_month_prices) > 0:
            # If fewer than 3 prices available, use what we have
            peak_avg = sum(all_month_prices) / len(all_month_prices)
        else:
            peak_avg = 0
        
        # Add the Peak Price Average as the first column
        resort_price_data[resort_key]["Peak Price Average"] = round(peak_avg, 2)
        
        # Process prices for each date in the date range
        for date in date_range:
            day_of_month = date.day
            date_key = date.strftime("%Y-%m-%d")
            
            # Check if we have pricing for this day
            day_str = str(day_of_month)
            
            if day_str in pricing and pricing[day_str] is not None:
                # Convert price to USD
                price_in_local = float(pricing[day_str])
                price_in_usd = convert_to_usd(price_in_local, currency)
                resort_price_data[resort_key][date_key] = round(price_in_usd, 2)
            else:
                # No price available for this day
                resort_price_data[resort_key][date_key] = None
        
        # Calculate average price for the selected date range (ignoring None values)
        date_prices = [resort_price_data[resort_key][date_key] 
                      for date_key in resort_price_data[resort_key] 
                      if date_key != "Peak Price Average" and resort_price_data[resort_key][date_key] is not None]
        
        avg_price = sum(date_prices) / len(date_prices) if date_prices else 0
        resort_price_data[resort_key]["Average Price"] = round(avg_price, 2)
        
        # Calculate deal score: how good of a deal are the selected dates compared to peak prices
        # Lower percentage means better deal
        if peak_avg > 0 and avg_price > 0:
            # Deal score is the ratio of current price to peak price (as a percentage)
            deal_score = (avg_price / peak_avg) * 100
            resort_price_data[resort_key]["Deal Score"] = round(deal_score, 1)
        else:
            resort_price_data[resort_key]["Deal Score"] = None
    
    return resort_price_data

def format_as_usd(df):
    """Format all numeric values as USD currency for display"""
    formatted_df = df.copy()
    for col in formatted_df.columns:
        if col == "Deal Score":
            # Format deal score as a percentage
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
        elif col != "Resort" and pd.api.types.is_numeric_dtype(formatted_df[col]):
            formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
    return formatted_df

def plot_price_trends(df, selected_resorts):
    """Create a line chart of prices over time for selected resorts"""
    if not selected_resorts:
        return None
    
    # Filter for selected resorts and exclude the average price column
    price_cols = [col for col in df.columns if col != "Average Price"]
    filtered_df = df.loc[selected_resorts, price_cols]
    
    # Convert to numeric for plotting
    plot_df = filtered_df.apply(pd.to_numeric, errors='coerce')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for resort in plot_df.index:
        ax.plot(plot_df.columns, plot_df.loc[resort], marker='o', label=resort)
    
    ax.set_title('Price Trends by Date')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='Resort', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def create_price_comparison_chart(df, selected_resorts):
    """Create a bar chart comparing average prices of selected resorts"""
    if not selected_resorts:
        return None
    
    # Get average prices for selected resorts
    avg_prices = df.loc[selected_resorts, "Average Price"]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_prices.plot(kind='bar', ax=ax, color='skyblue')
    
    ax.set_title('Average Price Comparison')
    ax.set_xlabel('Resort')
    ax.set_ylabel('Average Price (USD)')
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # Add price labels on top of bars
    for i, price in enumerate(avg_prices):
        ax.text(i, price + 50, f"${price:,.2f}", ha='center')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def find_best_nights(df, resort_key, month=6, year=2025, nights=3, consecutive=False):
    """Find the best nights deal for a specific resort
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing the pricing data
    resort_key : str
        Key for the resort in the DataFrame
    month : int
        Month to search in (1-12)
    year : int
        Year to search in
    nights : int
        Number of nights to find
    consecutive : bool
        Whether the nights must be consecutive
    
    Returns:
    --------
    tuple
        (best_dates, best_avg_price, best_deal_score)
    """
    # Create a calendar for the month
    month_start = datetime(year, month, 1)
    month_days = pd.date_range(start=month_start, periods=31, freq='D')
    month_days = [d for d in month_days if d.month == month]  # Only keep days in the target month
    
    best_deal = None
    best_dates = None
    best_avg_price = float('inf')
    
    # Get pricing for this resort
    if resort_key not in df.index:
        return None, None, None
    
    resort_data = df.loc[resort_key]
    peak_price = resort_data.get("Peak Price Average", 0)
    
    if peak_price <= 0:
        return None, None, None
    
    if consecutive:
        # Find best consecutive nights
        for i in range(len(month_days) - nights + 1):
            consecutive_dates = month_days[i:i+nights]
            date_strings = [d.strftime("%Y-%m-%d") for d in consecutive_dates]
            
            # Check if we have prices for all these dates
            prices = []
            valid_period = True
            
            for date_str in date_strings:
                if date_str in resort_data and pd.notnull(resort_data[date_str]):
                    prices.append(resort_data[date_str])
                else:
                    valid_period = False
                    break
                    
            if valid_period and prices:
                avg_price = sum(prices) / len(prices)
                deal_score = (avg_price / peak_price) * 100
                
                if avg_price < best_avg_price:
                    best_avg_price = avg_price
                    best_deal = deal_score
                    best_dates = consecutive_dates
    else:
        # Find best any nights (not necessarily consecutive)
        # First, collect all valid dates with prices
        valid_dates = []
        valid_prices = []
        
        for day in month_days:
            date_str = day.strftime("%Y-%m-%d")
            if date_str in resort_data and pd.notnull(resort_data[date_str]):
                valid_dates.append(day)
                valid_prices.append(resort_data[date_str])
        
        # If we don't have enough valid dates, return None
        if len(valid_dates) < nights:
            return None, None, None
        
        # Find the best n dates based on price
        date_price_pairs = list(zip(valid_dates, valid_prices))
        date_price_pairs.sort(key=lambda x: x[1])  # Sort by price (lowest first)
        
        best_n_dates = date_price_pairs[:nights]
        best_dates = [d[0] for d in best_n_dates]
        best_prices = [d[1] for d in best_n_dates]
        
        # Calculate the average price and deal score
        best_avg_price = sum(best_prices) / len(best_prices)
        best_deal = (best_avg_price / peak_price) * 100
        
        # Sort the dates chronologically for display
        best_dates.sort()
    
    return best_dates, best_avg_price, best_deal

def main():
    st.set_page_config(
        page_title="St. Regis Resort Price Finder",
        page_icon="🌴",
        layout="wide"
    )
    
    st.title("🌴 St. Regis Resort Price Finder")
    st.subheader("Find and compare prices across St. Regis resorts worldwide")
    
    # St. Regis resort data embedded directly in the code
    ST_REGIS_DATA = [
        {
          "name": "The St. Regis Saadiyat Island Resort, Abu Dhabi",
          "location": "Abu Dhabi, UAE",
          "currency": "AED",
          "pricing": {
            "1": 639, "2": 639, "3": 639, "4": 639, "5": 639, "6": 639, "7": 639,
            "8": 639, "9": 639, "10": 639, "11": 639, "12": 639, "13": 639, "14": 639,
            "15": 639, "16": 639, "17": 639, "18": 639, "19": 639, "20": 639, "21": 639,
            "22": 639, "23": 639, "24": 639, "25": 639, "26": 639, "27": 639, "28": 639
          }
        },
        {
          "name": "The St. Regis Bal Harbour Resort",
          "location": "Miami Beach, Florida, USA",
          "currency": "USD",
          "pricing": {
            "1": 894, "2": 894, "3": 309, "4": 309, "5": 902, "6": 932, "7": 932,
            "8": 819, "9": 309, "10": 309, "11": 309, "12": 1044, "13": 1044, "14": 309,
            "15": 1044, "16": 1044, "17": 1044, "18": 1044, "19": 1044, "20": 1044, "21": 1044,
            "22": 309, "23": 992, "24": 309, "25": 992, "26": 992, "27": 1037, "28": 1037
          }
        },
        {
          "name": "The St. Regis Mardavall Mallorca Resort",
          "location": "Mallorca, Spain",
          "currency": "EUR",
          "pricing": {
            "1": 1092, "2": 1092, "3": 1092, "4": 1092, "5": 1092, "6": 1092, "7": 1092,
            "8": 1092, "9": 1092, "10": 1092, "11": 1092, "12": None, "13": None, "14": None,
            "15": None, "16": None, "17": None, "18": None, "19": None, "20": None, "21": 1162,
            "22": None, "23": None, "24": None, "25": None, "26": None, "27": None, "28": None
          }
        },
        {
          "name": "The St. Regis Le Morne Resort",
          "location": "Le Morne, Mauritius",
          "currency": "USD",
          "pricing": {
            "2": 160, "3": 160, "4": 160, "5": 160, "6": 160, "7": 160,
            "8": 160, "9": 160, "10": 160, "11": 160, "12": 160, "13": 160, "14": 160,
            "15": 160, "16": 160, "17": 160, "18": 160, "19": 160, "20": 160, "21": 160,
            "22": 160, "23": 160, "24": 160, "25": 160, "26": 160, "27": 160, "28": 160
          }
        },
        {
          "name": "The St. Regis La Bahia Blanca Resort, Tamuda Bay",
          "location": "Tamuda Bay, Morocco",
          "currency": "MAD",
          "pricing": {
            "1": 1937, "2": 3630, "3": 3630, "4": 3630, "5": 3630, "6": 4400, "7": 4400,
            "8": 1937, "9": 4050, "10": 3700, "11": 3700, "12": 4400, "13": 1937, "14": 1937,
            "15": 1937, "16": 4400, "17": None, "18": 1937, "19": 1937, "20": 1937, "21": 4400,
            "22": 4400, "23": 4400, "24": 4400, "25": 1937, "26": 4400, "27": 4400, "28": 4400
          }
        },
        {
          "name": "The St. Regis Riyadh",
          "location": "Riyadh, Saudi Arabia",
          "currency": "SAR",
          "pricing": {
            "1": 560, "2": 560, "3": 560, "4": 560, "5": 560, "6": 560, "7": 560,
            "8": 560, "9": 1610, "10": 560, "11": 560, "12": 560, "13": 560, "14": 560,
            "15": 560, "16": 1750, "17": 1750, "18": 560, "19": 1610, "20": 560, "21": 1610,
            "22": 1610, "23": 1610, "24": 1540, "25": 1610, "26": 560, "27": 560, "28": 560
          }
        },
        {
          "name": "The St. Regis Sanya Yalong Bay Resort",
          "location": "Sanya, Hainan, China",
          "currency": "CNY",
          "pricing": {
            "1": 943, "2": 943, "3": 943, "4": 943, "5": 943, "6": 943, "7": 943,
            "8": 943, "9": 943, "10": 943, "11": 943, "12": 943, "13": 943, "14": 943,
            "15": 943, "16": 943, "17": 943, "18": 943, "19": 943, "20": 943, "21": 943,
            "22": 943, "23": 943, "24": 1306, "25": 1080, "26": 943, "27": 943, "28": 943
          }
        },
        {
          "name": "The St. Regis Bora Bora Resort",
          "location": "Bora Bora, French Polynesia",
          "currency": "XPF",
          "pricing": {
            "1": 181600, "3": 29713, "4": 29713, "5": 29713, "6": 29713, "7": 29713,
            "8": 29713, "11": 340000, "12": 210400, "13": 279200, "14": 258400,
            "15": 196000, "16": 272800, "18": 268000, "19": 272000, "20": 261600, "21": 253600,
            "22": 275200, "24": 250400, "25": 170400, "26": 176000, "27": 202400, "28": 256000
          }
        },
        {
          "name": "The St. Regis Maldives Vommuli Resort",
          "location": "Dhaalu Atoll, Maldives",
          "currency": "USD",
          "pricing": {} 
        },
        {
          "name": "The St. Regis Langkawi",
          "location": "Langkawi, Malaysia",
          "currency": "MYR",
          "pricing": {
            "1": 806, "2": 1200, "3": 806, "4": 806, "5": 1350, "6": 1350, "7": 806,
            "8": 1200, "9": 806, "10": 1200, "11": 1200, "12": 806, "13": 1350, "14": 1200,
            "15": 1200, "16": 806, "17": 1350, "18": 1575, "19": 1350, "20": 1200, "21": 1200,
            "22": 1200, "23": 806, "24": 1306, "25": 1200, "26": 806, "27": 1200, "28": 1200
          }
        },
        {
          "name": "The St. Regis Bali Resort",
          "location": "Nusa Dua, Indonesia",
          "currency": "IDR",
          "pricing": {
            "1": 6740000, "2": 2996973, "3": 6740000, "4": 6680000, "5": 6980000, "6": 7220000, "7": 7620000,
            "8": 6900000, "9": 6600000, "10": 6820000, "11": 7650000, "12": 6820000, "13": 6900000, "14": 6760000,
            "15": 7300000, "16": 7720000, "17": 6820000, "18": 8940000, "19": 27360000, "20": 13480000, "21": 10740000,
            "22": 7300000, "23": 8320000, "24": 6520000, "25": 8040000, "26": 6820000, "27": 7860000, "28": 7140000
          }
        },
        {
          "name": "The St. Regis Kanai Resort, Riviera Maya",
          "location": "Playa Del Carmen, Mexico",
          "currency": "USD",
          "pricing": {
            "1": 219, "2": 531, "3": 531, "4": 219, "5": 219, "6": 488, "7": 219,
            "8": 488, "9": 488, "10": 488, "11": 488, "12": 488, "13": 488, "14": 488,
            "15": 219, "17": 611, "18": 611, "19": 611, "20": 466, "21": 466,
            "22": 219, "23": 575, "24": 575, "25": 575, "26": 575, "27": 466, "28": 466
          }
        },
        {
          "name": "The St. Regis Punta Mita Resort",
          "location": "Punta de Mita, Mexico",
          "currency": "USD",
          "pricing": {
            "5": 738, "6": 878, "7": 878,
            "8": 941, "9": 857, "10": 857, "11": 857, "12": 857, "13": 703, "14": 703,
            "15": 582, "16": 582, "17": 582, "18": 582, "19": 582, "20": 588, "21": 838,
            "22": 838, "23": 838, "24": 838, "25": 582, "26": 582, "27": 542, "28": 542
          }
        },
        {
          "name": "The St. Regis Cap Cana Resort",
          "location": "Punta Cana, Dominican Republic",
          "currency": "USD",
          "pricing": {
            "1": 324, "2": 324, "3": 174, "4": 324, "5": 324, "6": 324, "7": 174,
            "8": 324, "9": 324, "10": 324, "11": 174, "12": 324, "13": 324, "14": 324,
            "15": 324, "16": 174, "17": 324, "18": 324, "19": 324, "20": 324, "21": 324,
            "22": 324, "23": 324, "24": 324, "25": 324, "26": 324, "27": 324, "28": 324
          }
        },
        {
          "name": "The St. Regis Bermuda Resort",
          "location": "St. George's, Bermuda",
          "currency": "USD",
          "pricing": {
            "1": 294, "2": 294, "3": 644, "4": 757, "5": 757, "6": 794, "7": 794,
            "8": 907, "9": 944, "10": 944, "11": 944, "12": 1057, "13": 1094, "14": 1094,
            "15": 719, "16": 869, "17": 869, "18": 869, "23": 682, "24": 869, "25": 869, "26": 869, "27": 869, "28": 1019
          }
        },
        {
          "name": "The St. Regis Bahia Beach Resort, Puerto Rico",
          "location": "Rio Grande, Puerto Rico",
          "currency": "USD",
          "pricing": {
            "1": 1146, "2": 1146, "3": 1146, "4": 1146, "5": 1146, "7": 368,
            "8": 707, "9": 707, "10": 707, "11": 707, "12": 368, "13": 728, "14": 728,
            "15": 707, "16": 707, "17": 707, "18": 707, "19": 1043, "20": 1281, "21": 1281,
            "22": 368, "23": 368, "24": 707, "25": 368, "26": 368, "27": 728, "28": 728
          }
        }
    ]

    # Use the embedded data
    resort_data = ST_REGIS_DATA
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Price Table", "Monthly Deal Finder"])
    
    with tab1:
        # Price Table section
        st.header("Resort Price Table")
        
        # Price Table specific sidebar controls
        st.sidebar.header("Price Table Options")
        
        # Date range controls for Price Table
        start_date = st.sidebar.date_input(
            "Start Date", 
            value=datetime(2025, 6, 17),
            min_value=datetime(2025, 6, 1),
            max_value=datetime(2025, 6, 30)
        )
        
        end_date = st.sidebar.date_input(
            "End Date", 
            value=datetime(2025, 6, 21),
            min_value=start_date,
            max_value=datetime(2025, 6, 30)
        )
        
        # Create a range of dates for Price Table
        date_range = pd.date_range(start=start_date, end=end_date)
        date_strings = [d.strftime("%Y-%m-%d") for d in date_range]
        resort_price_data = process_json_data(resort_data, date_range)
        
        # Create a DataFrame with all the required data
        df = pd.DataFrame.from_dict(resort_price_data, orient='index')
        
        # Add max price slider after we have the data
        if len(df) > 0 and "Average Price" in df.columns:
            price_max = min(5000, df["Average Price"].max())
            max_price = st.sidebar.slider(
                "Max Price (USD)", 
                min_value=0, 
                max_value=int(price_max), 
                value=min(1000, int(price_max)),
                step=100
            )
        else:
            max_price = st.sidebar.slider(
                "Max Price (USD)", 
                min_value=0, 
                max_value=5000, 
                value=1000, 
                step=100
            )
        
        # Convert the DataFrame to a pivot table for display
        if len(df) > 0:
            # Make sure dates are in the columns and resorts are in the rows
            date_columns = [col for col in df.columns if col.startswith("2025-")]
            pivot_df = df.copy()
            
            # Add peak price average column
            if "Peak Price Average" not in pivot_df.columns:
                pivot_df = calculate_peak_price_average(pivot_df)
            
            # Calculate average price for the selected date range
            pivot_df["Average Price"] = pivot_df[date_columns].mean(axis=1)
            
            # Calculate Deal Score (current average price as % of peak price)
            pivot_df["Deal Score"] = (pivot_df["Average Price"] / pivot_df["Peak Price Average"]) * 100
            
            # Create color for the Deal Score
            def color_deals(val):
                if pd.isna(val):
                    return ''
                
                # Handle percentage strings like '100.0%'
                if isinstance(val, str) and '%' in val:
                    try:
                        # Remove the '%' character and convert to float
                        val = float(val.replace('%', ''))
                    except (ValueError, TypeError):
                        return ''  # If conversion fails, return empty style
                else:
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        return ''  # If conversion fails, return empty style
                
                # Apply color based on value with higher contrast
                if val < 70:
                    return 'background-color: #1a8836; color: white; font-weight: bold;'  # Darker Green (excellent deal)
                elif val < 85:
                    return 'background-color: #0c7a99; color: white; font-weight: bold;'  # Darker Blue (good deal)
                elif val < 95:
                    return 'background-color: #e6ac00; color: black; font-weight: bold;'  # Darker Yellow (fair deal)
                else:
                    return 'background-color: #d9dde0; color: black;'  # Darker Gray (regular price)
            
            # Filter by max price
            filtered_df = pivot_df[pivot_df["Average Price"] <= max_price]
            
            if len(filtered_df) == 0:
                st.warning(f"No resorts found with average price below ${max_price}")
            else:
                # Show the filtered DataFrame
                display_df = filtered_df[date_columns + ["Average Price", "Peak Price Average", "Deal Score"]].copy()
                
                # Format the table for display
                display_df = format_as_usd(display_df)
                # Make sure Deal Score is formatted properly
                def format_deal_score(x):
                    if pd.isnull(x):
                        return ""
                    # Check if it's already a string percentage
                    if isinstance(x, str) and '%' in x:
                        return x
                    # Otherwise convert to float and format
                    try:
                        return f"{float(x):.1f}%"
                    except (ValueError, TypeError):
                        return str(x)
                
                display_df["Deal Score"] = display_df["Deal Score"].apply(format_deal_score)
                
                # Apply styling to the Deal Score column
                styled_df = display_df.style.applymap(lambda x: color_deals(x), subset=["Deal Score"])
                
                st.dataframe(styled_df)
                
                # Deal Score legend
                st.write("**Deal Score Legend:**")
                legend_cols = st.columns(4)
                legend_cols[0].markdown("<span style='background-color: #28a745; color: white; padding: 5px;'>< 70% - Excellent Deal</span>", unsafe_allow_html=True)
                legend_cols[1].markdown("<span style='background-color: #17a2b8; color: white; padding: 5px;'>70-85% - Good Deal</span>", unsafe_allow_html=True)
                legend_cols[2].markdown("<span style='background-color: #ffc107; color: black; padding: 5px;'>85-95% - Fair Deal</span>", unsafe_allow_html=True)
                legend_cols[3].markdown("<span style='background-color: #f8f9fa; color: black; padding: 5px;'>95%+ - Regular Price</span>", unsafe_allow_html=True)
                
                # Export option
                st.download_button(
                    label="Download Price Table as CSV",
                    data=display_df.to_csv(index=True),
                    file_name=f"St_Regis_Resort_Prices_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Price Recommendations
                st.subheader("Price Recommendations")
                sorted_df = filtered_df.sort_values(by="Deal Score")
                
                # Best Deal
                if len(sorted_df) > 0:
                    best_deal = sorted_df.iloc[0]
                    best_deal_score = best_deal["Deal Score"]
                    best_deal_resort = best_deal.name
                    best_deal_price = best_deal["Average Price"]
                    best_deal_peak = best_deal["Peak Price Average"]
                    
                    st.write(f"**Best Value:** {best_deal_resort} at ${best_deal_price:.2f} per night (${best_deal_peak:.2f} peak price, {best_deal_score:.1f}% score)")
                    
                    # If we have multiple resorts, find the best luxury option
                    if len(sorted_df) > 1:
                        # Define luxury resorts (those with peak price in top 33%)
                        price_threshold = sorted_df["Peak Price Average"].quantile(0.67)
                        luxury_df = sorted_df[sorted_df["Peak Price Average"] >= price_threshold]
                        
                        if len(luxury_df) > 0:
                            best_luxury = luxury_df.sort_values(by="Deal Score").iloc[0]
                            best_luxury_score = best_luxury["Deal Score"]
                            best_luxury_resort = best_luxury.name
                            best_luxury_price = best_luxury["Average Price"]
                            best_luxury_peak = best_luxury["Peak Price Average"]
                            
                            st.write(f"**Best Luxury Deal:** {best_luxury_resort} at ${best_luxury_price:.2f} per night (${best_luxury_peak:.2f} peak price, {best_luxury_score:.1f}% score)")
                        
                        # Budget option - lowest average price
                        budget_pick = sorted_df.sort_values(by="Average Price").iloc[0]
                        budget_resort = budget_pick.name
                        budget_price = budget_pick["Average Price"]
                        budget_score = budget_pick["Deal Score"]
                        
                        st.write(f"**Budget Pick:** {budget_resort} at ${budget_price:.2f} per night ({budget_score:.1f}% score)")
    
    with tab2:
        # Monthly Deal Finder section
        st.header("Monthly Deal Finder")
        
        # Monthly Deal Finder specific sidebar controls
        st.sidebar.header("Monthly Deal Finder Options")
        
        # Month to view
        month_to_view = st.sidebar.selectbox(
            "Month for Best Deals Analysis", 
            options=[6], 
            format_func=lambda x: f"June {2025}"
        )
        
        # Number of nights for deal analysis
        nights_count = st.sidebar.slider(
            "Number of Nights for Deal Analysis", 
            min_value=2, 
            max_value=7, 
            value=3,
            help="Find the best deals for this many nights"
        )
        
        # Consecutive nights option
        consecutive_only = st.sidebar.checkbox(
            "Must be consecutive nights", 
            value=False,
            help="If checked, only consecutive night stays will be considered. Otherwise, the best individual nights will be found."
        )
        
        # Create a calendar month view (June 2025) - 30 days
        month_days = pd.date_range(start=datetime(2025, 6, 1), end=datetime(2025, 6, 30))
        
        # Create tabs for Deal Calendar and Best Deals Summary
        mdf_tab1, mdf_tab2 = st.tabs(["Deal Calendar", "Best Deals Summary"])
        
        with mdf_tab1:
            if consecutive_only:
                st.write("This calendar shows the best consecutive night stays for each resort during the entire month.")
            else:
                st.write("This calendar highlights the best individual nights for each resort during the entire month.")
        
        # Calculate deal data for all resorts
        best_deals_data = []
        resort_calendars = {}
        
        # Process each resort
        for resort_key in df.index:
            # Get the pricing data for all days in the month
            resort_data = df.loc[resort_key]
            peak_price = resort_data.get("Peak Price Average", 0)
            
            if peak_price <= 0:
                continue
                
            # Collect all day pricing and deal scores
            day_prices = {}
            day_deal_scores = {}
            
            # Make sure we have data for every day in the month
            for day in month_days:
                date_str = day.strftime("%Y-%m-%d")
                day_num = str(day.day)  # Get the day number as string (e.g., "1", "2", etc.)
                
                # Check if this date exists in our processed data
                if date_str in resort_data and pd.notnull(resort_data[date_str]):
                    price = resort_data[date_str]
                    deal_score = (price / peak_price) * 100
                # Alternative method - try to get directly from the JSON pricing data
                elif "pricing" in resort_data and day_num in resort_data["pricing"]:
                    # Extract from original pricing data
                    price_in_local = float(resort_data["pricing"][day_num])
                    if "currency" in resort_data:
                        currency = resort_data["currency"]
                    else:
                        currency = "USD"
                    price = convert_to_usd(price_in_local, currency)
                    deal_score = (price / peak_price) * 100 if peak_price > 0 else 100
                else:
                    # If date is missing or None, use a placeholder value
                    price = None
                    deal_score = None
                
                day_prices[day] = price
                day_deal_scores[day] = deal_score
            
            # Find best deals (either consecutive or individual)
            best_dates, best_price, deal_score = find_best_nights(
                df, resort_key, month=6, year=2025, nights=nights_count, consecutive=consecutive_only
            )
            
            if not best_dates or not best_price or not deal_score:
                continue
                
            # Calculate savings
            savings_pct = 100 - deal_score
            savings_amt = peak_price - best_price
            
            # Format dates differently based on consecutive flag
            if consecutive_only:
                start_date = best_dates[0]
                end_date = best_dates[-1]
                date_display = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            else:
                date_display = ", ".join([d.strftime("%b %d") for d in best_dates])
            
            # Store the deal info
            best_deals_data.append({
                "Resort": resort_key,
                "Best Dates": date_display,
                "Average Price": best_price,
                "Peak Price": peak_price,
                "Deal Score": deal_score,
                "Savings %": savings_pct,
                "Savings Amount": savings_amt
            })
            
            # Save the calendar data for this resort
            resort_calendars[resort_key] = {
                "day_prices": day_prices,
                "day_deal_scores": day_deal_scores,
                "best_dates": best_dates
            }
        
        # Display deal calendars for each resort
        for resort_key, calendar_data in resort_calendars.items():
            st.subheader(resort_key)
            
            # Create the calendar grid for this resort
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            
            # Display day headers
            for i, day in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
                [col1, col2, col3, col4, col5, col6, col7][i].write(f"**{day}**")
            
            # Create the calendar layout
            first_day = month_days[0]
            first_weekday = first_day.weekday()  # Monday is 0, Sunday is 6
            
            # Fill in the calendar
            day_index = 0
            for week in range(5):  # Max 5 weeks in a month
                cols = [col1, col2, col3, col4, col5, col6, col7]
                
                # Skip days before the first of the month
                if week == 0:
                    for i in range(first_weekday):
                        cols[i].write("")
                
                # Fill in the days
                for weekday in range(7):
                    if week == 0 and weekday < first_weekday:
                        continue
                    
                    if day_index >= len(month_days):
                        break
                    
                    day = month_days[day_index]
                    day_str = day.strftime("%d")
                    
                    # Check if this day is one of the best deal days
                    is_best_day = day in calendar_data["best_dates"]
                    
                    # Get price and deal score
                    price = calendar_data["day_prices"].get(day, None)
                    deal_score = calendar_data["day_deal_scores"].get(day, None)
                    
                    # Format the cell content
                    if price is not None and deal_score is not None:
                        # Determine color based on deal score with higher contrast
                        if is_best_day:
                            # Highlight as one of the best days - bold vibrant green
                            cols[weekday].markdown(f"<div style='background-color:#1b9e41;color:white;padding:5px;border-radius:3px;text-align:center;font-weight:bold;'><b>{day_str}</b><br>${price:.0f}<br>{deal_score:.0f}%</div>", unsafe_allow_html=True)
                        elif deal_score < 70:
                            # Excellent deal - darker green with better contrast
                            cols[weekday].markdown(f"<div style='background-color:#2da44e;color:white;padding:5px;border-radius:3px;text-align:center;font-weight:bold;'>{day_str}<br>${price:.0f}<br>{deal_score:.0f}%</div>", unsafe_allow_html=True)
                        elif deal_score < 85:
                            # Good deal - darker blue with better contrast
                            cols[weekday].markdown(f"<div style='background-color:#0969da;color:white;padding:5px;border-radius:3px;text-align:center;font-weight:bold;'>{day_str}<br>${price:.0f}<br>{deal_score:.0f}%</div>", unsafe_allow_html=True)
                        elif deal_score < 95:
                            # Fair deal - deeper yellow with better contrast
                            cols[weekday].markdown(f"<div style='background-color:#d29922;color:black;padding:5px;border-radius:3px;text-align:center;font-weight:bold;'>{day_str}<br>${price:.0f}<br>{deal_score:.0f}%</div>", unsafe_allow_html=True)
                        else:
                            # Regular price - darker gray with better contrast
                            cols[weekday].markdown(f"<div style='background-color:#959da5;color:white;padding:5px;border-radius:3px;text-align:center;'>{day_str}<br>${price:.0f}<br>{deal_score:.0f}%</div>", unsafe_allow_html=True)
                    else:
                        # No data - use black background for unavailable days
                        cols[weekday].markdown(f"<div style='background-color:#000000;color:#808080;padding:5px;border-radius:3px;text-align:center;'>{day_str}<br>No data</div>", unsafe_allow_html=True)
                    
                    day_index += 1
            
            st.markdown(f"""<div style='margin-top:10px;margin-bottom:20px;'>
            <span style='background-color:#1b9e41;color:white;padding:2px 8px;border-radius:3px;font-weight:bold;'>Best {nights_count} Days</span>
            <span style='background-color:#2da44e;color:white;padding:2px 8px;border-radius:3px;margin-left:10px;font-weight:bold;'>Excellent Deal</span>
            <span style='background-color:#0969da;color:white;padding:2px 8px;border-radius:3px;margin-left:10px;font-weight:bold;'>Good Deal</span>
            <span style='background-color:#d29922;color:black;padding:2px 8px;border-radius:3px;margin-left:10px;font-weight:bold;'>Fair Deal</span>
            <span style='background-color:#959da5;color:white;padding:2px 8px;border-radius:3px;margin-left:10px;'>Regular Price</span>
            <span style='background-color:#000000;color:#808080;padding:2px 8px;border-radius:3px;margin-left:10px;'>No Data</span>
            </div>""", unsafe_allow_html=True)
            
            st.markdown("---")
        
        # Create a matrix of deal scores for all resorts and days
        calendar_data = {}
        
        for resort_key in df.index:
            resort_row = {}
            peak_price = df.loc[resort_key, "Peak Price Average"]
            
            if peak_price > 0:
                for day in month_days:
                    day_str = day.strftime("%Y-%m-%d")
                    day_num = day.strftime("%d")
                    
                    if day_str in df.columns and pd.notnull(df.loc[resort_key, day_str]):
                        price = df.loc[resort_key, day_str]
                        deal_score = (price / peak_price) * 100
                        resort_row[day_num] = deal_score
                    else:
                        resort_row[day_num] = None
                
                calendar_data[resort_key] = resort_row
        
        if calendar_data:
            calendar_df = pd.DataFrame.from_dict(calendar_data, orient='index')
            
            # Function to color code the deal scores in the calendar
            def color_calendar_deals(val):
                if pd.notnull(val):
                    if val < 70:
                        return 'background-color: #28a745; color: white'
                    elif val < 85:
                        return 'background-color: #17a2b8; color: white'
                    elif val < 95:
                        return 'background-color: #ffc107; color: black'
                    else:
                        return 'background-color: #f8f9fa; color: black'
                return 'background-color: #eeeeee'
            
            # Function to format the values as percentages
            def format_pct(val):
                if pd.notnull(val):
                    return f"{val:.0f}%"
                return ""
            
            # Apply styling and formatting directly without trying to use another DataFrame as formatter
            styled_calendar = calendar_df.style.applymap(color_calendar_deals).format(format_pct)
            
            # Display the calendar heatmap with deal scores
            st.dataframe(styled_calendar, use_container_width=True)
            
            st.info('''
            **Deal Score Legend:**
            <span style='background-color:#28a745;color:white;padding:2px 8px;border-radius:3px;'>Excellent Deal</span>
            <span style='background-color:#17a2b8;color:white;padding:2px 8px;border-radius:3px;margin-left:10px;'>Good Deal</span>
            <span style='background-color:#ffc107;color:black;padding:2px 8px;border-radius:3px;margin-left:10px;'>Fair Deal</span>
            <span style='background-color:#f8f9fa;color:black;padding:2px 8px;border-radius:3px;margin-left:10px;'>Regular Price</span>
            Grey = No pricing available
            ''')
        else:
            st.warning("No deal data available for the selected parameters.")

if __name__ == '__main__':
    main()
