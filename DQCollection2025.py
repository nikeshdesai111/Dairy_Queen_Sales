import pandas as pd
import os

# List of month identifiers
months = ['Jan25', 'Feb25', 'Mar25', 'Apr25', 'May25', 'Jun25']

# Base file path
base_path = '/Users/nickydesai/Desktop/DairyQueen/2025'

# Blizzard items of interest
valid_items = ['SM BLIZZARD', 'MD BLIZZARD', 'LG BLIZZARD']

for month in months:
    input_file = os.path.join(base_path, f'{month}.csv')
    output_file = os.path.join(base_path, f'formatted{month}.csv')

    print(f'\nProcessing {month}...')

    # Read all columns as strings to avoid dtype issues
    df_raw = pd.read_csv(input_file, header=None, dtype=str, low_memory=False, encoding='latin1')

    # Extract relevant columns: Date (col 1), Item Name (col 5), Price (col 9)
    df_clean = df_raw[[1, 5, 9]].copy()
    df_clean.columns = ['Date', 'Item Name', 'Price']

    # Clean and drop missing
    df_clean['Date'] = df_clean['Date'].str.strip()
    df_clean['Item Name'] = df_clean['Item Name'].str.strip()
    df_clean['Price'] = df_clean['Price'].str.replace('$', '', regex=False).str.strip()

    # Convert price to numeric
    df_clean['Price'] = pd.to_numeric(df_clean['Price'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Date', 'Item Name', 'Price'])

    # Filter for Blizzard items
    df_clean = df_clean[df_clean['Item Name'].isin(valid_items)]

    if df_clean.empty:
        print(f'No Blizzard sales found for {month}. Skipping.')
        continue

    # Group by Date: count items, sum prices
    df_grouped = df_clean.groupby('Date').agg(
        Item_Sales=('Item Name', 'count'),
        Total_Price=('Price', 'sum')
    ).reset_index()

    # Compute average price
    df_grouped['Avg_Price'] = df_grouped['Total_Price'] / df_grouped['Item_Sales']

    # Drop Total_Price column before saving
    df_grouped.drop(columns=['Total_Price'], inplace=True)

    # Convert date column and sort
    df_grouped['Date'] = pd.to_datetime(df_grouped['Date'], errors='coerce')
    df_grouped = df_grouped.dropna(subset=['Date'])
    df_grouped.set_index('Date', inplace=True)
    df_grouped.sort_index(inplace=True)

    # Save to CSV
    df_grouped.to_csv(output_file)
    print(f'Saved: {output_file}')