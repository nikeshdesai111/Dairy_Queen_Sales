import pandas as pd
import os

# List of month identifiers
months = ['Jan22', 'Feb22', 'Mar22', 'Apr22', 'May22', 'Jun22',
          'Jul22', 'Aug22', 'Sep22', 'Oct22', 'Nov22', 'Dec22']

# Base file path
base_path = '/Users/nickydesai/Desktop/DairyQueen/2022'

# Blizzard items of interest
valid_items = ['SM BLIZZARD', 'MD BLIZZARD', 'LG BLIZZARD']

for month in months:
    input_file = os.path.join(base_path, f'{month}.csv')
    output_file = os.path.join(base_path, f'formatted{month}.csv')

    print(f'\nProcessing {month}...')

    # Read all columns as strings to avoid dtype issues
    df_raw = pd.read_csv(input_file, header=None, dtype=str, low_memory=False)

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















#Need to have 0 sales for the days that have no sales, fill in the missing dates






#Showed this for a day but we can now do for a month. Then we can save them to different csv files per month. Then we can just concatenate them together and we should have a decent enough dataset.
#If we want weather data, we will have to merge that in later and we should figure out what to use for that. We should also think about what other variables we might want to include. 