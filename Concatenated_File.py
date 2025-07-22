import pandas as pd
import os

# Base directory
base_path = '/Users/nickydesai/Desktop/DairyQueen'

# Years and month identifiers
years = ['2022', '2023', '2024', '2025']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Items of interest
valid_items = ['SM BLIZZARD', 'MD BLIZZARD', 'LG BLIZZARD']

# Master DataFrame to hold all years
all_data = pd.DataFrame()

for year in years:
    year_path = os.path.join(base_path, year)
    yearly_data = pd.DataFrame()

    for month in months:
        filename = f'{month}{year[-2:]}.csv'
        filepath = os.path.join(year_path, filename)

        if not os.path.exists(filepath):
            print(f"Skipping missing file: {filepath}")
            continue

        try:
            df_raw = pd.read_csv(filepath, header=None, encoding='windows-1252')
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        # Extract relevant columns: Date (1), Item Name (5), Price (8)
        df_clean = df_raw[[1, 5, 9]].copy()
        df_clean.columns = ['Date', 'Item Name', 'Price']

        # Clean up
        df_clean['Date'] = df_clean['Date'].astype(str).str.strip()
        df_clean['Item Name'] = df_clean['Item Name'].astype(str).str.strip()
        df_clean['Price'] = df_clean['Price'].astype(str).str.replace('$', '', regex=False).str.strip()

        # Convert price to numeric
        df_clean['Price'] = pd.to_numeric(df_clean['Price'], errors='coerce')

        # Drop invalid rows
        df_clean = df_clean.dropna(subset=['Date', 'Item Name', 'Price'])

        # Filter for valid Blizzard items
        df_clean = df_clean[df_clean['Item Name'].isin(valid_items)]

        if df_clean.empty:
            continue

        # Group by Date: count items and sum price
        df_grouped = df_clean.groupby('Date').agg(
            Item_Sales=('Item Name', 'count'),
            Total_Price=('Price', 'sum')
        ).reset_index()

        # Calculate average price per Blizzard
        df_grouped['Avg_Price'] = df_grouped['Total_Price'] / df_grouped['Item_Sales']

        # Drop Total_Price if not needed in output
        df_grouped.drop(columns=['Total_Price'], inplace=True)

        # Clean up date format
        df_grouped['Date'] = pd.to_datetime(df_grouped['Date'], errors='coerce')
        df_grouped = df_grouped.dropna(subset=['Date'])
        df_grouped.set_index('Date', inplace=True)
        df_grouped.sort_index(inplace=True)

        # Append monthly data to yearly data
        yearly_data = pd.concat([yearly_data, df_grouped])

    # Append year to full dataset
    all_data = pd.concat([all_data, yearly_data])

# Final sort
all_data.sort_index(inplace=True)

# Generate complete daily date range
full_index = pd.date_range(start=all_data.index.min() - pd.Timedelta(days=1),
                           end=all_data.index.max(), freq='D')

# Fill missing dates: 0 sales, NaN for Avg_Price
all_data = all_data.reindex(full_index)
all_data['Item_Sales'] = all_data['Item_Sales'].fillna(0).astype(int)

# Avg_Price stays NaN where no Blizzards sold
all_data.index.name = 'Date'

# Save final CSV
output_path = os.path.join(base_path, 'all_years_blizzard_sales.csv')
all_data.to_csv(output_path)

print("✅ All data processed and saved to 'all_years_blizzard_sales.csv'")

