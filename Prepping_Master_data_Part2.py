import pandas as pd

# Set path
base_path = '/Users/nickydesai/Desktop/DairyQueen'

# Load combined dataset
combined_path = f'{base_path}/combined_dataset.csv'
df_combined = pd.read_csv(combined_path, parse_dates=['Date'])
df_combined.set_index('Date', inplace=True)

# Load weather dataset
weather_path = f'{base_path}/weather_data.csv'
df_weather = pd.read_csv(weather_path, parse_dates=['Date'])
df_weather.set_index('Date', inplace=True)

# Select only relevant columns from weather data
weather_cols = ['Max_Temp', 'Min_Temp', 'Avg_Temp', 'Precipitation']
df_weather = df_weather[weather_cols]

# ✅ Merge weather data into combined dataset based on Date index
# This will overwrite any existing values in these columns where weather data is not null
for col in weather_cols:
    if col in df_combined.columns:
        # Fill missing values in combined dataset with values from weather data
        df_combined[col] = df_combined[col].combine_first(df_weather[col])
    else:
        # If column doesn't exist in combined dataset, just add it
        df_combined[col] = df_weather[col]

# ✅ Save updated dataset
output_path = f'{base_path}/combined_dataset_with_weather.csv'
df_combined.to_csv(output_path)

print(f"Merged dataset saved to: {output_path}")