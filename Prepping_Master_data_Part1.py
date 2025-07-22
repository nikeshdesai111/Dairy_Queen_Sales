import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.metrics import mean_squared_error

# ---------------------------- Load Data ---------------------------- #

# Load the CSV and parse dates
file_path = '/Users/nickydesai/Desktop/DairyQueen/all_years_blizzard_sales.csv'
df = pd.read_csv(file_path, parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Load the forecasting dataset
forecast_path = '/Users/nickydesai/Desktop/DairyQueen/Forecasting_Future.csv'
df_forecast = pd.read_csv(forecast_path, parse_dates=['Date'])
df_forecast.set_index('Date', inplace=True)

# Merge both datasets on the Date index, keeping all columns
df_combined = pd.merge(df, df_forecast, left_index=True, right_index=True, how='outer')
df_combined.sort_index(inplace=True)

# ---------------------- Pre-cleaning: Handle 0 sales ---------------------- #

# Set Avg_Price to NaN where Item_Sales is 0 (before modeling)
df_combined.loc[df_combined['Item_Sales'] == 0, 'Avg_Price'] = np.nan

# ---------------------------- Fit Logarithmic Model ---------------------------- #

# Training data (before July 1, 2025)
train_df = df_combined.loc[df_combined.index <= '2025-06-30'].copy()

# Mask for valid (non-missing and non-zero) prices
mask = train_df['Avg_Price'].notna()

# Add ordinal date and log date
train_df['Date_Ordinal'] = train_df.index.map(pd.Timestamp.toordinal)
train_df['Log_Date'] = np.log(train_df['Date_Ordinal'])

# Prepare X and y
X = train_df.loc[mask, 'Log_Date'].values.reshape(-1, 1)
y = train_df.loc[mask, 'Avg_Price'].values

# Fit the model
log_model = LinearRegression()
log_model.fit(X, y)

# Predict on full training data
X_all = train_df['Log_Date'].values.reshape(-1, 1)
train_df['Avg_Price_Pred_Log'] = log_model.predict(X_all)

# Evaluate MSE
mse = mean_squared_error(y, train_df.loc[mask, 'Avg_Price_Pred_Log'])

# Plot
plt.figure(figsize=(12, 6))
plt.plot(train_df.loc[mask].index, train_df.loc[mask, 'Avg_Price'], label='Actual Avg_Price', alpha=0.7)
plt.plot(train_df.loc[mask].index, train_df.loc[mask, 'Avg_Price_Pred_Log'], color='green', label='Logarithmic Fit', linewidth=2)
plt.title('Average Price Over Time with Logarithmic Best Fit (Training Data Only)')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print equation
a = log_model.coef_[0]
b = log_model.intercept_
print("Logarithmic Equation (Training Data):")
print(f"Avg_Price = {a:.5f} * log(Date_Ordinal) + {b:.2f}")
print(f"Mean Squared Error (MSE): {mse:.5f}")

# ---------------------------- Impute Future Avg_Price ---------------------------- #

# Test data (after July 1, 2025)
test_df = df_combined.loc[df_combined.index > '2025-06-30'].copy()

# Add ordinal and log date
test_df['Date_Ordinal'] = test_df.index.map(pd.Timestamp.toordinal)
test_df['Log_Date'] = np.log(test_df['Date_Ordinal'])

# Predict average price using the trained model
X_test = test_df['Log_Date'].values.reshape(-1, 1)
predicted_prices = log_model.predict(X_test)

# Impute only where Avg_Price is missing
mask_missing = test_df['Avg_Price'].isna()
test_df.loc[mask_missing, 'Avg_Price'] = predicted_prices[mask_missing.values]

# Assign imputed values back into the combined dataset
df_combined.loc[test_df.index, 'Avg_Price'] = test_df['Avg_Price']

# Confirm result
print(df_combined.loc[df_combined.index > '2025-06-30', 'Avg_Price'].head())

# ---------------------------- Add Store Hours ---------------------------- #

# Create day of week column: Monday=0, Sunday=6
df_combined['DayOfWeek'] = df_combined.index.dayofweek
is_weekend = df_combined['DayOfWeek'].isin([4, 5])  # Friday or Saturday
cutoff = pd.Timestamp("2025-07-18")

# Default store hours
df_combined['Store_Hours'] = 12
df_combined.loc[(df_combined.index < cutoff) & is_weekend, 'Store_Hours'] = 12.5
df_combined.loc[(df_combined.index >= cutoff) & is_weekend, 'Store_Hours'] = 14

# Set store hours to NaN if no sales
df_combined.loc[df_combined['Item_Sales'] == 0, 'Store_Hours'] = np.nan

# Clean up
df_combined.drop(columns='DayOfWeek', inplace=True)


# ---------------------------- Save Final Dataset ---------------------------- #

output_path = '/Users/nickydesai/Desktop/DairyQueen/combined_dataset.csv'
df_combined.to_csv(output_path)