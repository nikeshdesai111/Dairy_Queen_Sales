import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit



# Periodogram plotting function
def plot_periodogram(ts, detrend='constant', ax=None):
    from scipy import signal

    freqs, spectrum = signal.periodogram(ts, detrend=detrend)

    if ax is None:
        fig, ax = plt.subplots()

    ax.step(freqs, spectrum, color='purple')
    ax.set_xscale('log')
    ax.set_ylabel('Power')
    ax.set_xlabel('Frequency (1/day)')
    ax.set_title('Periodogram')

    # Define tick positions and labels
    freq_ticks = [
        1/365,    # Annual
        1/182.5,  # Semiannual
        1/91.25,  # Quarterly
        1/60.83,  # Bimonthly
        1/30.42,  # Monthly
        1/14,     # Biweekly
        1/7,      # Weekly
        1/3.5     # Semiweekly
    ]

    labels = [
        'Annual', 'Semiannual', 'Quarterly', 'Bimonthly',
        'Monthly', 'Biweekly', 'Weekly', 'Semiweekly'
    ]

    ax.set_xticks(freq_ticks)
    ax.set_xticklabels(labels, rotation=45)
    ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()

#Seasonal plot function

def seasonal_plot(y, period='weekday', freq='day'):
    """
    Plot seasonal patterns for sales data.

    Parameters:
    - y: pd.Series with DateTimeIndex
    - period: str ('weekday' or 'month')
    - freq: str ('day', 'hour', etc.)

    Returns:
    - ax: matplotlib axis with the seasonal plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create DataFrame for plotting
    df = pd.DataFrame({'sales': y.copy()})
    df['date'] = df.index

    # Define period and x-axis grouping
    if period == 'weekday':
        df['period'] = df['date'].dt.day_name()
        df['period'] = pd.Categorical(df['period'],
                                      categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                      ordered=True)
        x_label = 'Day of Week'
        df_grouped = df.groupby('period')['sales'].median().reset_index()
        sns.barplot(x='period', y='sales', data=df_grouped, ax=ax, palette='viridis')
    elif period == 'month' and freq == 'day':
        df['period'] = df['date'].dt.month_name()
        df['day'] = df['date'].dt.day
        x_label = 'Day of Month'

        month_order = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        df['period'] = pd.Categorical(df['period'], categories=month_order, ordered=True)

        df_nonzero = df[df['sales'] > 0]
        df_grouped = df_nonzero.groupby(['period', 'day'])['sales'].mean().reset_index()
        df_grouped['period'] = pd.Categorical(df_grouped['period'], categories=month_order, ordered=True)

        palette = sns.color_palette("coolwarm", n_colors=12)

        sns.lineplot(
            x='day',
            y='sales',
            hue='period',
            data=df_grouped,
            ci=False,
            ax=ax,
            palette=palette,
            legend='full'
        )

        # Adjust y-axis
        max_y = df_grouped['sales'].max()
        ax.set_ylim(0, max_y * 1.15)

        # Optional: Make y-axis ticks more readable
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))

    else:
        raise ValueError("Unsupported combination. Try period='weekday' or period='month' with freq='day'.")

    # Final labeling
    ax.set_title(f"Seasonal Plot by {period.capitalize()} ({freq})")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Sales")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    if period == 'month':
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    fig.tight_layout(rect=[0, 0, 0.85, 1] if period == 'month' else [0, 0, 1, 1])
    return ax

#___________________________________________________________________________________________________________________________________________
# Load the full dataset once
file_path = '/Users/nickydesai/Desktop/DairyQueen/combined_dataset_with_weather.csv'
full_df = pd.read_csv(file_path, parse_dates=['Date'])
full_df.set_index('Date', inplace=True)

# Original full dataset
train_df = full_df.loc[:'2025-06-30']
test_df = full_df.loc['2025-07-01':]

# Now split train_df into train and validation sets
train_cutoff = '2024-12-31'

train_df_train = train_df.loc[:train_cutoff]  # training portion
train_df_val = train_df.loc[pd.to_datetime(train_cutoff) + pd.Timedelta(days=1):]  # validation portion

# Drop the irrelevant columns
train_timeseries_df_train = train_df_train.drop(columns=['Avg_Price', 'Avg_Temp', 'Precipitation', 'Store_Hours', 'Min_Temp', 'Max_Temp'])
train_timeseries_df_val = train_df_val.drop(columns=['Avg_Price', 'Avg_Temp', 'Precipitation', 'Store_Hours', 'Min_Temp', 'Max_Temp'])
test_timeseries_df = test_df.drop(columns=['Avg_Price', 'Avg_Temp', 'Precipitation', 'Item_Sales', 'Store_Hours', 'Min_Temp', 'Max_Temp'])


# Display data
print(train_timeseries_df_train.head())
print(test_timeseries_df.head())

# Find days with 0 sales
potential_holidays = train_timeseries_df_train[train_timeseries_df_train['Item_Sales'] == 0]
print(potential_holidays)


#-------------------------------------------------------------------------------------------------------------------------------------
# Plot periodogram of sales
fig, ax1 = plt.subplots()
plot_periodogram(train_timeseries_df_train['Item_Sales'], ax=ax1)
ax1.set_title("Periodogram of Blizzard Sales (Train)")
ax1.set_ylim(0, 1e7)  # Set manually based on max power
plt.show(block=False)
plt.savefig("/Users/nickydesai/Desktop/DairyQueen/Charts/blizzard_periodogram.png", bbox_inches='tight', dpi=300)

#From the periodogram, looks like there is high variance at the annual and the weekly levels

#--------------------------------------------------------------------------------------------------------------------------------------

#Show how sales vary by week, and how sales vary by month

# Example: Plot weekly seasonality (day of week) across days of month
ax =seasonal_plot(y=train_timeseries_df_train['Item_Sales'], period='weekday', freq='day')
ax.set_title("Blizzard Sales Patterns Across Days of the Week (Train)")  # Your custom title
plt.show(block=False)
plt.savefig("/Users/nickydesai/Desktop/DairyQueen/Charts/blizzard_weekday_pattern.png", bbox_inches='tight', dpi=300)


# Or plot monthly seasonality across days of month
month_ax= seasonal_plot(y=train_timeseries_df_train['Item_Sales'], period='month', freq='day')
month_ax.set_title("Daily Sales Patterns by Day of Month and Month (Train)")  # Your custom title
plt.show(block=False)
plt.savefig("/Users/nickydesai/Desktop/DairyQueen/Charts/blizzard_monthly_pattern.png", bbox_inches='tight', dpi=300)

#------------------------------------------------------------------------------------------------------------------------------------



#Rolling trend line
#Can fix this if you want!!!

y = train_timeseries_df_train["Item_Sales"]

plt.figure(figsize=(14, 6))
plt.plot(train_timeseries_df_train.index, y, label="Actual Blizzard Sales", alpha=0.3)
plt.plot(y.rolling(30, center=True, min_periods=1).mean(), label="30-day Trend", color='orange', linewidth=2)
plt.plot(y.rolling(90, center=True, min_periods=1).mean(), label="90-day Trend", color='green', linewidth=2)
plt.plot(y.rolling(180, center=True, min_periods=1).mean(), label="180-day Trend", color='red', linewidth=2)
plt.title("Actual Blizzard Sales with Rolling Trend Lines (Train)")
plt.xlabel("Date")
plt.ylabel("Blizzard Sales")
plt.legend()
plt.tight_layout()
plt.show(block=False)
plt.savefig("/Users/nickydesai/Desktop/DairyQueen/Charts/blizzard_rolling_trend.png", bbox_inches='tight', dpi=300)


#--------------------------------------------------------------------------------------------------------------------------------------
#Getting the linear trend...

'''
y = train_timeseries_df["Item_Sales"]

dp = DeterministicProcess(
    index=train_timeseries_df.index,
    constant=True,
    order=1,
    drop=True
)

X_full = dp.in_sample()

model = LinearRegression()
model.fit(X_full, y)  # Fit on *all* sales, zeros included

trend = pd.Series(model.predict(X_full), index=train_timeseries_df.index)
slope = model.coef_[0]

plt.figure(figsize=(14, 6))
plt.plot(train_timeseries_df.index, y, label="Actual Sales", alpha=0.6)
plt.plot(
    train_timeseries_df.index,
    trend,
    label=f"Trend (Linear, including 0s)\nSlope: {slope:.6f} per day",
    linewidth=2,
    color='orange'
)
plt.xlabel("Date")
plt.ylabel("Blizzard Sales")
plt.title("Actual Sales vs. Linear Trend - Training Data")
plt.legend()
plt.tight_layout()
plt.show(block=False)

'''

#Here I want y_timeseries_train to be train_timeseries_df_train['Item_Sales']


y_time_series_train= train_timeseries_df_train['Item_Sales']
y_time_series_val = train_timeseries_df_val['Item_Sales']


dp = DeterministicProcess(
    index=y_time_series_train.index,
    constant=True,
    order=1,
    drop=True
)
X_train = dp.in_sample()
model = LinearRegression()
model.fit(X_train, y_time_series_train)

X_val = dp.out_of_sample(steps=len(y_time_series_val))
X_val.index = y_time_series_val.index
y_pred_val = pd.Series(model.predict(X_val), index=y_time_series_val.index)



plt.figure(figsize=(14, 6))

# Plot actual sales
plt.plot(y_time_series_train, label="Actual Blizzard Sales (Train)", color="black", alpha=0.6)
plt.plot(y_time_series_val, label="Actual Blizzard Sales (Validation)", color="gray", linestyle="-", alpha=0.6)

# Plot trend lines
plt.plot(y_time_series_train.index, model.predict(X_train), label="Fitted Blizzard Sales (Train)", color="orange", linewidth=2)
plt.plot(y_time_series_val.index, y_pred_val, label="Forecasted Blizzard Sales (Validation)", color="red", linewidth=2)

plt.title("Blizzard Sales Forecast: Trend (Train & Validation)")
plt.xlabel("Date")
plt.ylabel("Blizzard Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
plt.savefig("/Users/nickydesai/Desktop/DairyQueen/Charts/blizzard_linear_trend.png", bbox_inches='tight', dpi=300)


#---------------------------------------------------------------------------------------------------------------------------------------
#Trend line with annual and weekly seasonality

y_time_series_train= y_time_series_train.asfreq('D')
y_time_series_val= y_time_series_val.asfreq('D')

fourier = CalendarFourier(freq='YE', order=4)

dp = DeterministicProcess(
    index=y_time_series_train.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True
)

X_train = dp.in_sample()
model = LinearRegression()
model.fit(X_train, y_time_series_train)

# Predictions
y_pred_train = pd.Series(model.predict(X_train), index=y_time_series_train.index)

X_val = dp.out_of_sample(steps=len(y_time_series_val))
X_val.index = y_time_series_val.index
y_pred_val = pd.Series(model.predict(X_val), index=y_time_series_val.index)

# Plotting
plt.figure(figsize=(14, 6))

# Actual sales
plt.plot(y_time_series_train, label="Actual Blizzard Sales (Train)", color="black", alpha=0.6)
plt.plot(y_time_series_val, label="Actual Blizzard Sales (Validation)", color="gray", linestyle="-", alpha=0.6)

# Predicted trends
plt.plot(y_time_series_train.index, y_pred_train, label="Fitted Blizzard Sales (Train)", color="orange", linewidth=2)
plt.plot(y_time_series_val.index, y_pred_val, label="Forecasted Blizzard Sales (Validation)", color="red", linewidth=2)

plt.title("Blizzard Sales Forecast: Trend + Seasonality (Train & Validation))", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Blizzard Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
plt.savefig("/Users/nickydesai/Desktop/DairyQueen/Charts/blizzard_seasonality.png", bbox_inches='tight', dpi=300)

'''

#Trend line with annual and weekly seasonality

fourier = CalendarFourier(freq='YE', order=4)

dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True
)

X_full = dp.in_sample()

model = LinearRegression()
model.fit(X_full, y)  # Fit on all sales including zeros

y_pred = pd.Series(model.predict(X_full), index=y.index)

plt.figure(figsize=(15, 5))
plt.plot(y, label='Actual Sales', color='black', alpha=0.6)
plt.plot(y_pred, label='Trend + Seasonality Fit (Including Zero Sales)', color='red', linestyle='--', linewidth=2)
plt.title('Actual Sales vs. Predicted Blizzard Sales (Trend + Seasonality) - Training Data')
plt.xlabel('Date')
plt.ylabel('Blizzard Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
'''
#----------------------------------------------------------------------------------------------------------------------------------------------------

# 3. Define holidays
closed_holidays_and_weather = pd.to_datetime([
    '2022-01-01', '2022-01-29', '2022-04-17', '2022-11-24', '2022-12-25',
    '2023-01-01', '2023-11-23', '2023-12-25',
    '2024-11-28', '2024-12-25', '2022-12-13',
    '2025-12-25', '2025-11-27'
])


is_closed_train = y_time_series_train.index.to_series().isin(closed_holidays_and_weather).astype(int)
is_closed_val = y_time_series_val.index.to_series().isin(closed_holidays_and_weather).astype(int)

# 4. Define Deterministic Process (without exog)
dp = DeterministicProcess(
    index=y_time_series_train.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True
)

# 5. Create design matrix and add holiday indicator manually
X_train = dp.in_sample()
X_train["Holiday_Closed"] = is_closed_train.values

# Fit model
model = LinearRegression()
model.fit(X_train, y_time_series_train)

# 6. Predictions on training set
y_pred_train = pd.Series(model.predict(X_train), index=y_time_series_train.index)

# Create validation matrix and add holiday indicator manually
X_val = dp.out_of_sample(steps=len(y_time_series_val))
X_val.index = y_time_series_val.index
X_val["Holiday_Closed"] = is_closed_val.values

# Predictions on validation set
y_pred_val = pd.Series(model.predict(X_val), index=y_time_series_val.index)

#Clipping to prevent projected sales from being less than 0
y_pred_train = y_pred_train.clip(lower=0)
y_pred_val = y_pred_val.clip(lower=0)

y_pred_train[is_closed_train == 1] = 0
y_pred_val[is_closed_val == 1] = 0

# 7. Plotting
plt.figure(figsize=(14, 6))

# Actual sales
plt.plot(y_time_series_train, label="Actual Blizzard Sales (Train)", color="black", alpha=0.6)
plt.plot(y_time_series_val, label="Actual Blizzard Sales (Validation)", color="gray", linestyle="-", alpha=0.6)

# Predicted trends
plt.plot(y_time_series_train.index, y_pred_train, label="Fitted Blizzard Sales (Train)", color="orange", linewidth=2)
plt.plot(y_time_series_val.index, y_pred_val, label="Forecasted Blizzard Sales (Validation)", color="red", linewidth=2)

plt.title("Blizzard Sales Forecast: Trend + Seasonality + Closures (Train & Validation)", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Blizzard Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
plt.savefig("/Users/nickydesai/Desktop/DairyQueen/Charts/blizzard_closed_days.png", bbox_inches='tight', dpi=300)

#------------------------------------------------------------------------------------------------------------------------------------------------
#A bit of digging around to see if there are weird sales days that weren't tehcnically closure days...

# 1. Combine training and validation sets
y_full = pd.concat([y_time_series_train, y_time_series_val])
is_closed_full = pd.concat([is_closed_train, is_closed_val])
is_closed = is_closed_full.loc[y_full.index]

# 2. Only keep open days
y_open = y_full[is_closed == 0]

# 3. Compute z-scores
mean_sales = y_open.mean()
std_sales = y_open.std()
z_scores = (y_open - mean_sales) / std_sales

# 4. Set threshold for abnormal (e.g., z < -2 or z > 2)
abnormal_low = y_open[z_scores < -2]
abnormal_high = y_open[z_scores > 2]

# 5. Print abnormal low/high sales days
print("Abnormally LOW sales days (store was open):")
for date, sales in abnormal_low.items():
    print(f"{date.date()}: {sales:.2f}")

print("\nAbnormally HIGH sales days (store was open):")
for date, sales in abnormal_high.items():
    print(f"{date.date()}: {sales:.2f}")


#-----------------------------------------------------------------------------------------------------------------------------------------------

# 1. Define High-Sales Holidays (store is open)
high_sales_holidays = pd.to_datetime([
    # Mother's Day (2nd Sunday of May)
    '2022-05-08', '2023-05-14', '2024-05-12', '2025-05-11', '2026-05-10',
    # Memorial Day (last Monday of May)
    '2022-05-30', '2023-05-29', '2024-05-27', '2025-05-26', '2026-05-25',
    # Independence Day
    '2022-07-04', '2023-07-04', '2024-07-04', '2025-07-04', '2026-07-04',
    # Father's Day (3rd Sunday of June)
    '2022-06-19', '2023-06-18', '2024-06-16', '2025-06-15', '2026-06-21',
    # Easter Sundays (except 2022, which is a closed day)
    '2023-04-09', '2024-03-31', '2025-04-20', '2026-04-05',
    # Juneteenth (June 19)
    '2022-06-19', '2023-06-19', '2024-06-19', '2025-06-19', '2026-06-19'
])

# 2. Define Low-Sales Holidays (store is open)
low_sales_holidays = pd.to_datetime([
    # Christmas Eve
    '2022-12-24', '2023-12-24', '2024-12-24', '2025-12-24',
    # Post–New Year's slump
    '2022-01-02', '2023-01-02', '2024-01-02', '2025-01-02', '2026-01-02',
    '2022-01-03', '2023-01-03', '2024-01-03', '2025-01-03', '2026-01-03'
])

# 3. Create indicator variables for the full combined index
combined_index = pd.concat([y_time_series_train, y_time_series_val, test_timeseries_df]).index

is_closed_full = combined_index.to_series().isin(closed_holidays_and_weather).astype(int)
is_high_sales_full = combined_index.to_series().isin(high_sales_holidays).astype(int)
is_low_sales_full = combined_index.to_series().isin(low_sales_holidays).astype(int)

# 4. Slice indicators to match train and validation sets
is_closed_train = is_closed_full.loc[y_time_series_train.index]
is_closed_val = is_closed_full.loc[y_time_series_val.index]
is_closed_test = is_closed_full.loc[test_timeseries_df.index]

is_high_sales_train = is_high_sales_full.loc[y_time_series_train.index]
is_high_sales_val = is_high_sales_full.loc[y_time_series_val.index]
is_high_sales_test = is_high_sales_full.loc[test_timeseries_df.index]

is_low_sales_train = is_low_sales_full.loc[y_time_series_train.index]
is_low_sales_val = is_low_sales_full.loc[y_time_series_val.index]
is_low_sales_test = is_low_sales_full.loc[test_timeseries_df.index]

# 5. Define Deterministic Process
dp = DeterministicProcess(
    index=combined_index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True
)

# 6. Create design matrices and add holiday indicators
X_all= dp.in_sample()

X_train = X_all.loc[y_time_series_train.index].copy()
X_train["Holiday_Closed"] = is_closed_train.values
X_train["High_Sales_Holiday"] = is_high_sales_train.values
X_train["Low_Sales_Holiday"] = is_low_sales_train.values

# Fit model
lin_model = LinearRegression()
lin_model.fit(X_train, y_time_series_train)

# Predict on train
y_pred_train = pd.Series(lin_model.predict(X_train), index=y_time_series_train.index)

# Validation
X_val = X_all.loc[y_time_series_val.index].copy()
X_val["Holiday_Closed"] = is_closed_val.values
X_val["High_Sales_Holiday"] = is_high_sales_val.values
X_val["Low_Sales_Holiday"] = is_low_sales_val.values

y_pred_val = pd.Series(lin_model.predict(X_val), index=y_time_series_val.index)

#Test Data

X_test = X_all.loc[test_timeseries_df.index].copy()
X_test["Holiday_Closed"] = is_closed_test.values
X_test["High_Sales_Holiday"] = is_high_sales_test.values
X_test["Low_Sales_Holiday"] = is_low_sales_test.values

y_pred_test = pd.Series(lin_model.predict(X_test), index=test_timeseries_df.index)

# 7. Post-process predictions
y_pred_train = y_pred_train.clip(lower=0)
y_pred_val = y_pred_val.clip(lower=0)
y_pred_test = y_pred_test.clip(lower=0)

# Zero-out predictions on closed days
y_pred_train[is_closed_train == 1] = 0
y_pred_val[is_closed_val == 1] = 0
y_pred_test[is_closed_test == 1] = 0 

# 8. Plotting
plt.figure(figsize=(14, 6))

# Actual sales
plt.plot(y_time_series_train, label="Actual Blizzard Sales (Train)", color="black", alpha=0.6)
plt.plot(y_time_series_val, label="Actual Blizzard Sales (Validation)", color="gray", linestyle="-", alpha=0.6)

# Predicted
plt.plot(y_time_series_train.index, y_pred_train, label="Fitted Blizzard Sales (Train)", color="orange", linewidth=2)
plt.plot(y_time_series_val.index, y_pred_val, label="Forecasted Blizzard Sales (Validation)", color="red", linewidth=2)

plt.title("Blizzard Sales Forecast: Trend + Seasonality + Closures + Holiday Effects (Train & Validation)", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Blizzard Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
plt.savefig("/Users/nickydesai/Desktop/DairyQueen/Charts/blizzard_holiday_effects.png", bbox_inches='tight', dpi=300)




'''
#Now we want to handle holidays and handle days of closure due to snow days
#This will be model_1

# Define holidays
closed_holidays = pd.to_datetime([
    '2022-01-01', '2022-04-17', '2022-11-24', '2022-12-25',
    '2023-01-01', '2023-11-23', '2023-12-25',
    '2024-11-28', '2024-12-25', '2022-01-29', '2022-12-13', '2025-12-25', '2025-11-27'
])

# Create dummy for closed days
is_closed = y.index.isin(closed_holidays).astype(int)

# Create deterministic features


y = y.asfreq('D')
fourier = CalendarFourier(freq='YE', order=4)
dp_correct = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True
)

X_full = dp_correct.in_sample()
X_full['is_closed'] = is_closed

# Fit the model
model1 = LinearRegression()
model1.fit(X_full, y)

# Predict and clip negative predictions to 0
y_pred = pd.Series(model1.predict(X_full), index=y.index)
y_pred = y_pred.clip(lower=0)

# Plot
plt.figure(figsize=(15, 5))
plt.plot(y, label='Actual Sales', color='black', alpha=0.6)
plt.plot(y_pred, label='Trend + Seasonality + Closure Dummies', color='red', linestyle='--', linewidth=2)
plt.title('Actual Sales vs. Predicted Blizzard Sales (Trend + Seasonality + Closure Dates) - Training Data')
plt.xlabel('Date')
plt.ylabel('Blizzard Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
'''
#-----------------------------------------------------------------------------------------------------------------------------------------
#I think that I need to add the train predictions back to the train_df_train dataset and the validation predictions to the train_df_val dataset
# Make sure the indexes match
train_df_train = train_df_train.copy()
train_df_val = train_df_val.copy()

train_df_train["LinReg_pred_train"] = pd.Series(y_pred_train, index=train_df_train.index)
train_df_val["LinReg_pred_val"] = pd.Series(y_pred_val, index=train_df_val.index)
test_df['LinReg_pred_test'] = pd.Series(y_pred_test, index=test_df.index)

print(train_df_train.head())
print(train_df_val.head())

'''
# Add training predictions to train_df_train
train_df_train = train_df_train.copy()  # to avoid modifying the original if needed
train_df_train["Predicted_Item_Sales"] = y_pred_train

# Add validation predictions to train_df_val
train_df_val = train_df_val.copy()
train_df_val["Predicted_Item_Sales"] = y_pred_val


print(train_df_train.head())
print(train_df_val.head())
'''
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#Do This LATER>>>>>>>>>>>>>>>>>>>
'''
#Using the linear regression model to predict the item sales for the test data...
test_index = pd.to_datetime(test_timeseries_df.index)

# Create test feature matrix
X_test = dp.out_of_sample(steps=len(test_index))
X_test.index = test_index

# Add holiday dummies
X_test['is_closed'] = test_index.isin(closed_holidays_and_weather).astype(int)
X_test['is_high_sales_holiday'] = test_index.isin(high_sales_holidays).astype(int)
X_test['is_low_sales_holiday'] = test_index.isin(low_sales_holidays).astype(int)

# Predict test sales
y_test_pred = pd.Series(model.predict(X_test), index=test_index).clip(lower=0)

# Store predictions in test_timeseries_df
test_timeseries_df['lin_forecast_sales'] = y_test_pred.values

# === Plotting ===
plt.figure(figsize=(15, 5))

# Plot train actual sales
plt.plot(y.index, y, label='Train Actual Sales', color='black', alpha=0.6)

# Plot train predictions
plt.plot(y_pred.index, y_pred, label='Train Predictions (Linear Model)', color='red', linewidth=2, linestyle='--')

# Plot test predictions
plt.plot(y_test_pred.index, y_test_pred, label='Test Predictions (Linear Model)', color='blue', linewidth=2, linestyle='--')

# Final formatting
plt.title('Train Actual vs. Train & Test Blizzard Sale Predictions (Linear Regression)')
plt.xlabel('Date')
plt.ylabel('Blizzard Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)




test_df['y_test_pred']= y_test_pred
'''
#------------------------------------------------------------------------------------------------------------------------------------------------
#Let's now take a look at y_deseason for the training data 

# Assuming train_df_train already contains these columns:
# 'Item_Sales' (actual values) and 'LinReg_pred_train' (predicted seasonal trend)

print(train_df_train.head())

# Compute deseasonalized sales
y_deseason_train = train_df_train['Item_Sales'] - train_df_train['LinReg_pred_train']
y_deseason_val = train_df_val['Item_Sales'] - train_df_val['LinReg_pred_val']

# For debugging / verification
print("Item_Sales shape:", train_df_train['Item_Sales'].shape)
print("LinReg_pred_train shape:", train_df_train['LinReg_pred_train'].shape)
print("y_deseason_train shape:", y_deseason_train.shape)

# Create DataFrame to inspect values
df_check = pd.DataFrame({
    'Actual': train_df_train['Item_Sales'],
    'Predicted (Seasonal)': train_df_train['LinReg_pred_train'],
    'Deseasonalized': y_deseason_train
})

print(df_check.head(10))
print(df_check)

# Plot deseasonalized values
plot_params = {
    'figsize': (10, 4),
    'title': 'Deseasonalized Blizzard Sales (Train)',
    'ylabel': 'Deseasoned Blizzard Sales',
    'xlabel': 'Date'
}

plt.figure(figsize=plot_params['figsize'])
ax = y_deseason_train.plot(color='black', alpha=0.6)
ax.set_title(plot_params['title'])
ax.set_ylabel(plot_params['ylabel'])
ax.set_xlabel(plot_params['xlabel'])
plt.tight_layout()
plt.show(block=False)
plt.savefig("/Users/nickydesai/Desktop/DairyQueen/Charts/blizzard_deseasoned.png", bbox_inches='tight', dpi=300)

# Periodogram to analyze seasonality in residuals
fig, ax2 = plt.subplots()
plot_periodogram(y_deseason_train, ax=ax2)
ax2.set_title("Periodogram of Deseasoned Blizzard Sales (Train)")
ax2.set_ylim(0, 1e7)
plt.show(block=False)
plt.savefig("/Users/nickydesai/Desktop/DairyQueen/Charts/blizzard_periodogram_residuals.png", bbox_inches='tight', dpi=300)


#-------------------------------------------------------------------------------------------------------------------
#We now need to add y_deseason to the train_df dataset, such that the y_deseason fits the index of the train dataset

# ------------------------------------------------------------
# Attach target
train_df_train["Sales_deseasoned_train"] = y_deseason_train
train_df_val["Sales_deseasoned_val"] = y_deseason_val

# ------------------------------------------------------------
# Combine training and validation sets for custom CV split
train_df_train["split"] = -1  # Training split
train_df_val["split"] = 0     # Validation split

combined_df = pd.concat([train_df_train, train_df_val])

# ------------------------------------------------------------
# Prepare features and target
feature_cols_to_drop = ["Sales_deseasoned_train", "Sales_deseasoned_val", "Item_Sales", "LinReg_pred_train", "LinReg_pred_val", "split"]
X_all = combined_df.drop(columns=[col for col in feature_cols_to_drop if col in combined_df.columns])
y_all = combined_df["Sales_deseasoned_train"].fillna(combined_df["Sales_deseasoned_val"])

# Custom CV split
ps = PredefinedSplit(test_fold=combined_df["split"])

# ------------------------------------------------------------
# Hyperparameter space
param_dist = {
    "n_estimators": [100, 300, 500, 700, 900, 1100, 1300],
    "max_depth": [2, 3, 4, 5, 6, 7],
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
    "gamma": [0, 0.5, 1, 2, 3],
    "min_child_weight": [1, 2, 3, 4, 5, 6],
    "reg_alpha": [0, 0.1, 0.5, 1, 2],
    "reg_lambda": [0.01, 0.1, 0.5, 1, 2],
    "scale_pos_weight": [1, 2, 5],
    "max_delta_step": [0, 1, 5],
    "tree_method": ["hist", "exact"],
    "grow_policy": ["depthwise", "lossguide"]
}

# ------------------------------------------------------------
# Model and randomized search
xgb = XGBRegressor(objective="reg:squarederror", random_state=42)

search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=30,
    scoring="neg_mean_squared_error",
    cv=ps,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_all, y_all)

print(f"\n✅ Best Hyperparameters:\n{search.best_params_}")
print(f"🔍 Best Validation MSE: {-search.best_score_:.4f}")

# ------------------------------------------------------------
# Retrain model using best parameters on train_df_train
best_model = search.best_estimator_

X_train = X_all[combined_df["split"] == -1]
X_val = X_all[combined_df["split"] == 0]
y_train = y_all[combined_df["split"] == -1]
y_val = y_all[combined_df["split"] == 0]

# Fit on training
best_model.fit(X_train, y_train)

# ------------------------------------------------------------
# Store predictions
train_df_train["XGB_pred_train"] = best_model.predict(X_train)
train_df_val["XGB_pred_val"] = best_model.predict(X_val)

# Override predictions where LinReg is 0
train_df_train.loc[train_df_train["LinReg_pred_train"] == 0, "XGB_pred_train"] = 0
train_df_val.loc[train_df_val["LinReg_pred_val"] == 0, "XGB_pred_val"] = 0

# ------------------------------------------------------------
# Predict on test_df
feature_cols_to_drop_test = ["Item_Sales", "LinReg_pred_test"]
X_test = test_df.drop(columns=[col for col in feature_cols_to_drop_test if col in test_df.columns])

test_df["XGB_pred_test"] = best_model.predict(X_test)

# Zero out if LinReg predicted 0
if "LinReg_pred_test" in test_df.columns:
    test_df.loc[test_df["LinReg_pred_test"] == 0, "XGB_pred_test"] = 0

# Final hybrid forecast
test_df["official_prediction_test"] = test_df["LinReg_pred_test"] + test_df["XGB_pred_test"]
test_df.loc[test_df["LinReg_pred_test"] == 0, "official_prediction_test"] = 0
test_df["official_prediction_test"] = test_df["official_prediction_test"].clip(lower=0)

# ------------------------------------------------------------
# Plot actual vs predicted
plt.figure(figsize=(14, 6))

# Training
plt.plot(train_df_train.index, train_df_train["Sales_deseasoned_train"], label="Deseasoned Blizzard Sales (Train)", color="black", linestyle='-', alpha=0.6)
plt.plot(train_df_train.index, train_df_train["XGB_pred_train"], label="Fitted Deseasoned Blizzard Sales (Train)", color="orange", linestyle='-')

# Validation
plt.plot(train_df_val.index, train_df_val["Sales_deseasoned_val"], label="Deseasoned Blizzard Sales (Validation)", color="gray", linestyle='-', alpha=0.6)
plt.plot(train_df_val.index, train_df_val["XGB_pred_val"], label="Forecasted Deseasoned Blizzard Sales (Validation)", color="red", linestyle='-')

plt.title("Deseasonalized Blizzard Sales Forecast: Train & Validation")
plt.xlabel("Date")
plt.ylabel("Deseasoned Blizzard Sales")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show(block=False)
plt.savefig("/Users/nickydesai/Desktop/DairyQueen/Charts/blizzard_xgboost_residuals.png", bbox_inches='tight', dpi=300)
#-----------------------------------------------------------------------------------------------------------------------------------------------
print(train_df_train.head())
print(train_df_val.head())
print(test_df.head(20))

train_df_train['official_prediction_train']= train_df_train['LinReg_pred_train'] + train_df_train['XGB_pred_train']
train_df_val['official_prediction_val']= train_df_val['LinReg_pred_val'] + train_df_val['XGB_pred_val']


plt.figure(figsize=(15, 6))

# Plot actual sales
plt.plot(train_df_train.index, train_df_train['Item_Sales'], label='Actual Blizzard Sales (Train)', color='black', alpha=0.6)
plt.plot(train_df_train.index, train_df_train['official_prediction_train'], label='Fitted Blizzard Sales (Train)', color='orange')

plt.plot(train_df_val.index, train_df_val['Item_Sales'], label='Actual Blizzard Sales (Validation)', color='gray', alpha=0.6)
plt.plot(train_df_val.index, train_df_val['official_prediction_val'], label='Forecasted Blizzard Sales (Validation)', color='red')

plt.title('Blizzard Sales Forecast: LinReg + XGBoost (Train & Validation)')
plt.xlabel('Date')
plt.ylabel('Item Sales')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show(block=False)
plt.savefig("/Users/nickydesai/Desktop/DairyQueen/Charts/blizzard_hybrid.png", bbox_inches='tight', dpi=300)

#----------------------------------------------------------------------------------------------------------------------

#Now I want a plot of item sales for train, official_prediction_train, item sales for val, offical predictions val, and official prediction test all together

plt.figure(figsize=(18, 6))

# Plot item sales for training data
plt.plot(train_df_train.index, train_df_train["Item_Sales"], label="Actual Blizzard Sales (Train)", color="black", alpha=0.6)
plt.plot(train_df_train.index, train_df_train["official_prediction_train"], label="Fitted Blizzard Sales (Train)", color="orange")

# Plot item sales for validation data
plt.plot(train_df_val.index, train_df_val["Item_Sales"], label="Actual Blizzard Sales (Validation)", color="gray")
plt.plot(train_df_val.index, train_df_val["official_prediction_val"], label="Forecasted Blizzard Sales (Validation)", color="red")

# Plot test predictions (no actual sales available here)
plt.plot(test_df.index, test_df["official_prediction_test"], label="Forecasted Blizzard Sales (Test)", color="green")

plt.title("Blizzard Sales Forecast: LinReg + XGBoost (Train & Validation & Test)")
plt.xlabel("Date")
plt.ylabel("Deseasonalized Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
plt.savefig("/Users/nickydesai/Desktop/DairyQueen/Charts/blizzard_test_results.png", bbox_inches='tight', dpi=300)




'''
# Drop columns and split features/target
feature_cols_to_drop = ["Sales_deseasoned", "Item_Sales", "LinReg_pred_train"]
X = train_df_train.drop(columns=feature_cols_to_drop)
y = train_df_train["Sales_deseasoned"]

# ------------------------------------------------------------
# Split into train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------
# Grid of n_estimators to try
estimator_grid = [100, 200, 300, 400, 500, 1000]
val_errors = {}

# ------------------------------------------------------------
# Loop through and evaluate each
for n in estimator_grid:
    model = XGBRegressor(
        n_estimators=n,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=2,
        min_child_weight=5,
        reg_lambda=1,
        reg_alpha=0.5,
        random_state=42,
        objective="reg:squarederror"
    )
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    val_errors[n] = mse
    print(f"n_estimators = {n} → Validation MSE: {mse:.4f}")

# ------------------------------------------------------------
# Select best n_estimators
best_n = min(val_errors, key=val_errors.get)
print(f"\n✅ Best n_estimators: {best_n} with MSE: {val_errors[best_n]:.4f}")

# ------------------------------------------------------------
# Retrain full model using best n_estimators
best_model = XGBRegressor(
    n_estimators=best_n,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=2,
    min_child_weight=5,
    reg_lambda=1,
    reg_alpha=0.5,
    random_state=42,
    objective="reg:squarederror"
)

best_model.fit(X, y)
'''

'''
# ------------------------------------------------------------
# Final predictions on training data
train_df['y_deseasoned_residual'] = best_model.predict(X)
train_df['official_pred'] = train_df['y_pred'] + train_df['y_deseasoned_residual']


#---------------------------------------------------------------------------------------------------------------------------------------------------------
#Now I want a graph of item sales against the official predictions

plt.figure(figsize=(15, 5))
plt.plot(train_df['Item_Sales'], label='Actual Sales', color='black', alpha=0.6)
plt.plot(train_df["official_pred"], label='Official Predicted Sales', color='red', linestyle='--', linewidth=2)

plt.title('Actual Sales vs. Official Predicted Blizzard Sales - Training Data')
plt.xlabel('Date')
plt.ylabel('Blizzard Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#I can also determine the MSE of the predictions agaisnt actual sales
#I want to do a deseasoned graph now, but show the official predications minus the actual item_sales
# Define styling parameters
# Define plotting parameters
# Define consistent plot parameters
plot_params = {
    'figsize': (10, 4),
    'title': 'Prediction Residuals: Official Predicted - Actual Blizzard Sales (Deseasoned)',
    'ylabel': 'Prediction Residuals',
    'xlabel': 'Date'
}

# Compute residuals
residuals = train_df["official_pred"] - train_df["Item_Sales"]

# Create figure
plt.figure(figsize=plot_params['figsize'])

# Plot residuals using default matplotlib blue and line width
ax = residuals.plot()  # Don't specify color or linewidth to match defaults

# Match y-axis scale to y_deseason
ax.set_ylim(y_deseason.min(), y_deseason.max())

# Set labels and title
ax.set_title(plot_params['title'])
ax.set_ylabel(plot_params['ylabel'])
ax.set_xlabel(plot_params['xlabel'])

plt.tight_layout()
plt.show(block=False)



#Calculating the MSE:
# Actual vs Predicted

# Calculate MSE
mse = mean_squared_error(train_df["Item_Sales"], train_df["official_pred"])

print(f"Mean Squared Error (MSE): {mse:.2f}")


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#Now do the test predictions for the the y_deseasoned_residual for the test data... and also substract the y_preds from the y_deseasoned_residual to obstain the offical preds


# -----------------------------------------------
# Step 1: Drop leakage columns from test set
# -----------------------------------------------
test_df = test_df.copy()
test_feature_cols_to_drop = ["Item_Sales", "y_test_pred"]

# Use same feature columns as in training
X_test = test_df.drop(columns=test_feature_cols_to_drop)

# -----------------------------------------------
# Step 2: Predict residuals using best_model
# -----------------------------------------------
test_df['y_test_deseasoned_residual'] = best_model.predict(X_test)

# -----------------------------------------------
# Step 3: Combine with seasonal predictions
# -----------------------------------------------
test_df['official_pred'] = test_df['y_test_pred'] + test_df['y_test_deseasoned_residual']

# -----------------------------------------------
# Step 4: Preview results
# -----------------------------------------------
print(test_df[['y_test_pred', 'y_test_deseasoned_residual', 'official_pred']].head(10))


#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#Now graph the official preds for the test with the graph earlier

plt.figure(figsize=(15, 5))

# Plot actual sales from train data
plt.plot(train_df['Item_Sales'], label='Actual Sales (Train)', color='black', alpha=0.6)

# Plot official predictions from train data
plt.plot(train_df["official_pred"], label='Predicted Sales (Train)', color='red', linestyle='--', linewidth=2)

# Plot official predictions from test data
plt.plot(test_df["official_pred"], label='Predicted Sales (Test)', color='blue', linestyle='--', linewidth=2)

# Titles and labels
plt.title('Actual vs. Predicted Blizzard Sales (Train & Test)')
plt.xlabel('Date')
plt.ylabel('Blizzard Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#Trying to understand why the test residuals might be way off, or if that's the case

plt.figure(figsize=(8, 4))
plt.hist(train_df['y_deseasoned_residual'], bins=30, color='red', edgecolor='black', alpha=0.7)
plt.title("Distribution of Predicted Residuals (Train)", fontsize=14)
plt.xlabel("Residual Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# ----------------------------
# Plot: Test Residuals
# ----------------------------
plt.figure(figsize=(8, 4))
plt.hist(test_df['y_test_deseasoned_residual'], bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title("Distribution of Predicted Residuals (Test)", fontsize=14)
plt.xlabel("Residual Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
#------------------------------------------------------------------------------------------------------------------------------------------------

for col in X.columns:
    plt.figure(figsize=(8, 5))           # create a new figure each iteration
    sns.scatterplot(data=test_df, x=col, y='y_test_deseasoned_residual')
    plt.title(f'Residuals vs. {col}')
    plt.axhline(0, linestyle='--', color='gray')
    plt.xlabel(col)
    plt.ylabel('Test Residuals')
    plt.tight_layout()
    plt.show()





test_df.to_csv("/Users/nickydesai/Desktop/DairyQueen/test_predictions.csv", index=True)


'''
