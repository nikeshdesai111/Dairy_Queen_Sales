# Dairy_Queen_Sales  
**Forecasting Blizzard Sales with a Hybrid Machine Learning Model**

---

## Project Overview  
This project aims to forecast daily Blizzard sales at the **Dairy Queen Grill & Chill** location in Worcester, MA, using a hybrid machine learning model.  

- **Training Period:** January 1, 2022 – June 30, 2025  
- **Validation Period:** January 1, 2025 – June 30, 2025  
- **Forecasting Period:** July 1, 2025 – August 1, 2026  

Each observation represents the total number of Blizzard sales per day, aggregated across all sizes (small, medium, and large).  

### Key Features Used:
- Average Blizzard Price  
- Daily Average, Low, and High Temperatures  
- Precipitation Amount  
- Number of Store Operating Hours  

A **hybrid model** combining **Linear Regression** (for trend and seasonality) and **XGBoost** (to model residuals) was used to produce accurate forecasts.

---

## Objective  
The main goals of the project were to:  
- Accurately forecast daily Blizzard sales  
- Identify and model patterns such as weekday effects, holidays, closures, and seasonality  
- Provide the store owner with a data-driven tool to improve inventory planning and reduce reliance on guesswork  

Since no prior forecasting model existed for this store, this work is a first-of-its-kind and critical for data-informed business decisions.

---

## Methodology  

### 1. **Data Collection**  
Sales data was obtained from Dairy Queen's internal **ParBrink** system, requiring authorized access.  
For each month in the training period, sales by item type were downloaded individually. These files are stored in the repository inside the `Sales_by_year.zip` archive.

### 2. **Data Processing**  
Custom Python scripts were written to extract and format Blizzard sales data:  
- `DQCollection_2022.py`, `DQCollection_2023.py`, etc.: Extract Blizzard sales (small, medium, large), group by date  
- `Concatenated_File.py`: Combines all yearly sales data into a master file `all_years_blizzard_sales.csv`  

### 3. **Forecasting Period Setup**  
A forecasting template (`Forecasting_Future.csv`) was created in Excel with future dates and weather predictions scraped from [Weather Underground](https://www.wunderground.com/history/daily/us/ma/worcester).  

### 4. **Master Dataset Construction**  
Two scripts finalize the data engineering pipeline:  
- `Prepping_Master_Data_Part1.py`: Merges `all_years_blizzard_sales.csv` with `Forecasting_Future.csv`  
- `Prepping_Master_Data_Part2.py`: Adds historical weather data from "weather_data.csv" and outputs `Combined_dataset_with_weather.csv`  

This final dataset is the input for modeling.

---

## Modeling Approach  

Modeling is done in `Time_Series_Model.py`, which walks through each step with detailed inline comments. A separate written report also accompanies the code, explaining each modeling choice and visual output.

### Hybrid Model Components:
- **Linear Regression**  
  - Captures long-term trends using a time-based trend variable  
  - Models seasonality using **Fourier Transforms**  
  - Incorporates dummy variables for holidays, closures, and special effects  
- **XGBoost Regressor**  
  - Trained on the residuals from the linear model  
  - Uses features like weather, pricing, and operating hours  
  - Refines the forecast by capturing non-linear patterns and interactions  

Final predictions for the test period are visualized and assessed for accuracy and business insight.

---

## File Structure (as it exists on my local drive)

Folders labeled 2022-2025 and the csv files inside them can be found within the "Sales_by_year" zip file.

```plaintext
Dairy_Queen/
├── 2022/
│   ├── Jan22.csv
│   ├── Feb22.csv
│   ├── Mar22.csv
│   ├── Apr22.csv
│   ├── May22.csv
│   ├── Jun22.csv
│   ├── Jul22.csv
│   ├── Aug22.csv
│   ├── Sep22.csv
│   ├── Oct22.csv
│   ├── Nov22.csv
│   ├── Dec22.csv
├── 2023/
│   ├── Jan23.csv
│   ├── Feb23.csv
│   ├── Mar23.csv
│   ├── Apr23.csv
│   ├── May23.csv
│   ├── Jun23.csv
│   ├── Jul23.csv
│   ├── Aug23.csv
│   ├── Sep23.csv
│   ├── Oct23.csv
│   ├── Nov23.csv
│   ├── Dec23.csv
├── 2024/
│   ├── Jan24.csv
│   ├── Feb24.csv
│   ├── Mar24.csv
│   ├── Apr24.csv
│   ├── May24.csv
│   ├── Jun24.csv
│   ├── Jul24.csv
│   ├── Aug24.csv
│   ├── Sep24.csv
│   ├── Oct24.csv
│   ├── Nov24.csv
│   ├── Dec24.csv
├── 2025/
│   ├── Jan25.csv
│   ├── Feb25.csv
│   ├── Mar25.csv
│   ├── Apr25.csv
│   ├── May25.csv
│   ├── Jun25.csv
├── python_files/
│   ├── DQCollection_2022.py
│   ├── DQCollection_2023.py
│   ├── DQCollection_2024.py
│   ├── DQCollection_2025.py
│   ├── Concatenated_File.py
│   ├── Prepping_Master_Data_Part1.py
│   ├── Prepping_Master_Data_Part2.py
│   └── Time_Series_Model.py
├── all_years_blizzard_sales.csv
├── Forecasting_Future.csv
├── Combined_Dataset.csv
├── Combined_dataset_with_weather.csv
├── Weather_data.csv
├── README.md
```

## Tech Stack

### Data Manipulation & File Handling
- [**pandas**](https://pandas.pydata.org/) – Powerful data structures for data analysis and manipulation.
- [**os**](https://docs.python.org/3/library/os.html) – Accessing the file system to read/write data and manage file paths.

### Data Visualization
- [**matplotlib**](https://matplotlib.org/) – Core Python library for creating static and interactive visualizations.
- [**seaborn**](https://seaborn.pydata.org/) – Statistical data visualization built on top of Matplotlib for enhanced graphics.

### Numerical Computing
- [**numpy**](https://numpy.org/) – Core library for numerical and array operations in Python.

### Machine Learning & Model Evaluation
- [**scikit-learn**](https://scikit-learn.org/) – Widely-used ML library offering tools for:
  - Linear regression (`LinearRegression`)
  - Feature scaling (`StandardScaler`)
  - Train/test splits and validation (`train_test_split`, `PredefinedSplit`)
  - Model evaluation (`mean_squared_error`)
  - Hyperparameter tuning (`GridSearchCV`, `RandomizedSearchCV`)
- [**XGBoost**](https://xgboost.readthedocs.io/) – Efficient and scalable gradient boosting library, used here via `XGBRegressor`.

### Time Series Modeling
- [**statsmodels**](https://www.statsmodels.org/) – Used for time series decomposition and deterministic trend modeling with:
  - `DeterministicProcess`
  - `CalendarFourier` (Fourier terms for seasonal patterns)













