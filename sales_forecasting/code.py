import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# ==========================================
# 1. DATA GENERATION / LOADING
# ==========================================
# Generates realistic synthetic multi-store daily sales time-series data
# (If you have 'train.csv' from Kaggle Rossmann dataset, use: df = pd.read_csv('train.csv'))

np.random.seed(42)
dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
stores = [1, 2, 3, 4, 5]

data_list = []
for store_id in stores:
    base_sales = np.random.randint(3000, 7000)
    for date in dates:
        # Seasonality factors: Day of week and Month
        day_effect = 1.3 if date.dayofweek in [4, 5] else 1.0  # Higher on Fri/Sat
        month_effect = 1.4 if date.month in [11, 12] else 1.0  # Holiday shopping season
        promo = np.random.choice([0, 1], p=[0.7, 0.3])
        promo_effect = 1.25 if promo == 1 else 1.0
        
        noise = np.random.normal(0, 300)
        sales = int((base_sales * day_effect * month_effect * promo_effect) + noise)
        
        data_list.append({
            'Date': date,
            'Store': store_id,
            'Promo': promo,
            'SchoolHoliday': np.random.choice([0, 1], p=[0.8, 0.2]),
            'Sales': max(0, sales)
        })

df = pd.DataFrame(data_list)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by=['Store', 'Date'], inplace=True)

print("Dataset Preview:")
print(df.head())

# ==========================================
# 2. FEATURE ENGINEERING (TIME-SERIES)
# ==========================================
# Extract Date/Calendar Features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

# Lag Features (Past Sales)
for lag in [1, 7, 14, 30]:
    df[f'Sales_Lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)

# Rolling Window Features (Moving Averages & Standard Deviation)
for window in [7, 14, 30]:
    df[f'Sales_Rolling_Mean_{window}'] = (
        df.groupby('Store')['Sales']
        .transform(lambda x: x.shift(1).rolling(window=window).mean())
    )
    df[f'Sales_Rolling_Std_{window}'] = (
        df.groupby('Store')['Sales']
        .transform(lambda x: x.shift(1).rolling(window=window).std())
    )

# Drop rows with NaNs created by lag/rolling features
df.dropna(inplace=True)

# ==========================================
# 3. TIME-SERIES TRAIN-TEST SPLIT
# ==========================================
# Note: For time-series, DO NOT use random split. Use temporal split.
split_date = '2023-07-01'

train_df = df[df['Date'] < split_date]
test_df = df[df['Date'] >= split_date]

features = [col for col in df.columns if col not in ['Date', 'Sales']]
target = 'Sales'

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# ==========================================
# 4. MODEL TRAINING & REGRESSION
# ==========================================
rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# ==========================================
# 5. TIME-SERIES EVALUATION METRICS
# ==========================================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
rmspe = np.sqrt(np.mean(np.square((y_test - y_pred) / y_test))) * 100

print("\n================ TIME-SERIES EVALUATION METRICS ================")
print(f"Root Mean Squared Error (RMSE) : ${rmse:.2f}")
print(f"Mean Absolute Error (MAE)       : ${mae:.2f}")
print(f"Mean Absolute Percentage Error : {mape:.2f}%")
print(f"Root Mean Sq. Percentage Error  : {rmspe:.2f}%")

# ==========================================
# 6. VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 5))
sample_store = test_df[test_df['Store'] == 1]
sample_preds = y_pred[X_test['Store'] == 1]

plt.plot(sample_store['Date'], sample_store['Sales'], label='Actual Sales', color='blue', alpha=0.7)
plt.plot(sample_store['Date'], sample_preds, label='Forecasted Sales', color='orange', linestyle='--')
plt.title('Actual vs Forecasted Daily Sales (Store 1)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()