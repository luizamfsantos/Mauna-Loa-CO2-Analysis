# load data
import pandas as pd
path = 'data/processed/CO2_clean.csv'
df = pd.read_csv(path)

# split data
from src.preprocessing import test_train_split
train_df, test_df = test_train_split(df)

# separate features and labels
X_train = train_df[['t']]
y_train = train_df[['CO2_concentration']]

# load parameters for quadratic model
import json
import numpy as np
with open('models/trend_modeling/quadratic_model.json', 'r') as file:
    model_info = json.load(file)
coef = np.array(model_info['coefficients'])
intercept = model_info['intercept']
degree = model_info['order']

# create polynomial model
coef_with_intercept = np.append(coef, intercept)
predict = np.poly1d(coef_with_intercept)

# use polynomial model to predict CO2 concentration
y_pred = predict(X_train)
test_y_pred = predict(test_df[['t']])

# calculate linear residuals
quadratic_residuals = y_train - y_pred
test_quadratic_residuals = test_df['CO2_concentration'] - test_y_pred.flatten()

# calculate rmse and mape
from src.utils import calculate_rmse, calculate_mape
rmse = calculate_rmse(test_df['CO2_concentration'], test_y_pred.flatten())
mape = calculate_mape(test_df['CO2_concentration'], test_y_pred.flatten())
print('RMSE of quadratic model:', rmse)
print('MAPE of quadratic model:', mape)

# add residuals column to df_train
train_df = train_df.copy() # avoid SettingWithCopyWarning
train_df['residuals'] = quadratic_residuals

# change exact_date column to datetime
from datetime import datetime
train_df['exact_date'] = train_df['exact_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

# add month column to df_train
train_df['month'] = train_df['exact_date'].apply(lambda x: x.month)

# calculate monthly residuals averages
monthly_residuals = train_df.groupby('month')['residuals'].mean().reset_index()

# calculate interpolated monthly residuals
from scipy.interpolate import interp1d
f = interp1d(monthly_residuals['month'], monthly_residuals['residuals'], kind='cubic')

# Merge the monthly_residuals DataFrame with train_df based on the 'month' column
train_df = pd.merge(train_df, monthly_residuals, on='month', how='left')

# Rename the column containing the mean residuals
train_df = train_df.rename(columns={'residuals_x': 'residuals', 'residuals_y': 'mean_monthly_residuals'})

# De-seasonalize the residuals column by subtracting the mean_monthly_residuals column
train_df['de_seasonalized_residuals'] = train_df['residuals'] - train_df['mean_monthly_residuals']

# Calculate the overall model Ct = Ft + St + Et where Ft is the trend and St is the seasonal component
from src.utils import extract_month_from_t
def predict(t):
    # Extract the month from t
    month = extract_month_from_t(t)
    # Find the mean monthly residual for that month
    mean_monthly_residual = monthly_residuals[monthly_residuals['month'] == month]['residuals'].values[0]
    return coef[0] * t**2 + coef[1] * t + mean_monthly_residual + intercept

# calculate predict using f from interp1d
def predict_interpolated(t):
    # Calculate seasonality using the function f
    t_month = ((t * 12 + 0.5) % 12) or 12
    seasonality = f(t_month)
    return coef[0] * t**2 + coef[1] * t + seasonality + intercept
    
# Predict the CO2 concentration for each row in df_train
train_df['predicted_CO2'] = train_df['t'].apply(predict)

# Repeat steps for the whole dataset: add predicted_CO2 column to df
df['predicted_CO2'] = df['t'].apply(predict)
split_date = pd.to_datetime(test_df['exact_date'].iloc[0])

# Repeat steps for test_df: add predicted_CO2 column to test_df
test_df = test_df.copy() # avoid SettingWithCopyWarning
test_df['predicted_CO2'] = test_df['t'].apply(predict)

# Calculate rsme and mape
from src.utils import calculate_rmse, calculate_mape
rmse = calculate_rmse(test_df['CO2_concentration'], test_df['predicted_CO2'])
mape = calculate_mape(test_df['CO2_concentration'], test_df['predicted_CO2'])
print('RMSE of final model:', rmse)
print('MAPE of final model:', mape)

 