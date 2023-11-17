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

# calculate linear residuals
quadratic_residuals = y_train - y_pred

# plot ACF/PACF
from src.autocovariance import plot_acf_pacf
import matplotlib.pyplot as plt

# Plot ACF/PACF
plot_acf_pacf(quadratic_residuals, lags=30, super_title='ACF/PACF of Residuals from Quadratic Trend Model', save_path='images/trend_residuals_acf_pacf.png')
