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