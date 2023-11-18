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

# plot monthly residuals, interpolation and real data
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
f = interp1d(monthly_residuals['month'], monthly_residuals['residuals'])
x = np.linspace(1, 12, 100)
y = f(x)
plt.figure(figsize=(10, 6))
plt.plot(monthly_residuals['month'], monthly_residuals['residuals'], 'o', x, y, '-',c='#2F76D9')
plt.scatter(train_df['month'], train_df['residuals'], c='#2FCBD9')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Seasonal Influence', fontsize=14)
plt.xticks(np.arange(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.title('Periodic Signal', fontsize=16)
plt.legend(['Monthly Averages', 'Interpolation', 'Real Data'])
plt.savefig('images/seasonality_03.png')
plt.show()




  