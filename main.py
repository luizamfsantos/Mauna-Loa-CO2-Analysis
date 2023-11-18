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

# calculate sinusoidal approximation
from src.unseasoning import calculate_sinusoidal_approximation
amp, phase_shift, mean = calculate_sinusoidal_approximation(train_df[['month', 'residuals']], col_name_t='month', col_name_y='residuals', period=12)

# plot sinusoidal approximation
from src.unseasoning import sine_function
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(1, 13)
y = sine_function(x, amp, phase_shift, mean)
plt.plot(x, y, label='Approximation')  # Add label for the line

plt.xlabel('Month')
plt.ylabel('Residuals')
plt.title('Sinusoidal Approximation')

# scatter plot of real values
plt.scatter(train_df['month'], train_df['residuals'], color='red', label='Real Values')

plt.legend()
plt.show()


# plt.savefig('images/seasonality_01')

# save sinusoidal approximation parameters
import json
seasonality_info = {
    'amp': amp,
    'phase_shift': phase_shift,
    'mean': mean,
    'period': 12
}
with open('models/seasonality_modeling/season_v01.json', 'w') as file:
    json.dump(seasonality_info, file)
    