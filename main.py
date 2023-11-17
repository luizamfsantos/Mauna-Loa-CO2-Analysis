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

# plot results of quadratic model on top of training data
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color='blue', s=1)
plt.plot(X_train, y_pred, color='red')
plt.title('Quadratic Model')
plt.xlabel('Time (months)')
plt.ylabel('CO2 concentration (ppm)')
plt.show()