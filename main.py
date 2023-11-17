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

# calculate MSE, AIC, BIC
from src.utils import evaluate_MSE, evaluate_AIC, evaluate_BIC
k = 1
quadratic_MSE = evaluate_MSE(quadratic_residuals)
quadratic_AIC = evaluate_AIC(k, quadratic_residuals)
quadratic_BIC = evaluate_BIC(k, quadratic_residuals)
print(f'Quadratic MSE: {quadratic_MSE}')
print(f'Quadratic AIC: {quadratic_AIC}')
print(f'Quadratic BIC: {quadratic_BIC}')