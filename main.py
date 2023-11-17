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

# train and save models
from src.detrending import polynomial_regression
from src.utils import save_model
import json

for degree in range(1, 5):
    # Perform polynomial regression
    reg, coef = polynomial_regression(X_train, y_train, degree)
    
    # Save model
    model_info = save_model(reg, coef, degree)
    print(model_info)
    
    # Map degree numbers to descriptive names
    degree_names = {1: 'linear', 2: 'quadratic', 3: 'cubic', 4: 'quartic'}
    
    # Get the descriptive name for the degree
    degree_name = degree_names.get(degree, f'degree_{degree}')
    
    # Create file name based on the degree name
    file_name = f'models/trend_modeling/{degree_name}_model.json'
    
    # Write model_info to a JSON file with the specific file name
    with open(file_name, 'w') as f:
        json.dump(model_info, f, indent=4)
