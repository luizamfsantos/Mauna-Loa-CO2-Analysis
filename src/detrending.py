import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def polynomial_regression(X, y, degree):
    '''
    Fits a polynomial regression model of the specified degree to the input data.

    Parameters:
    -----------
    X: pandas.DataFrame 
        The input features.
    y: pandas.DataFrame 
        The target variable.
    degree: int
        The degree of the polynomial regression model.

    Returns:
    --------
    tuple: A tuple containing the fitted regression model and the R-squared score.
    '''
    poly = PolynomialFeatures(degree=degree, include_bias=False)  # create a PolynomialFeatures object
    X_poly = poly.fit_transform(X)  # transform the features to include polynomial terms
    reg = LinearRegression().fit(X_poly, y)  # fit the linear regression model
    coef = reg.coef_ # get the parameters of the model
    return reg, coef


def calculate_residuals(y_test, y_pred): 
    '''This function calculates the residual errors for a given model.'''
    pass

def calculate_metrics(y_test, y_pred): 
    '''This function calculates the RMSE and MAPE for a given model.'''
    pass

if __name__ == '__main__':
    path = 'data/processed/CO2_clean.csv'
    df = pd.read_csv(path)
    from preprocessing import test_train_split
    train_df, test_df = test_train_split(df)
    X_train = train_df[['t']]
    y_train = train_df[['CO2_concentration']]
    # fit a polynomial regression model of degree 1
    reg, coef = polynomial_regression(X_train, y_train, degree=1)
    print(f'y={coef[0][0]}x + {reg.intercept_[0]}')
    # fit a polynomial regression model of degree 2
    reg, coef = polynomial_regression(X_train, y_train, degree=2)
    print(f'y={coef[0][0]}x^2 + {coef[0][1]}x + {reg.intercept_[0]}')
    # fit a polynomial regression model of degree 3
    reg, coef = polynomial_regression(X_train, y_train, degree=3)
    print(f'y={coef[0][0]}x^3 + {coef[0][1]}x^2 + {coef[0][2]}x + {reg.intercept_[0]}')