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
    pass