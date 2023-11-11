import pandas as pd
import numpy as np
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
    if degree > 1:
        X_poly = np.vander(X.squeeze(), degree + 1)
        reg = LinearRegression().fit(X_poly, y)
    else:
        reg = LinearRegression().fit(X, y)
    coef = reg.coef_
    return reg, coef


if __name__ == '__main__':
    path = 'data/processed/CO2_clean.csv'
    df = pd.read_csv(path)
    from preprocessing import test_train_split
    from utils import calculate_residuals, plot_residuals, calculate_rmse, calculate_mape
    train_df, test_df = test_train_split(df)
    X_train = train_df[['t']]
    y_train = train_df[['CO2_concentration']]
    X_test = test_df[['t']]
    y_test = test_df[['CO2_concentration']]
    # fit a polynomial regression model of degree 1
    reg, coef = polynomial_regression(X_train, y_train, degree=1)
    print(f'y={coef[0][0]}x + {reg.intercept_[0]}')
    residuals1 = calculate_residuals(reg, X_test, y_test, degree=1)
    pred = reg.predict(X_test)
    plot_residuals(residuals1)
    rmse = calculate_rmse(y_test, pred)
    mape = calculate_mape(y_test, pred)
    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape}')
    # # fit a polynomial regression model of degree 2
    reg, coef = polynomial_regression(X_train, y_train, degree=2)
    print(f'y={coef[0][0]}x^2 + {coef[0][1]}x + {reg.intercept_[0]}')
    residuals2 = calculate_residuals(reg, X_test, y_test, degree=2)
    X_test_poly_2 = PolynomialFeatures(2).fit_transform(X_test)
    pred = reg.predict(X_test_poly_2)
    plot_residuals(residuals2)
    rmse = calculate_rmse(y_test, pred)
    mape = calculate_mape(y_test, pred)
    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape}')
    # fit a polynomial regression model of degree 3
    reg, coef = polynomial_regression(X_train, y_train, degree=3)
    print(f'y={coef[0][0]}x^3 + {coef[0][1]}x^2 + {coef[0][2]}x + {reg.intercept_[0]}')
    residuals3 = calculate_residuals(reg, X_test, y_test, degree=3)
    X_test_poly_3 = PolynomialFeatures(3).fit_transform(X_test)
    pred = reg.predict(X_test_poly_3)
    plot_residuals(residuals3)
    rmse = calculate_rmse(y_test, pred)
    mape = calculate_mape(y_test, pred)
    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape}')
