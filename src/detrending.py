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
    return reg, coef[0][:degree]


def remove_trend(data, coef, intercept, degree): 
    '''
    This function removes the deterministic trend from the time series data based on the selected model.
    
    Parameters:
    ----------
    data: pandas.DataFrame
        data to be de-trended
        contains column 'CO2_concentration'
    coef: numpy.array
        contains the coefficients for the linear model
    intercept: float
        the value of the intercept for the linear model
    degree: int
        specifies the type of model. 
        e.g. degree = 1, y = a_1*x+a_0
        e.g. degree = 2, y = a_2*x**2 + a_1*x+a_0

    Returns:
    --------
    data: pandas.DataFrame
        data detrended
    
    Example:
    --------
    >>> data = pd.DataFrame({'t':[1,2,3],'CO2_concentration':[5.1,5.4,6.3]})
    >>> coef = np.array([.5])
    >>> intercept = 4.5
    >>> degree = 1
    >>> data = remove_trend(data,coef,intercept,degree)
    >>> data
        t, CO2_concentration
    0   1, 4.6
    1   2, 4.4
    2   3, 4.8
    '''
    assert isinstance(coef, np.ndarray)
    assert len(coef) == degree
    coef = coef.reshape(1,degree)
    trend = np.array(data['t']).reshape(-1,1).dot(coef)
    trend = np.sum(trend, axis=1) + intercept
    trend = pd.DataFrame(trend, columns=['trend'])
    data['CO2_concentration'] = data['CO2_concentration'] - trend['trend']
    return data

if __name__ == '__main__':
    path = 'data/processed/CO2_clean.csv'
    df = pd.read_csv(path)
    from preprocessing import test_train_split
    train_df, _ = test_train_split(df)
    X_train = train_df[['t']]
    y_train = train_df[['CO2_concentration']]
    degree = 2
    reg, coef = polynomial_regression(X_train, y_train, degree)
    intercept = reg.intercept_
    data_detrended = remove_trend(df, coef, intercept, degree)
    # add a column 'month' to the dataframe
    data_detrended['month'] = pd.to_datetime(df['exact_date']).dt.month
    print(data_detrended.head())