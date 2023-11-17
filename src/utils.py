from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm 

def calculate_residuals(model, X_test, y_test, degree):
    '''
    Calculate the residuals for a given model.

    Parameters:
    model (object): The trained regression model.
    X_test (array-like): Test data features.
    y_test (array-like): True labels for the test data.
    degree (int): The degree of the polynomial regression model.

    Returns:
    residuals (array-like): Residual errors calculated as y_test - model.predict(X_test).
    '''
    try:
        # Transform X to the correct degree if degree > 1
        if degree > 1:
            poly = PolynomialFeatures(degree)
            X_test = poly.fit_transform(X_test)
        
        # Predict with the model
        y_pred = model.predict(X_test)
        
        # Calculate residuals
        residuals = y_test - y_pred

        return residuals
    except Exception as e:
        raise ValueError(f"An error occurred during residual calculation: {e}")

def plot_residuals(residuals):
    '''
    Plot the residuals for a given model.

    Parameters:
    residuals (array-like): Residual errors.

    Returns:
    None
    '''
    plt.scatter(np.arange(len(residuals)), residuals, color='b', alpha=0.6)
    plt.title('Residuals Plot')
    plt.xlabel('Observation')
    plt.ylabel('Residuals')
    plt.show()


def estimate_parameters(X_train, y_train, degree):
    # Estimate parameters for the polynomial model
    model_coeffs = np.polyfit(X_train, y_train, degree)
    return np.poly1d(model_coeffs)


def calculate_rmse(y_test, y_pred):
    # Calculate RMSE
    return mean_squared_error(y_test, y_pred, squared=False)

def calculate_mape(y_test, y_pred):
    # Calculate MAPE
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

def evaluate_MSE(residuals):
    ''' 
    Finds the MSE given the residuals of the model.
    '''
    return np.mean(residuals**2)

def evaluate_AIC(k, residuals):
  """
  Finds the AIC given the number of parameters estimated and 
  the residuals of the model. Assumes residuals are distributed 
  Gaussian with unknown variance. 
  """
  standard_deviation = np.std(residuals)
  log_likelihood = norm.logpdf(residuals, 0, scale=standard_deviation)
  return 2 * k - 2 * np.sum(log_likelihood)

def evaluate_BIC(k, residuals):
  """
  Finds the AIC given the number of parameters estimated and 
  the residuals of the model. Assumes residuals are distributed 
  Gaussian with unknown variance. 
  """
  standard_deviation = np.std(residuals)
  log_likelihood = norm.logpdf(residuals, 0, scale=standard_deviation)
  return k * np.log(len(residuals)) - 2 * np.sum(log_likelihood)

def generate_model(degree):
    terms = [f'coefficients[{i}]*x**{degree - i}' for i in range(degree)]
    model = ' + '.join(terms)
    return model

def save_model(reg, coef, degree):
    model = generate_model(degree)
    model_info = {
        'coefficients': coef.tolist(),
        'intercept': reg.intercept_.tolist()[0],
        'params': reg.get_params(),
        'order': degree,
        'model': model
    }
    return model_info