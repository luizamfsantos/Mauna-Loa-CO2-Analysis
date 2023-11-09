from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

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

def plot_residuals(residuals):
    # Plot the residuals
    # Your code for plotting residuals
    pass

def calculate_rmse(y_test, y_pred):
    # Calculate RMSE
    return mean_squared_error(y_test, y_pred, squared=False)
