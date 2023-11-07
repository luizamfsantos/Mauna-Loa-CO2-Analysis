from sklearn.metrics import mean_squared_error

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
