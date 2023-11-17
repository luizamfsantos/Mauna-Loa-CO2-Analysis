import statsmodels.api as sm
import matplotlib.pyplot as plt

def plot_acf_pacf(residuals, lags=30, super_title=None, save_path=None):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Plot ACF
    sm.graphics.tsa.plot_acf(residuals, lags=lags, ax=ax[0])
    ax[0].set_title('Autocorrelation Function (ACF)')

    # Plot PACF
    sm.graphics.tsa.plot_pacf(residuals, lags=lags, ax=ax[1])
    ax[1].set_title('Partial Autocorrelation Function (PACF)')

    if super_title:
        plt.suptitle(super_title, fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def remove_seasonality(data, model): 
    '''This function removes the seasonality from the time series data based on the selected model.'''
    pass

def compute_acf_pacf(data): 
    '''This function computes the ACF and PACF for the pre-processed data.'''
    pass

def fit_ma_model(data): 
    '''This function fits an MA(1) model to the pre-processed data.'''
    pass

def fit_ar_model(data): 
    '''This function fits an AR(1) model to the pre-processed data.'''
    pass

def evaluate_model_performance(model): 
    '''This function evaluates the fitted model using AIC and BIC.'''
    pass

def conduct_residual_analysis(residuals): 
    '''This function checks for remaining patterns in the residuals.'''
    pass