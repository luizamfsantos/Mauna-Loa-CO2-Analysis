import pandas as pd 
from scipy.interpolate import interp1d
import numpy as np
from scipy.optimize import curve_fit

def calculate_monthly_averages(data): 
    '''This function calculates the average signal for each month.'''
    average_signal = data.groupby('month')['CO2_concentration'].mean().reset_index()
    return average_signal

def sine_function(X, amp, phase_shift, mean):
    return amp * np.sin(X + phase_shift) + mean

def calculate_sinusoidal_approximation(data, col_name_t='month', col_name_y='CO2_concentration', period=12):
    ''' 
    Fits a sinusoidal approximation to the data assuming the model has been de-trended.
    
    Parameters:
    data (DataFrame): Input data containing time (col_name_t) and values (col_name_y).
    col_name_t (str): Name of the column containing time-related data (default: 'month').
    col_name_y (str): Name of the column containing the target values (default: 'CO2_concentration').
    period (int): Period of the sinusoidal function (default: 12).
    
    Returns:
    amp (float): Amplitude of the sinusoidal function.
    phase_shift (float): Phase shift of the sinusoidal function.
    mean (float): Mean of the sinusoidal function.
    '''
    try:
        params, _ = curve_fit(f=sine_function, xdata=data[col_name_t], ydata=data[col_name_y], p0=[1, 0, 0])
        amp, phase_shift, mean = params
        return amp, phase_shift, mean
    except RuntimeError:
        print("Optimization failed. Check input data and initial parameters.")
        return None
    except KeyError as e:
        print(f"Column {str(e)} not found in the provided data.")
        return None

def plot_periodic_signal(data): 
    '''This function plots the periodic signal against time.'''
    interpolated_signal = calculate_interpolation(data)
    interpolated_values = interpolated_signal(data['month'])


if __name__ == "__main__":
    path = 'data/processed/CO2_detrended.csv'
    df = pd.read_csv(path)
    # from preprocessing import test_train_split
    # df_train, df_test = test_train_split(df)
    # average_signal = calculate_monthly_averages(df_train)
    # print(average_signal)
    # interpolated_signal = calculate_interpolation(df_train)
    # print(interpolated_signal)
    # # plot_periodic_signal(df_train)