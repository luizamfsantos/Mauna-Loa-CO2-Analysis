import pandas as pd 
from scipy.interpolate import interp1d

def calculate_monthly_averages(data): 
    '''This function calculates the average signal for each month.'''
    average_signal = data.groupby('month')['CO2_concentration'].mean().reset_index()
    return average_signal

def calculate_interpolation(data): 
    '''This function calculates the interpolation of the average signal.'''
    average_data  = calculate_monthly_averages(data)
    interpolated_signal = interp1d(average_data['month'], average_data['CO2_concentration'], kind='linear', fill_value='extrapolate')
    return interpolated_signal

def plot_periodic_signal(data): 
    '''This function plots the periodic signal against time.'''
    interpolated_signal = calculate_interpolation(data)
    interpolated_values = interpolated_signal(data['month'])


if __name__ == "__main__":
    path = 'data/processed/CO2_detrended.csv'
    df = pd.read_csv(path)
    from preprocessing import test_train_split
    df_train, df_test = test_train_split(df)
    average_signal = calculate_monthly_averages(df_train)
    print(average_signal)
    interpolated_signal = calculate_interpolation(df_train)
    print(interpolated_signal)
    # plot_periodic_signal(df_train)