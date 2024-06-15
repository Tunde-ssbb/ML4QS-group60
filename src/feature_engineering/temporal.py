import pandas as pd
import numpy as np
# from lib.Chapter4.TemporalAbstraction import NumericalAbstraction
import datetime as dt

# calculate crude derivative of columns in col_names
def calculate_window_difference(data, windowsize, column_names):
    
    for col in column_names:
        # Calculate the difference for each datapoint with index i and i-windowsize
        data[col + '_diff'] = data[col] - data[col].shift(windowsize)
    
    return data

# calculate a weighted average over a window (if weights is left to None this is just an aggregated mean)
def calculate_weighted_average(data, windowsize, column_names, weights=None):
    # Create a copy of the DataFrame to avoid modifying the original one
    
    # If no weights are provided, use equal weights
    if weights is None:
        weights = np.ones(windowsize)
    else:
        # Ensure weights length matches the window size
        if len(weights) != windowsize:
            raise ValueError("Length of weights must match the window size")

    # Normalize weights so that they sum to 1
    weights = np.array(weights) / np.sum(weights)

    for col in column_names:
        # Calculate the weighted average for each window
        data[col + '_weighted_avg'] = data[col].rolling(window=windowsize).apply(
            lambda x: np.sum(weights * x), raw=True
        )
    
    return data

def calculate_window_std(data, windowsize, col_names):

    for col in col_names:
        # Compute the rolling standard deviation
        rolling_std_col = f'{col}_std'
        data[rolling_std_col] = data[col].rolling(window=windowsize).std()
    
    return data


def diff_minmax(data, windowsize, col_names):
    for col in col_names:
        if col in data.columns:
            diff = f'{col}_minmax_diff'
            data[diff] = data[col].rolling(window=windowsize).max() - data[col].rolling(window=windowsize).min() 
        else:
            print(f'columname {col} is not valid')
    return data

def diff_rot_arm_been(data, windowsize):
    diff= 'arm_leg_max_diff'
    diff2 = 'arm_leg_max_time_diff'
    data['arm_rot_abs'] = data[['arm_gyr_y', 'arm_gyr_z']].mean(axis=1)
    data['leg_rot_abs'] = data[['leg_gyr_y', 'leg_gyr_z']].mean(axis=1)
    
    temp_data = data[['arm_rot_abs', 'leg_rot_abs']].copy()
    data[diff] = data['arm_rot_abs'].rolling(window=windowsize).max() - data['leg_rot_abs'].rolling(window=windowsize).max()
    temp_data['onnodig']= temp_data['arm_rot_abs'].rolling(window=windowsize).apply(get_idxmax, raw=False)
    temp_data['onnodig2'] = temp_data['leg_rot_abs'].rolling(window=windowsize).apply(get_idxmax, raw=False)
    data[diff2] = temp_data['onnodig'] - temp_data['onnodig2']

    temp_data['duur_rec_arm'] = temp_data['arm_rot_abs'].rolling(window=windowsize).apply(lambda x: (x < 0).sum(), raw=False)
    temp_data['duur_rec_been'] = temp_data['leg_rot_abs'].rolling(window=windowsize).apply(lambda x: (x < 0).sum(), raw=False)
    temp_data['duur_push_arm'] = temp_data['arm_rot_abs'].rolling(window=windowsize).apply(lambda x: (x  >0).sum(), raw=False)
    temp_data['duur_push_been'] = temp_data['leg_rot_abs'].rolling(window=windowsize).apply(lambda x: (x > 0).sum(), raw=False)

    try:
        data['relative_rec'] = (temp_data['duur_rec_arm'] + temp_data['duur_rec_been'])/ (temp_data['duur_push_arm'] + temp_data['duur_push_been'])
    except: 
        data['relative_rec'] = (temp_data['duur_rec_arm'] + temp_data['duur_rec_been'])
    return data
    

def get_idxmax(window):
    return (window.idxmax())*0.1


def leg_acc(data, windowsize):
    norm_rec = 'norm_leg_acc_rec'
    norm_push = 'norm_leg_acc_push'
    try: 
        data[norm_rec] = data['leg_acc_y'].rolling(window=windowsize).max()/data[data['leg_acc_y']> 0].rolling(window=windowsize).mean()
    except: 
        data[norm_rec] = data['leg_acc_y'].rolling(window=windowsize).max()
    
    try: 
        data[norm_push] = data['leg_acc_y'].rolling(window=windowsize).min()/data[data['leg_acc_y']< 0].rolling(window=windowsize).mean()
    except: 
        data[norm_push] = data['leg_acc_y'].rolling(window=windowsize).min()
    return data
