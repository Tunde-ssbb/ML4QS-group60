import pandas as pd
import numpy as np

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