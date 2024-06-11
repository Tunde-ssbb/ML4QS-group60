import pandas as pd 
import matplotlib.pyplot as plt
from lib.Chapter4.FrequencyAbstraction import FourierTransformation
from src.preprocessing.interpolate import interpolate


def fourier_per_session(data, window_size, sampling_rate = 10):
    fourier_cols = ['arm_gyr_x', 'arm_gyr_y', 'arm_gyr_z', 'arm_acc_x', 'arm_acc_y', 'arm_acc_z', 'leg_gyr_x', 'leg_gyr_y', 'leg_gyr_z', 'leg_acc_x', 'leg_acc_y', 'leg_acc_z']
    print(f"performing fourier on cols: {fourier_cols}")

    ft = FourierTransformation()

    sessions = data['session_id'].unique()
    print(f"seperating sessions: {sessions}")

    sessions_fouried = []

    # per session:
    for session_id in sessions:

        # extract session data only
        session_data = data.loc[data['session_id'] == session_id]

        # interpolate to ensure no missing values
        session_data = interpolate(session_data, fourier_cols)  

        # perform fourier
        session_data = ft.abstract_frequency(session_data, fourier_cols, window_size=window_size, sampling_rate = sampling_rate)
        sessions_fouried.append(session_data)

    # combine sessions data
    full_data_fouried = pd.concat(sessions_fouried, axis = 0)

    return full_data_fouried


#run this to create the fouried dataset
"""

cols = ['time', 'arm_gyr_x', 'arm_gyr_y', 'arm_gyr_z', 'arm_acc_x', 'arm_acc_y', 'arm_acc_z', 'leg_gyr_x', 'leg_gyr_y', 'leg_gyr_z', 'leg_acc_x', 'leg_acc_y', 'leg_acc_z', 'dist', 'pace', 'HR', 'exp_lvl', 'session_id']

# read data and define a name
data_path = "./measurement-data/full_data.csv"
data = pd.read_csv(data_path, names=cols, header = 0)
session_name = "full_data"


data_fourier = fourier_per_session(data, 100)
print(data_fourier)

data_fourier.to_csv("./fouried_data.csv")

"""
