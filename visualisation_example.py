from src.visualisation.data_visualisation import create_movement_plot, create_distribution_plot, create_performance_plot, create_temporal_frequency_plot
from src.preprocessing.interpolate import interpolate
from src.feature_engineering.frequency import remove_frequencies
from src.feature_engineering.temporal import calculate_window_difference
import pandas as pd
import os

# define these column names for plots to work
# exp_lvl and session_id only required for the performance plot
cols = ['time', 'arm_gyr_x', 'arm_gyr_y', 'arm_gyr_z', 'arm_acc_x', 'arm_acc_y', 'arm_acc_z', 'leg_gyr_x', 'leg_gyr_y', 'leg_gyr_z', 'leg_acc_x', 'leg_acc_y', 'leg_acc_z', 'dist', 'pace', 'HR', 'exp_lvl', 'session_id']

# read data and define a name
data_path = "./fouried_data.csv"
data = pd.read_csv(data_path)
session_name = "full_data"

#interpolate HR values per session_id
#print("nan before", data['HR'].isna().sum())
#data = interpolate(data, ['HR', 'pace', 'dist'], 'session_id')
#print("nan after", data['HR'].isna().sum())

print(f"number of columns before: {len(data.columns)}")

#data = remove_frequencies(data, 2)
#data = calculate_window_difference(data, 3, data.columns[5:])

print(f"number of columns after: {len(data.columns)}")


#data.to_csv("./smaller_data_with_diff.csv")

#define folder which to save the plots to 
path = os.path.abspath("./figures/temporal_frequency")

#create_distribution_plot(data, session_name, path)
#create_movement_plot(data, session_name, path)
#create_performance_plot(data, session_name, path)

features = ['arm_gyr_x', 'arm_gyr_y', 'arm_gyr_z', 'arm_acc_x', 'arm_acc_y', 'arm_acc_z', 'leg_gyr_x', 'leg_gyr_y', 'leg_gyr_z', 'leg_acc_x', 'leg_acc_y', 'leg_acc_z']

for feature in features:
    create_temporal_frequency_plot(data, feature, path)


