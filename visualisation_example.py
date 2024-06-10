from src.visualisation.data_visualisation import create_movement_plot, create_distribution_plot, create_performance_plot
import pandas as pd
import os

cols = ['time', 'arm_gyr_x', 'arm_gyr_y', 'arm_gyr_z', 'arm_acc_x', 'arm_acc_y', 'arm_acc_z', 'leg_gyr_x', 'leg_gyr_y', 'leg_gyr_z', 'leg_acc_x', 'leg_acc_y', 'leg_acc_z', 'dist', 'pace', 'HR', 'exp_lvl', 'session_id']

data_path = "./measurement-data/full_data.csv"
data = pd.read_csv(data_path, names=cols, header = 0)
session_name = "full_data"

path = os.path.abspath("./figures")


create_distribution_plot(data, session_name, path)
create_movement_plot(data, session_name, path)
create_performance_plot(data, session_name, path)

