import src.visualisation.data_visualisation as dv # create_movement_plot, create_distribution_plot, create_performance_plot, create_temporal_frequency_plot
from src.preprocessing.interpolate import interpolate
import src.feature_engineering.do_fe as fe

import pandas as pd
import os



# read data and define a name


freq_path = os.path.abspath("./figures/freq_dist_per_lvl")
fig_path = os.path.abspath("./figures")
temp_path = os.path.abspath("./figures/temporal_frequency_new")


data_path = "./measurement-data/full_data.csv"
data = pd.read_csv(data_path)




#dv.create_movement_plot(data, "full_data", fig_path)
dv.create_performance_plot(data, "full_data" , fig_path)


# data_path = "./src/machine_learning/fouried_data.csv"
# data = pd.read_csv(data_path)

# features = ['arm_gyr_x', 'arm_gyr_y', 'arm_gyr_z', 'arm_acc_x', 'arm_acc_y', 'arm_acc_z', 'leg_gyr_x', 'leg_gyr_y', 'leg_gyr_z', 'leg_acc_x', 'leg_acc_y', 'leg_acc_z']

# for feature in features:
#     dv.create_temporal_frequency_plot(data, feature, temp_path)


# for feature in features:
#     dv.create_freq_distributions_by_experience_level_plot(data, feature, freq_path)

# dv.create_distribution_plot(data, "full_data",  fig_path)








