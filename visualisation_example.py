import src.visualisation.data_visualisation as dv # create_movement_plot, create_distribution_plot, create_performance_plot, create_temporal_frequency_plot
from src.preprocessing.interpolate import interpolate
import src.feature_engineering.do_fe as fe

import pandas as pd
import os



# read data and define a name
data_path = "./measurement-data/full_data.csv"
data = pd.read_csv(data_path, index_col=0)


data = fe.feature_engineer(data)

print(data.columns)
print(data.shape)

data.to_csv("fe_data.csv")

"""

interpolate HR values per session_id
print("nan before", data['HR'].isna().sum())
data = interpolate(data, ['HR', 'pace', 'dist'], 'session_id')
print("nan after", data['HR'].isna().sum())

features = ['arm_gyr_x', 'arm_gyr_y', 'arm_gyr_z', 'arm_acc_x', 'arm_acc_y', 'arm_acc_z', 'leg_gyr_x', 'leg_gyr_y', 'leg_gyr_z', 'leg_acc_x', 'leg_acc_y', 'leg_acc_z']

features = []
for feature in data.columns:
    freq = ffe.extract_frequency(feature)
    if freq != None:
        features.append(feature)
print(f"number of columns before: {len(data.columns)}")

#data = remove_frequencies(data, 2)
#data = calculate_window_difference(data, 3, data.columns[5:])
#data = tfe.calculate_window_std(data, 100, features)

print(f"number of columns after: {len(data.columns)}")


#data.to_csv("./smaller_data_.csv")

#define folder which to save the plots to 
path = os.path.abspath("./")

#dv.create_distribution_plot(data, session_name, path)
#dv.create_movement_plot(data, session_name, path)
#dv.create_performance_plot(data, session_name, path)

features = ['arm_gyr_x', 'arm_gyr_y', 'arm_gyr_z', 'arm_acc_x', 'arm_acc_y', 'arm_acc_z', 'leg_gyr_x', 'leg_gyr_y', 'leg_gyr_z', 'leg_acc_x', 'leg_acc_y', 'leg_acc_z']

features = ['arm_gyr_x']
for feature in features:
    dv.create_freq_distributions_by_experience_level_plot(data, feature, path)

"""

