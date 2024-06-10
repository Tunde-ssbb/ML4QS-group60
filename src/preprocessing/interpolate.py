import pandas as pd
import matplotlib.pyplot as plt


def interpolate(data, col_name, group_by_col=None):
    print(f" NaN values in {col_name} column: {data[col_name].isna().sum()}")
    if group_by_col == None:
        data[col_name] = data[col_name].interpolate()
    else:
        data[col_name] = data.groupby(group_by_col)[col_name].apply(lambda group: group.interpolate())

    return data
        
cols = ['time', 'arm_gyr_x', 'arm_gyr_y', 'arm_gyr_z', 'arm_acc_x', 'arm_acc_y', 'arm_acc_z', 'leg_gyr_x', 'leg_gyr_y', 'leg_gyr_z', 'leg_acc_x', 'leg_acc_y', 'leg_acc_z', 'dist', 'pace', 'HR', 'exp_lvl', 'session_id']

# read data and define a name
data_path = "./measurement-data/full_data.csv"
data = pd.read_csv(data_path, names = cols, header=0)


print(data.columns)


data = data.loc[data['session_id'] == 1]
plt.scatter(data['time'][200:600], data['HR'][200:600],  marker ="+", label = 'before interpolation', )

data = interpolate(data, 'HR', 'session_id')

plt.scatter(data['time'][200:600], data['HR'][200:600], marker = ".", label = 'after interpolation', alpha = 0.5)





print("nan values sesh id: ", data['session_id'].isna().sum())
print("nan values HR: ", data['HR'].isna().sum())


plt.legend()
plt.savefig("tunde_1_HR")

