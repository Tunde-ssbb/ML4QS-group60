import os 
import sys 
print(os.getcwd())
sys.path.append(os.getcwd())
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import src.feature_engineering.frequency as ffe
import src.feature_engineering.temporal as tfe
import src.feature_engineering.domain as dfe


# """
# features: 
# - freqs [0.2,0.3,0.4,0.6,0.8,0.9]
# - freq diff
# - freq std
# - difference in acceleration diff
# """

#data = pd.read_csv('./measurement-data/without_nans/fouried_data.csv', index_col='Unnamed: 0')
#data = tfe.diff_minmax(data, 30, ['leg_acc_y'])
#data = tfe.diff_rot_arm_been(data, 30)
#data= tfe.leg_acc(data, 30)
#print(data.head)
#
#if data.index.duplicated().any():
#    data = data.reset_index(drop=True)
#
## Verify there are no duplicate column names
#if data.columns.duplicated().any():
#    raise ValueError("Duplicate column names found in the DataFrame")
#
#
#columns_to_plot = [
#    'leg_acc_y_minmax_diff', 'arm_rot_abs', 'leg_rot_abs', 
#    'arm_leg_max_diff', 'arm_leg_max_time_diff', 'relative_rec', 
#    'norm_leg_acc_rec', 'norm_leg_acc_push'
#]
#
## Set the style of the plots
#sns.set(style="whitegrid")
#
## Create a figure with subplots
#fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=(15, len(columns_to_plot) * 5))
#
## Plot each column
#for i, column in enumerate(columns_to_plot):
#    xlim_min = data[column].quantile(0.01)  # Adjust percentile as needed
#    xlim_max = data[column].quantile(0.99)  # Adjust percentile as needed
#    sns.histplot(data=data, x=column, hue='experience_level', kde=True, ax=axes[i], bins=60)
#    axes[i].set_title(f'Distribution of {column} by Experience Level', fontsize=12)
#    axes[i].set_xlabel('')
#    axes[i].set_ylabel('Density', fontsize=12)
#    axes[i].set_xlim(xlim_min, xlim_max)
#    axes[i].tick_params(axis='x', pad=3)
#
#
#
#
#
#
#lines = [] 
#labels = [] 
#
#i = 0  
#for ax in fig.axes: 
#    Line, Label = ax.get_legend_handles_labels() 
#    # print(Label) 
#    lines.extend(Line) 
#    labels.extend(Label)
#    if i > 0:
#        ax.get_legend().remove() 
#    i+=1
#
#
#    
##plt.subplots_adjust()
#fig.legend(lines, labels, loc='upper right') 
## Adjust layout
#plt.tight_layout(h_pad=0.9)
#plt.show()

def feature_engineer(data):
    features = list(data.columns)
    features.remove("time")
    features.remove("exp_lvl")
    features.remove("dist")
    features.remove("pace")


    data = ffe.fourier_per_session(data, 100)

    data = ffe.remove_frequencies(data, 2, except_freq=[0.2,0.3,0.4,0.6,0.8,0.9])


    data = tfe.calculate_window_difference(data, 3, features)
    data = tfe.calculate_window_std(data, 40, features)

    data = tfe.diff_minmax(data, 30, ['leg_acc_y'])
    data = tfe.diff_rot_arm_been(data, 30)
    data= tfe.leg_acc(data, 30)

    data = dfe.arm_v_leg_acc_derivative_diff(data)


    print(f"engineered columns: {data.columns}")

    return data

