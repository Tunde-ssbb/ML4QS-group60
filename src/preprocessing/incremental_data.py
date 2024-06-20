import pandas as pd
from src.machine_learning.util import read_and_preprocess
import matplotlib.pyplot as plt

def incremental_increase_decrease_data(data):
    no_transform = ['exp_lvl', 'time', 'session_id']
    no_transform_data = data[no_transform]

    data = data.drop(columns = no_transform)

    data = data.pct_change(1)

    data = pd.concat([no_transform_data, data], axis=1)
    return data
    

"""
data = read_and_preprocess("./measurement-data/without_nans/fouried_data.csv")

plt.style.use('fivethirtyeight')
data.plot(subplots=True,
                  layout=(6, 3),
                  figsize=(24,24),
                  fontsize=10, 
                  linewidth=2, 
                  title='Visualization of the Features')
plt.savefig("./figures/eda/Vis_features")

data = incremental_increase_decrease_data(data)

plt.style.use('fivethirtyeight')
data.plot(subplots=True,
                  layout=(6, 3),
                  figsize=(24,24),
                  fontsize=10, 
                  linewidth=2, 
                  title='Visualization of the transformed Features')
plt.savefig("./figures/eda/vis_incr_features")

"""