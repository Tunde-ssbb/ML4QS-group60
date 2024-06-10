""" 
Maybe also do different techniques depending on associated expertise?
- distance: k-distance neighbourhood as there is a steady increase, so any drastic changes should be caught, but there is not e.g. a simple normal distribution
- pace(s): k-distribution
- HPM: k-distance neighbourhood to catch big jumps
- arm/leg acceleration: k-distance for the y-axis, simple distribution for the others
- arm/leg rotation: k-distance, due to wide variety of values
"""
#%% Dependencies
# Externals
import pandas as pd 
import os
import matplotlib.pyplot as plt
import numpy as np
# Internals
from lib.util.VisualizeDataset import VisualizeDataset as vd
from lib.Chapter3.OutlierDetection import 

class OutlierRemoval():
	def __init__(self) -> None:
		self.data_files:str[pd.DataFrame] = {}
		self.combined_data = pd.DataFrame()
	
	def read_data(self):
		path = os.path.abspath(os.path.join(os.getcwd(), 'measurement-data/'))
		self.combined_data = pd.read_csv()


		print(os.getcwd())
		walk_dir = os.path.abspath(os.path.join(os.getcwd(), 'measurement-data/processed/'))
		print(walk_dir)
		for root, subdirs, files in os.walk(walk_dir):
			print('--\nroot = ' + root)
			for subdir in subdirs:
				print('\t- subdirectory ' + subdir)
				with os.scandir(root+'/'+subdir) as it:
					for entry in it:
						data = pd.read_csv(entry)
						self.data_files[subdir+'-'+entry.name] = data
	
	def visualize_data(self, after_removal: bool):
		for file, data in self.data_files.items():
			vis = vd(file)
			name = file.split('-')[0]
			session = file.rsplit('_')[-1][0]
			moment = 'after-outlier-removal' if after_removal else 'before-outlier-removal'

			# Movement figs

			# Create subplots
			fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

			# First plot: arm acceleration
			axs[0].set_title(name+'_'+session+'_movement_'+moment)
			axs[0].plot(data['Time'], data[name+'_Arm_'+session+'_X (m/s^2)'], "r+", label='x')
			axs[0].plot(data['Time'], data[name+'_Arm_'+session+'_Y (m/s^2)'], "b+", label='y')
			axs[0].plot(data['Time'], data[name+'_Arm_'+session+'_Z (m/s^2)'], "g+", label='z')
			axs[0].set_ylabel("Arm acceleration (m/s^2)")
			axs[0].legend(loc = 'upper left')

			# Second plot: arm rotation
			axs[1].plot(data['Time'], data[name+'_Arm_'+session+'_X (rad/s)'], "r+", label='x')
			axs[1].plot(data['Time'], data[name+'_Arm_'+session+'_Y (rad/s)'], "b+", label='y')
			axs[1].plot(data['Time'], data[name+'_Arm_'+session+'_Z (rad/s)'], "g+", label='z')
			axs[1].set_ylabel("Arm rotation (rad/s)")
			axs[1].legend(loc = 'upper left')

			# Third plot: leg acceleration
			axs[2].plot(data['Time'], data[name+'_Been_'+session+'_X (m/s^2)'], "r+", label='x')
			axs[2].plot(data['Time'], data[name+'_Been_'+session+'_X (m/s^2)'], "b+", label='y')
			axs[2].plot(data['Time'], data[name+'_Been_'+session+'_X (m/s^2)'], "g+", label='z')
			axs[2].set_ylabel("Leg acceleration (m/s^2)")
			axs[2].legend(loc = 'upper left')

			# Fourth plot: leg rotation
			axs[3].plot(data['Time'], data[name+'_Been_'+session+'_X (rad/s)'], "r+", label='x')
			axs[3].plot(data['Time'], data[name+'_Been_'+session+'_X (rad/s)'], "b+", label='y')
			axs[3].plot(data['Time'], data[name+'_Been_'+session+'_X (rad/s)'], "g+", label='z')
			axs[3].set_ylabel("Leg rotation (rad/s)")
			axs[3].legend(loc = 'upper left')

			# Set common x-label
			axs[3].set_xlabel("Time (min)")

			# Set x-ticks only for the lowest subplot
			for ax in axs[:-1]:
				ax.tick_params(labelbottom=False)
			
			# Format the x-ticks as integers representing minutes
			axs[3].set_xticks(np.arange(0,3001, 600))
			axs[3].set_xticklabels(range(0,6))

			# Saving fig
			img_file=name+'_'+session+'_movement_'+moment+'.png'
			subdir = file.split('.')[0]
			plt.savefig("./figures/" + subdir + '/' + img_file)

			# Performance figs
			fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

			axs[0].set_title(name+'_'+session+'_performance_'+moment)
			axs[0].plot(data['Time'], data[name+'_'+session+'_Distance (meters)'], 'r.')
			axs[0].set_ylabel("Distance (meters)")

			axs[1].plot(data['Time'], data[name+'_'+session+'_Pace (seconds)'], 'b.')
			axs[1].set_ylabel("Pace (seconds)")

			axs[2].plot(data['Time'], data[name+'_Arm_'+session+'_HR (bpm)'], 'g.')
			axs[2].set_ylabel("heartrate (bpm)")

			axs[2].set_xlabel("Time (min)")

			for ax in axs[:-1]:
				ax.tick_params(labelbottom=False)

			# Format the x-ticks as integers representing minutes
			axs[2].set_xticks(np.arange(0,3001, 600))
			axs[2].set_xticklabels(range(0,6))

			# Saving fig
			img_file=name+'_'+session+'_performance_'+moment+'.png'
			plt.savefig("./figures/" + subdir + '/' + img_file)

	def _remove_outliers_distribution_based(self):
		pass

	def _remove_outliers_density_based(self):
		
		pass

	def remove_outliers(self):
		for file, data in self.data_files:
			for attribute in data.columns.values:
				if 'Distance' in attribute:
					self._remove_outliers_density_based()
				elif 'Pace' in attribute:
					self._remove_outliers_distribution_based()
				elif 'HR' in attribute:
					self._remove_outliers_density_based()
				elif '(rad/s)' in attribute:
					self._remove_outliers_density_based()
				elif '(m/s^2)' and 'Y' in attribute:
					self._remove_outliers_density_based()
				elif '(m/s^2)' in attribute:
					self._remove_outliers_distribution_based()
				else:
					raise ValueError(f'Attribute {attribute} was not expected.')
		