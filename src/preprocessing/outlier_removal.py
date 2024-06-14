#%% Dependencies
# Externals
import pandas as pd 
import os
import numpy as np
import matplotlib.pyplot as plt
# Internals
from lib.util.VisualizeDataset import VisualizeDataset as vd
from lib.Chapter3.OutlierDetection import DistanceBasedOutlierDetection, DistributionBasedOutlierDetection

class OutlierRemoval():
	def __init__(self) -> None:
		self.combined_data = pd.DataFrame()
	
	def read_data(self, path):
		self.combined_data = pd.read_csv(path)

	def _remove_outliers_distribution_based_mixed_models(self, col, n_components):
		print(f'(mixed models) #rows prior to outlier removal of col {col}: {self.combined_data.shape[0]}')
		# Applying the k normal distributions fitting
		mm = DistributionBasedOutlierDetection()
		with_mixture = mm.mixture_model(self.combined_data, col, n_components)

		# Constants
		outlier_portion = int(with_mixture[col+'_mixture'].shape[0] / 200)
		cutoff = with_mixture[col+'_mixture'].nsmallest(outlier_portion).max()

		# Removing outliers
		with_mixture[col].mask(with_mixture[col+'_mixture'] <= cutoff, inplace=True)
		self.outlier_visualization(col, col+'_mixture', with_mixture)
		
		self.combined_data = with_mixture.drop(col+'_mixture', axis=1)
		print(f'#NaNs: {self.combined_data.isna().sum()}')

	def _remove_outliers_density_based(self, col):
		print(f'(LOF) #rows prior to outlier removal of col {col}: {self.combined_data.shape[0]}')
		# Calculating LOFs
		od = DistanceBasedOutlierDetection()
		with_lof = od.local_outlier_factor(self.combined_data, [col], 'euclidean', 5)

		# Constants
		std = with_lof['lof'].std()
		mean = with_lof['lof'].mean()
		print(with_lof['lof'].std())
		print(with_lof['lof'].mean())
		cutoff_low = mean - std
		cutoff_high = mean + std
		
		# Removing outliers based on cutoff
		with_lof[col].mask((with_lof['lof'] <= cutoff_low) | (with_lof['lof'] >= cutoff_high), inplace=True)
		self.outlier_visualization(col, 'lof')
		
		self.combined_data = with_lof.drop('lof', axis=1)
		print(f'#NaNs: {self.combined_data.isna().sum()}')

	def remove_outliers(self):
		print(self.combined_data.shape[0])
		for attribute in self.combined_data.columns.values:
			print(attribute)
			if 'dist' in attribute or 'pace' in attribute or '0' in attribute or 'time' in attribute or 'experience_level' in attribute or 'session_id' in attribute:
				continue
			elif 'HR' in attribute:
				self._remove_outliers_distribution_based_mixed_models(attribute, 3)
			elif 'leg_gyr' in attribute and 'z' in attribute:
				self._remove_outliers_distribution_based_mixed_models(attribute, 3)
			elif 'leg_gyr' in attribute:
				self._remove_outliers_distribution_based_mixed_models(attribute, 2)
			elif 'arm_gyr' in attribute:
				self._remove_outliers_distribution_based_mixed_models(attribute, 1)
			elif 'leg_acc' in attribute and 'x' in attribute:
				self._remove_outliers_distribution_based_mixed_models(attribute, 1)
			elif 'leg_acc' in attribute and 'y' in attribute:
				self._remove_outliers_distribution_based_mixed_models(attribute,3)
			elif 'leg_acc' in attribute and 'z' in attribute:
				self._remove_outliers_distribution_based_mixed_models(attribute, 2)
			elif 'arm_acc' in attribute and 'x' in attribute:
				self._remove_outliers_distribution_based_mixed_models(attribute, 2)
			elif 'arm_acc' in attribute:
				self._remove_outliers_distribution_based_mixed_models(attribute, 1)
			else:
				raise ValueError(f'Attribute {attribute} was not expected.')
		
	def write_data(self, path):
		self.combined_data.to_csv(path, index=False) 

	def outlier_visualization(self, og_col, col, data, path='./figures/without_outliers'):
		# Create subplots
		fig, axs = plt.subplots(figsize=(16, 16), squeeze=False)

		fig.suptitle(col, fontsize = 20)

		# First plot: arm acceleration
		axs[0,0].set_title(col, size = 20)
		axs[0,0].hist(data[col], bins = 500, alpha = 0.5, label=col[0])
		axs[0,0].set_xlabel("outlier method results", fontsize=20)
		axs[0,0].set_ylabel("frequency", fontsize=20)
		axs[0,0].tick_params(labelsize=16)
		axs[0,0].legend(loc = 'upper left', fontsize = 20)

		p = path + "/" +  col + "_distributions"
		plt.savefig(p)

		print(f"Saved distribution plot to: {p}")

		# Create subplots
		fig, axs = plt.subplots(figsize=(12, 16), squeeze=False)

		axs[0,0].plot(data[og_col], data[col], "r+", label='x')
		axs[0,0].set_ylabel(f'{col} vals')
		axs[0,0].legend(loc = 'upper left')

		# Set common x-label
		axs[0,0].set_xlabel(f"{og_col} vals")

		# Set x-ticks only for the lowest subplot
		for ax in axs[:-1]:
			ax.tick_params(labelbottom=False)

		# Format the x-ticks as integers representing minutes
		axs[0,0].set_xticks(np.arange(data[og_col].min(), data[og_col].max()))

		p = path + "/" +  col + "_performance"
		plt.savefig(p)

		print(f"Saved performance plot to: {p}")
