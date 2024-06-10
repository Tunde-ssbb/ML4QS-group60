#%% Dependencies
# Externals
import pandas as pd 
import os
# Internals
from lib.util.VisualizeDataset import VisualizeDataset as vd
from lib.Chapter3.OutlierDetection import DistanceBasedOutlierDetection, DistributionBasedOutlierDetection

class OutlierRemoval():
	def __init__(self) -> None:
		self.combined_data = pd.DataFrame()
	
	def read_data(self):
		path = os.path.abspath(os.path.join(os.getcwd(), 'measurement-data/full_data.csv'))
		self.combined_data = pd.read_csv(path)

	def _remove_outliers_distribution_based_mixed_models(self, col):
		print(f'#rows prior to outlier removal of col {col}: {self.combined_data.shape[0]}')
		# Applying the k normal distributions fitting
		mm = DistributionBasedOutlierDetection()
		with_mixture = mm.mixture_model(self.combined_data, col)

		# Constants
		cutoff = 0.05

		# Removing outliers
		mask = with_mixture[col+'_mixture'] <= cutoff
		self.combined_data = self.combined_data[mask]
		print(f'#rows after outlier removal of col: {col}: {self.combined_data.shape[0]}')

	def _remove_outliers_distribution_based_chauvenet(self, col):
		print(f'#rows prior to outlier removal of col {col}: {self.combined_data.shape[0]}')
		# Applying chauvenet's criterion
		ch = DistributionBasedOutlierDetection()
		with_outlier = ch.chauvenet(self.combined_data, col, 2)

		# Removing outliers
		mask = with_outlier[col+'_outlier'] == 0
		self.combined_data = self.combined_data[mask]
		print(f'#rows after outlier removal of col: {col}: {self.combined_data.shape[0]}')

	def _remove_outliers_density_based(self, col):
		print(f'#rows prior to outlier removal of col {col}: {self.combined_data.shape[0]}')
		# Calculating LOFs
		od = DistanceBasedOutlierDetection()
		with_lof = od.local_outlier_factor(self.combined_data, [col], 'euclidean', 5)

		# Constants
		cutoff_low = 0.95
		cutoff_high = 1.05

		# Removing outliers based on cutoff
		mask = with_lof['lof'] >= cutoff_low & with_lof['lof'] <= cutoff_high
		self.combined_data = self.combined_data[mask]
		print(f'#rows after outlier removal of col: {col}: {self.combined_data.shape[0]}')

	def remove_outliers(self):
		print(self.combined_data.shape[0])
		for attribute in self.combined_data.columns.values:
			if '0' in attribute or 'time' in attribute or 'experience_level' in attribute or 'session_id' in attribute:
				continue
			elif 'dist' in attribute:
				self._remove_outliers_density_based(attribute)
			elif 'pace' in attribute:
				self._remove_outliers_density_based(attribute)
			elif 'HR' in attribute:
				self._remove_outliers_density_based(attribute)
			elif 'leg_gyr' and 'z' in attribute:
				self._remove_outliers_density_based(attribute)
			elif 'leg_gyr' in attribute:
				self._remove_outliers_distribution_based_mixed_models(attribute)
			#elif 'arm_gyr' in attribute:
			#	self._remove_outliers_distribution_based_chauvenet(attribute)
			elif 'leg_acc' and 'x' in attribute:
				self._remove_outliers_distribution_based_chauvenet(attribute)
			elif 'leg_acc' and 'y' in attribute:
				self._remove_outliers_density_based(attribute)
			elif 'leg_acc' and 'z' in attribute:
				self._remove_outliers_distribution_based_mixed_models(attribute)
			elif 'arm_acc' in attribute:
				self._remove_outliers_distribution_based_mixed_models(attribute)
			else:
				raise ValueError(f'Attribute {attribute} was not expected.')
		
	def output_data(self):
		self.combined_data.to_csv('./measurement-data/outliers_removed/orem.csv')
		pass