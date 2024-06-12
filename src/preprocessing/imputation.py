#%% Dependencies
import pandas as pd

from src.preprocessing.interpolate import interpolate

class Imputation():
	def __init__(self) -> None:
		self.data = pd.DataFrame()
	
	def read_data(self, path):
		self.data = pd.read_csv(path)
	
	def apply_imputation(self):
		self.data = interpolate(self.data, self.data.columns.values)
	
	def write_data(self, path):
		self.data.to_csv(path)
