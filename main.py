#%%
import pandas as pd

from src.preprocessing.outlier_removal import OutlierRemoval
from src.preprocessing.imputation import Imputation
import src.visualisation.data_visualisation as vis
import src.feature_engineering.do_fe as dfe

def do_outlier_removal():
	orem = OutlierRemoval()
	orem.read_data('./measurement-data/full_data_time.csv')
	orem.remove_outliers()
	vis.create_distribution_plot(orem.combined_data, 'orem', './figures/without_outliers')
	orem.write_data('./measurement-data/outliers_removed/orem.csv')

def do_imputation():
	imp = Imputation()
	imp.read_data('./measurement-data/outliers_removed/orem.csv')
	imp.apply_imputation()
	vis.create_distribution_plot(imp.data, 'imp', './figures/without_nans')
	imp.write_data('./measurement-data/without_nans/fouried_data.csv')

def do_feature_engineering():
	data = pd.read_csv('./measurement-data/without_nans/fouried_data.csv', index_col='Unnamed: 0')
	data_dfe = dfe.feature_engineer(data)
	#vis.create_distribution_plot(data_dfe, 'dfe', './figures/with_features')
	data_dfe.to_csv('./src/machine_learning/fouried_data.csv')


def main():
	do_outlier_removal()
	do_imputation()
	do_feature_engineering()

main()



# %%
