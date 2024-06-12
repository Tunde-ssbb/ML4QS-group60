#%%
from src.preprocessing.outlier_removal import OutlierRemoval
from src.preprocessing.imputation import Imputation
import src.visualisation.data_visualisation as vis

def do_outlier_removal():
	orem = OutlierRemoval()
	orem.read_data('./measurement-data/full_data.csv')
	orem.remove_outliers()
	vis.create_distribution_plot(orem.combined_data, 'orem', './figures/without_outliers')
	orem.write_data('./measurement-data/outliers_removed/orem.csv')

def do_imputation():
	imp = Imputation()
	imp.read_data('./measurement-data/outliers_removed/orem.csv')
	imp.apply_imputation()
	vis.create_distribution_plot(imp.data, 'imp', './figures/without_nans')
	imp.write_data('./measurement-data/nans_removed/imp.csv')

def main():
	do_outlier_removal()
	do_imputation()

main()



# %%
