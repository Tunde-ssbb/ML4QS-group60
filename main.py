#%%
from src.preprocessing.outlier_removal import OutlierRemoval
import src.visualisation.data_visualisation as vis

def do_outlier_removal():
	orem = OutlierRemoval()
	orem.read_data()
	orem.remove_outliers()
	vis.create_distribution_plot(orem.combined_data, 'orem', './figures/without_outliers')

def main():
	do_outlier_removal()

main()



# %%
