#%%
from src.preprocessing.outlier_removal import OutlierRemoval
def main():
	orem = OutlierRemoval()
	orem.read_data()
	orem.remove_outliers()
	orem.visualize_data(after_removal=True)

main()

# %%
