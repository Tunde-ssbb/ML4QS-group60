import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.feature_engineering.frequency import extract_frequency
import seaborn as sns
from matplotlib.lines import Line2D


def create_movement_plot(data, name_session, path):
    # Create subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    # First plot: arm acceleration
    axs[0].set_title(name_session)
    axs[0].plot(data['time'], data['arm_acc_x'], "r+", label='x')
    axs[0].plot(data['time'], data['arm_acc_y'], "b+" , label='y')
    axs[0].plot(data['time'], data['arm_acc_z'], "g+" , label='z')
    axs[0].set_ylabel("arm acceleration (m/s^2)")
    axs[0].legend(loc = 'upper left')

    #exchange 'exp_level' for what you want to color by
    """
    axs[0].set_title(name_session)
    axs[0].scatter(data['time'], data['arm_acc_x'], c=data['exp_lvl'], cmap = "viridis", label='x')
    axs[0].scatter(data['time'], data['arm_acc_y'], c=data['exp_lvl'], cmap = "viridis", label='y')
    axs[0].scatter(data['time'], data['arm_acc_z'], c=data['exp_lvl'], cmap = "viridis", label='z')
    axs[0].set_ylabel("arm acceleration (m/s^2)")
    axs[0].legend(loc = 'upper left')
    """

    # Second plot: arm rotation
    axs[1].plot(data['time'], data['arm_gyr_x'], "r+", label='x')
    axs[1].plot(data['time'], data['arm_gyr_y'], "b+", label='y')
    axs[1].plot(data['time'], data['arm_gyr_z'], "g+", label='z')
    axs[1].set_ylabel("arm rotation (rad/s)")
    axs[1].legend(loc = 'upper left')

    # Third plot: leg acceleration
    axs[2].plot(data['time'], data['leg_acc_x'], "r+", label='x')
    axs[2].plot(data['time'], data['leg_acc_y'], "b+", label='y')
    axs[2].plot(data['time'], data['leg_acc_z'], "g+", label='z')
    axs[2].set_ylabel("leg acceleration (m/s^2)")
    axs[2].legend(loc = 'upper left')

    # Fourth plot: leg rotation
    axs[3].plot(data['time'], data['leg_gyr_x'], "r+", label='x')
    axs[3].plot(data['time'], data['leg_gyr_y'], "b+", label='y')
    axs[3].plot(data['time'], data['leg_gyr_z'], "g+", label='z')
    axs[3].set_ylabel("leg rotation (rad/s)")
    axs[3].legend(loc = 'upper left')

    # Set common x-label
    axs[3].set_xlabel("time (min)")

    # Set x-ticks only for the lowest subplot
    for ax in axs[:-1]:
        ax.tick_params(labelbottom=False)

    # Format the x-ticks as integers representing minutes
    axs[3].set_xticks(np.arange(0,3001, 600))
    axs[3].set_xticklabels(range(0,6))


    p = path + "/" +  name_session + "_movement"
    plt.savefig(p)

    print(f"Saved movement plot to: {p}")

    # Show plot
    #plt.show()



def create_performance_plot(data, name_session, path):
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    colormap = 'viridis'

    axs[0].set_title(name_session, size = 20)
    scatter1 = axs[0].scatter(data['time'], data['dist'], c=data['exp_lvl'], cmap=colormap)
    axs[0].set_ylabel("distance \n (m)", fontsize = 20)
    axs[0].tick_params(labelsize = 16)

    axs[1].scatter(data['time'], data['pace'], c=data['exp_lvl'], cmap=colormap)
    axs[1].set_ylabel("pace \n (s/500m)", fontsize = 20)
    axs[1].tick_params(labelsize = 16)


    axs[2].scatter(data['time'], data['HR'], c=data['exp_lvl'], cmap=colormap)
    axs[2].set_ylabel("heartrate \n (bpm)", fontsize = 20)
    axs[2].set_xlabel("time (min)", fontsize = 20)

    axs[2].tick_params(labelsize = 16)


    for ax in axs[:-1]:
        ax.tick_params(labelbottom=False)

    # Format the x-ticks as integers representing minutes
    axs[2].set_xticks(np.arange(0,3001, 600))
    axs[2].set_xticklabels(range(0,6))

    # Add a color bar
    cbar = fig.colorbar(scatter1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('skill Level')
    
    p = path + "/" +  name_session + "_performance"
    plt.savefig(p)

    print(f"Saved performance plot to: {p}")

    #plt.show()


#distriutions

def create_distribution_plot(data, session_name, path):

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))

    fig.suptitle(session_name, fontsize = 20)

    # First plot: arm acceleration
    axs[0,0].set_title('arm acceleration', size = 20)
    axs[0,0].hist(data['arm_acc_x'], bins = 100, alpha = 0.5, label='x')
    axs[0,0].hist(data['arm_acc_y'], bins = 100, alpha = 0.5, label='y')
    axs[0,0].hist(data['arm_acc_z'], bins = 100, alpha = 0.5, label='z')
    axs[0,0].set_xlabel("acceleration (m/s^2)", fontsize=20)
    axs[0,0].set_ylabel("frequency", fontsize=20)
    axs[0,0].tick_params(labelsize=16)
    axs[0,0].legend(loc = 'upper left', fontsize = 20)

    # Second plot: arm rotation
    axs[1,0].set_title('arm rotation', size = 20)
    axs[1,0].hist(data['arm_gyr_x'], bins = 100, alpha = 0.5, label='x')
    axs[1,0].hist(data['arm_gyr_y'], bins = 100, alpha = 0.5, label='y')
    axs[1,0].hist(data['arm_gyr_z'], bins = 100, alpha = 0.5, label='z')
    axs[1,0].set_xlabel("arm rotation (rad/s)" , fontsize = 20)
    axs[1,0].set_ylabel("frequency", fontsize = 20)
    axs[1,0].tick_params(labelsize=16)
    axs[1,0].legend(loc = 'upper left', fontsize = 20)

    # Third plot: leg acceleration
    axs[0,1].set_title('leg acceleration', size = 20)
    axs[0,1].hist(data['leg_acc_x'], bins = 100, alpha = 0.5, label='x')
    axs[0,1].hist(data['leg_acc_y'], bins = 100, alpha = 0.5, label='y')
    axs[0,1].hist(data['leg_acc_z'], bins = 100, alpha = 0.5, label='z')
    axs[0,1].set_xlabel("acceleration (m/s^2)", fontsize = 20)
    axs[0,1].set_ylabel("frequency", fontsize = 20)
    axs[0,1].tick_params(labelsize=16)
    axs[0,1].legend(loc = 'upper left', fontsize = 20)

    # Fourth plot: leg rotation
    axs[1,1].set_title('leg rotation', size = 20)
    axs[1,1].hist(data['leg_gyr_x'], bins = 100, alpha = 0.5, label='x')
    axs[1,1].hist(data['leg_gyr_y'], bins = 100, alpha = 0.5, label='y')
    axs[1,1].hist(data['leg_gyr_z'], bins = 100, alpha = 0.5, label='z')
    axs[1,1].set_xlabel("rotation (rad/s)", fontsize = 20)
    axs[1,1].set_ylabel("frequency", fontsize = 20)
    axs[1,1].tick_params(labelsize=16)
    axs[1,1].legend(loc = 'upper left', fontsize = 20)

    # Set common x-label

    p = path + "/" +  session_name + "_distributions"
    plt.savefig(p)

    print(f"Saved distribution plot to: {p}")



    # Show plot
    #plt.show()

def create_temporal_frequency_plot(data, feature, path):

    data['time_numeric'] = pd.to_datetime(data['time']).view(np.int64) // 10**6

    # Extract columns that start with the feature name
    feature_columns = [col for col in data.columns if col.startswith(feature)]
    
    # Extract frequencies from the column names
    frequencies = []
    for col in feature_columns:
        freq = extract_frequency(col)
        if freq != None and freq != 0.0:
            frequencies.append(freq)
        else:
            frequencies.append(np.nan)
    
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    
    for col, freq in zip(feature_columns, frequencies):
        if not np.isnan(freq):
            plt.scatter([freq] * len(data), data[col], c=data['time_numeric'], cmap='viridis', label=f'{col}', alpha=0.5)
    
    plt.colorbar(label='Time')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Values')
    plt.title(f'Scatter plot of {feature} values by frequency')
    
    p = path + "/" +  feature + "_temporal_frequencies"
    plt.savefig(p)

    print(f"Saved frequency plot to: {p}")


def create_freq_distributions_by_experience_level_plot(data, feature_prefix, path):
    #Extract columns that start with the feature prefix
    feature_columns = [col for col in data.columns if col.startswith(feature_prefix)]
    
    if not feature_columns:
        print("No columns found with the given feature prefix.")
        return
    
    #print(f"Feature columns: {feature_columns}")

    # Get unique experience levels
    exp_levels = data['exp_lvl'].unique()
    
    # Number of subplots (at most 5)
    num_subplots = min(len(exp_levels), 5)
    
    # Create subplots
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 8 * num_subplots), sharex=True)
    if num_subplots == 1:
        axes = [axes]  # Ensure axes is iterable

    all_handles_labels = {}

    for ax, exp_lvl in zip(axes, exp_levels):
        # Filter the dataframe for the current experience level
        exp_data = data[data['exp_lvl'] == exp_lvl]
        
        for col in feature_columns:
            # Extract the frequency from the column name
            freq = extract_frequency(col)
            if freq != None:
                sns.kdeplot(exp_data[col], ax=ax, label=f'Freq {freq} Hz', fill=True)
                if f'Freq {freq} Hz' not in all_handles_labels:
                    all_handles_labels[f'Freq {freq} Hz'] = Line2D([0], [0], color=sns.color_palette()[len(all_handles_labels) % len(sns.color_palette())])

        ax.set_title(f'Distribution of {feature_prefix} values for Experience Level {exp_lvl}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    if not all_handles_labels:
        print("No valid KDE plots were created.")
        return

    # Create a single legend outside the subplots
    #handles, labels = zip(*all_handles_labels.items())
    #fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, title='Frequency')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust to fit the legend

    p = path + "/" +  feature_prefix + "_freq_dist_per_lvl"
    plt.savefig(p)

    print(f"Saved distribution plot to: {p}")
