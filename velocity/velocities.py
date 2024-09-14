import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.patches as mpatches

def read_data(file_path):
    """
    Read the CSV data into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file containing the data.
    
    Returns:
    pd.DataFrame: The data read from the CSV file.
    """
    return pd.read_csv(file_path)

def calculate_statistics(data):
    flat_mean_velocities = np.mean(data[data['Terrain'] == 'flat']['Forward Velocities'])
    predefined_flat_mean_velocities = np.mean(data[data['Terrain'] == 'predefined_flat']['Forward Velocities'])
     # Sort the data according to the terrain list
    terrain_list = ['flat', 'predefined_flat', 'terrain','predefined_terrain', 'amputate_L3',"predefined_amputate_L3" , 
                    'amputate_R2', 'predefined_amputate_R2', "amputate_R2_L2","predefined_amputate_R2_L2","amputate_R3_L3","predefined_amputate_R3_L3"]  # Replace with your desired terrain list
    data['Terrain'] = pd.Categorical(data['Terrain'], categories=terrain_list, ordered=True)
    data.sort_values('Terrain', inplace=True)

    def percent_decrease(row):
        if row['Terrain'].startswith('predefined'):
            return 100 * (1 - (row['Forward Velocities'] / predefined_flat_mean_velocities))
        else:
            return 100 * (1 - (row['Forward Velocities'] / flat_mean_velocities))
    data['Velocity % of Flat'] = data.apply(percent_decrease, axis=1)
    means = data.groupby('Terrain')['Velocity % of Flat'].mean()
    stds = data.groupby('Terrain')['Velocity % of Flat'].std()
    
   
    
    return data, means, stds

def perform_statistical_tests(data, mean_velocities, std_velocities):
    """
    Perform Mann-Whitney U tests to compare the mean forward velocities.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and forward velocities.
    mean_velocities (pd.Series): Mean forward velocities for each terrain.
    std_velocities (pd.Series): Standard deviation of forward velocities for each terrain.
    
    Returns:
    pd.DataFrame: Results of the tests including p-values and U-statistics.
    """
    flat_mean = mean_velocities['flat']
    predefined_means = mean_velocities.filter(like='predefined')
    
    results = []
    for terrain in mean_velocities.index:
        print(f"terrain: {terrain}")
        terrain_mean = mean_velocities[terrain]
        predefined_terrain = f'predefined_{terrain}'
        
        if not terrain.startswith('predefined'):
                if terrain != 'flat':
                    # Mann-Whitney U test between terrain and flat
                    u_stat_flat, p_val_flat = mannwhitneyu(
                        data[data['Terrain'] == terrain]['Velocity % of Flat'],
                        data[data['Terrain'] == 'flat']['Velocity % of Flat'],
                        alternative='greater'
                    )
                else:
                    p_val_flat = np.nan
                    u_stat_flat = np.nan
                
                # Mann-Whitney U test between terrain and predefined terrain
                u_stat_predefined, p_val_predefined = mannwhitneyu(
                        data[data['Terrain'] == terrain]['Velocity % of Flat'],
                        data[data['Terrain'] == predefined_terrain]['Velocity % of Flat'],
                        alternative='less'
                    )
                
                results.append({
                    'Terrain': terrain,
                    'Mean Forward Velocity': terrain_mean,
                    'Std Forward Velocity': std_velocities[terrain],
                    'P-value (vs flat)': p_val_flat,
                    'P-value (vs predefined)': p_val_predefined,
                    'U-statistic (vs flat)': u_stat_flat,
                    'U-statistic (vs predefined)': u_stat_predefined
                })
        else:
            results.append({
                'Terrain': terrain,
                'Mean Forward Velocity': terrain_mean,
                'Std Forward Velocity': std_velocities[terrain],
                'P-value (vs flat)': np.nan,
                'P-value (vs predefined)': np.nan,
                'U-statistic (vs flat)': np.nan,
                'U-statistic (vs predefined)': np.nan
            })
    
    return pd.DataFrame(results)

def save_plot(fig, base_filename, title):
    """
    Save the plot as PNG and SVG files.
    
    Parameters:
    fig (matplotlib.figure.Figure): The figure object to save.
    base_filename (str): The base filename for saving theaplot.
    title (str): The title to add to the filename.
    """
    png_filename = f"{base_filename}{title}.png"
    svg_filename = f"{base_filename}{title}.svg"
    fig.savefig(png_filename)
    fig.savefig(svg_filename)
    fig.clf()
    fig.clear()

labels = []
def add_label(violin, label):
    """
    Function to add a label to the violin plot.
    Args:
        violin (matplotlib.container.ViolinPlot): The violin plot object.
        label (str): The label to be added.
    """
    color = violin["bodies"][0].get_facecolor().flatten()  # Get the color of the violin plot
    labels.append((mpatches.Patch(color=color), label))  # Append the color and label to the labels list

def set_axis_style(ax, labels):
    """
    Function to set the style of the x-axis.
    Args:
        ax (matplotlib.axes.Axes): The axes object.
        labels (list): List of labels for the x-axis.
    """
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)  # Set the tick positions and labels for the x-axis
    ax.set_xlim(0.25, len(labels) + 0.75)  # Set the limits of the x-axis
def plot_violinplot(velocities_neural, velocities_predefined, terrains, base_filename):
    """
    Plot a violin plot of percentage velocities for each terrain and its predefined version, and save the plot.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and percentage velocities.
    base_filename (str): The base filename for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate predefined and normal terrains
    add_label(ax.violinplot(velocities_neural, side='low', showmeans=True, showmedians=False, showextrema=False), "D-NLC")
    add_label(ax.violinplot(velocities_predefined, side='high', showmeans=True, showmedians=False, showextrema=False), "Predefined")
    ax.legend(*zip(*labels), loc=4)
    
    # Customize the plot
    ax.set_xlabel('Experiment')
    ax.set_ylabel('% Decrease compared to Neural Flat')
    ax.set_title('% Decrease of Forward Velocity Compared to Flat Terrain')
    set_axis_style(ax, terrains)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add a second y-axis on the right side
    ax2 = ax.twinx()
    ax2.set_ylabel('% Decrease compared to Predefined Flat')
    
    # Set y-axis limits
    ax.set_ylim(0, 130)  # Adjust the limits as needed
    ax2.set_ylim(0, 130)  # Adjust the limits as needed
    
    # Adjust the plot layout to accommodate the second y-axis
    fig.subplots_adjust(right=0.85)
    
    # Save the plot
    save_plot(fig, base_filename, 'violinplot/velocity_percent_flat')



def running(file_path, output_file, base_filename):
    """
    Main function to read data, calculate statistics, perform statistical tests, save results, and plot graphs.
    
    Parameters:
    file_path (str): The path to the CSV file containing the data.
    output_file (str): The path to save the results CSV file.
    base_filename (str): The base filename for saving the plots.
    
    Returns:
    pd.DataFrame: The results of the statistical tests.
    """
    data = read_data(file_path)
    data, mean_velocities, std_velocities = calculate_statistics(data)
    results_df = perform_statistical_tests(data, mean_velocities, std_velocities)
    results_df.to_csv(output_file, index=False)
    
    # Plot the velocities and save the plot
    velocities_neural = [data[data['Terrain'] == terrain]['Velocity % of Flat'] for terrain in data['Terrain'].unique() if not terrain.startswith('predefined') and 'flat' not in terrain]
    velocities_predefined = [data[data['Terrain'] == terrain]['Velocity % of Flat'] for terrain in data['Terrain'].unique() if terrain.startswith('predefined') and 'flat' not in terrain]
    terrains = data[data['Terrain'].str.startswith('predefined') == False]['Terrain'].unique().tolist()
    terrains = [terrain for terrain in terrains if 'flat' not in terrain]
    plot_violinplot(velocities_neural, velocities_predefined, terrains, base_filename)
    return results_df

def main():
    file_path = 'sheets/velocities.csv'
    output_file = 'sheets/velocities_table.csv'
    base_filename = 'graphs/all/'
    results_df = running(file_path, output_file, base_filename)

if __name__ == '__main__':
    main()