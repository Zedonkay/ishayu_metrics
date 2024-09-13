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
     # Sort the data according to the terrain list
    terrain_list = ['flat', 'predefined_flat', 'amputate_L3',"predefined_amputate_L3" , 'amputate_R2', 'predefined_amputate_R2',
                    'terrain','predefined_terrain',"amputate_R2_L2","predefined_amputate_R2_L2","amputate_R3_L3","predefined_amputate_R3_L3"]  # Replace with your desired terrain list
    data['Terrain'] = pd.Categorical(data['Terrain'], categories=terrain_list, ordered=True)
    data.sort_values('Terrain', inplace=True)
    means = data.groupby('Terrain')['Yaw Rate'].mean()
    stds = data.groupby('Terrain')['Yaw Rate'].std()
    
   
    
    return data, means, stds

def perform_statistical_tests(data, mean_yaws, std_yaws):
    """
    Perform Mann-Whitney U tests to compare the mean forward yaws.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and forward yaws.
    mean_yaws (pd.Series): Mean forward yaws for each terrain.
    std_yaws (pd.Series): Standard deviation of forward yaws for each terrain.
    
    Returns:
    pd.DataFrame: Results of the tests including p-values and U-statistics.
    """
    flat_mean = mean_yaws['flat']
    predefined_means = mean_yaws.filter(like='predefined')
    
    results = []
    for terrain in mean_yaws.index:
        print(f"terrain: {terrain}")
        terrain_mean = mean_yaws[terrain]
        predefined_terrain = f'predefined_{terrain}'
        
        if not terrain.startswith('predefined'):
                if terrain != 'flat':
                    # Mann-Whitney U test between terrain and flat
                    u_stat_flat, p_val_flat = mannwhitneyu(
                        data[data['Terrain'] == terrain]['Yaw Rate'],
                        data[data['Terrain'] == 'flat']['Yaw Rate'],
                        alternative='greater'
                    )
                else:
                    p_val_flat = np.nan
                    u_stat_flat = np.nan
                
                # Mann-Whitney U test between terrain and predefined terrain
                u_stat_predefined, p_val_predefined = mannwhitneyu(
                        data[data['Terrain'] == terrain]['Yaw Rate'],
                        data[data['Terrain'] == predefined_terrain]['Yaw Rate'],
                        alternative='less'
                    )
                
                results.append({
                    'Terrain': terrain,
                    'Mean Forward yaw': terrain_mean,
                    'Std Forward yaw': std_yaws[terrain],
                    'P-value (vs flat)': p_val_flat,
                    'P-value (vs predefined)': p_val_predefined,
                    'U-statistic (vs flat)': u_stat_flat,
                    'U-statistic (vs predefined)': u_stat_predefined
                })
        else:
            results.append({
                'Terrain': terrain,
                'Mean Forward yaw': terrain_mean,
                'Std Forward yaw': std_yaws[terrain],
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
def plot_violinplot(yaws_neural, yaws_predefined, terrains, base_filename):
    """
    Plot a violin plot of percentage yaws for each terrain and its predefined version, and save the plot.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and percentage yaws.
    base_filename (str): The base filename for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate predefined and normal terrains
    add_label(ax.violinplot(yaws_neural, side='low', showmeans=True, showmedians=False, showextrema=False), "Neural")
    add_label(ax.violinplot(yaws_predefined, side='high', showmeans=True, showmedians=False, showextrema=False), "Predefined")
    ax.legend(*zip(*labels), loc=4)
    
    # Customize the plot
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Mean Forward Yaw')
    ax.set_title('Mean Forward Yaw for each experiment')
    set_axis_style(ax, terrains)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    
    # Set y-axis limits
    ax.set_ylim(0, 1.1)  # Adjust the limits as needed
    
    # Save the plot
    save_plot(fig, base_filename, 'violinplot/yaw_percent_flat')



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
    data, mean_yaws, std_yaws = calculate_statistics(data)
    results_df = perform_statistical_tests(data, mean_yaws, std_yaws)
    results_df.to_csv(output_file, index=False)
    
    # Plot the yaws and save the plot
    yaws_neural = [data[data['Terrain'] == terrain]['Yaw Rate'] for terrain in data['Terrain'].unique() if not terrain.startswith('predefined') and 'flat' not in terrain]
    yaws_predefined = [data[data['Terrain'] == terrain]['Yaw Rate'] for terrain in data['Terrain'].unique() if terrain.startswith('predefined') and 'flat' not in terrain]
    terrains = data[data['Terrain'].str.startswith('predefined') == False]['Terrain'].unique().tolist()
    terrains = [terrain for terrain in terrains if 'flat' not in terrain]
    plot_violinplot(yaws_neural, yaws_predefined, terrains, base_filename)
    return results_df
'Velocity % of Flat'
def main():
    file_path = 'sheets/yaws.csv'
    output_file = 'sheets/yaw_table.csv'
    base_filename = 'graphs/all/'
    results_df = running(file_path, output_file, base_filename)

if __name__ == '__main__':
    main()