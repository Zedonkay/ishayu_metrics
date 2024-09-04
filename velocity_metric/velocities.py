import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np

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
    """
    Calculate the mean and standard deviation for each terrain.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and forward velocities.
    
    Returns:
    pd.Series: Mean forward velocities for each terrain.
    pd.Series: Standard deviation of forward velocities for each terrain.
    """
    mean_velocities = data.groupby('Terrain')['Forward Velocities'].mean()
    std_velocities = data.groupby('Terrain')['Forward Velocities'].std()
    return mean_velocities, std_velocities

def perform_statistical_tests(data, mean_velocities, std_velocities):
    """
    Perform t-tests to compare the mean forward velocities.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and forward velocities.
    mean_velocities (pd.Series): Mean forward velocities for each terrain.
    std_velocities (pd.Series): Standard deviation of forward velocities for each terrain.
    
    Returns:
    pd.DataFrame: Results of the t-tests including p-values and t-statistics.
    """
    flat_mean = mean_velocities['flat']
    predefined_means = mean_velocities.filter(like='predefined')
    
    results = []
    for terrain in mean_velocities.index:
        terrain_mean = mean_velocities[terrain]
        predefined_terrain = f'predefined_{terrain}'
        
        if terrain != 'flat' and not terrain.startswith('predefined'):
            if predefined_terrain in predefined_means.index:
                predefined_mean = predefined_means[predefined_terrain]
                
                # T-test between terrain and flat
                t_stat_flat, p_val_flat = ttest_ind(
                    data[data['Terrain'] == terrain]['Forward Velocities'],
                    data[data['Terrain'] == 'flat']['Forward Velocities']
                )
                
                # T-test between terrain and predefined terrain
                t_stat_predefined, p_val_predefined = ttest_ind(
                    data[data['Terrain'] == terrain]['Forward Velocities'],
                    data[data['Terrain'] == predefined_terrain]['Forward Velocities']
                )
                
                results.append({
                    'Terrain': terrain,
                    'Mean Forward Velocity': terrain_mean,
                    'Std Forward Velocity': std_velocities[terrain],
                    'P-value (vs flat)': p_val_flat,
                    'P-value (vs predefined)': p_val_predefined,
                    'T-statistic (vs flat)': t_stat_flat,
                    'T-statistic (vs predefined)': t_stat_predefined
                })
        else:
            results.append({
                'Terrain': terrain,
                'Mean Forward Velocity': terrain_mean,
                'Std Forward Velocity': std_velocities[terrain],
                'P-value (vs flat)': np.nan,
                'P-value (vs predefined)': np.nan,
                'T-statistic (vs flat)': np.nan,
                'T-statistic (vs predefined)': np.nan
            })
    
    return pd.DataFrame(results)



def save_plot(fig, base_filename, title):
    """
    Save the plot as PNG and SVG files.
    
    Parameters:
    fig (matplotlib.figure.Figure): The figure object to save.
    base_filename (str): The base filename for saving the plot.
    title (str): The title to add to the filename.
    """
    png_filename = f"{base_filename}{title}.png"
    svg_filename = f"{base_filename}{title}.svg"
    fig.savefig(png_filename)
    fig.savefig(svg_filename)
    fig.clf()
    fig.clear()

def plot_velocities(mean_velocities, std_velocities, base_filename):
    """
    Plot the mean forward velocities with standard deviation as error bars and save the plot.
    
    Parameters:
    mean_velocities (pd.Series): Mean forward velocities for each terrain.
    std_velocities (pd.Series): Standard deviation of forward velocities for each terrain.
    base_filename (str): The base filename for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_velocities.plot(kind='bar', yerr=std_velocities, capsize=4, color='skyblue', edgecolor='black', ax=ax)
    ax.set_xlabel('Terrain')
    ax.set_ylabel('Mean Forward Velocity')
    ax.set_title('Mean Forward Velocities with Standard Deviation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(fig, base_filename, 'bar graph/mean_forward_velocities')

def plot_boxplot(data, base_filename):
    """
    Plot a box plot of forward velocities for each terrain and save the plot.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and forward velocities.
    base_filename (str): The base filename for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    data.boxplot(column='Forward Velocities', by='Terrain', grid=False, patch_artist=True, boxprops=dict(facecolor='skyblue'), ax=ax)
    ax.set_xlabel('Terrain')
    ax.set_ylabel('Forward Velocities')
    ax.set_title('Box Plot of Forward Velocities by Terrain')
    plt.suptitle('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(fig, base_filename, 'boxplot/boxplot_forward_velocities')

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
    mean_velocities, std_velocities = calculate_statistics(data)
    results_df = perform_statistical_tests(data, mean_velocities, std_velocities)
    results_df.to_csv(output_file, index=False)
    
    # Plot the velocities and save the plots
    plot_velocities(mean_velocities, std_velocities, base_filename)
    plot_boxplot(data, base_filename)
    
    return results_df

def main():
    file_path = 'sheets/filtered_velocities.csv'
    output_file = 'sheets/filtered_velocities_table.csv'
    base_filename = 'graphs/filtered/'
    results_df = running(file_path, output_file, base_filename)
    file_path = 'sheets/velocities.csv'
    output_file = 'sheets/velocities_table.csv'
    base_filename = 'graphs/all/'
    results_df = running(file_path, output_file, base_filename)

if __name__ == '__main__':
    main()

    
