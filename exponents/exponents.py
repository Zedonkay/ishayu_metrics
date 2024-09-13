import pandas as pd
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
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
    data (pd.DataFrame): The data containing terrain and exponents.
    
    Returns:
    pd.Series: Mean exponents for each terrain.
    pd.Series: Standard deviation of exponents for each terrain.
    """
    mean_exponents = data.groupby('Terrain')['Exponent'].mean()
    std_exponents = data.groupby('Terrain')['Exponent'].std()
    mean_exponents.index.name = 'Terrain'  # Ensure the index name is set to 'Terrain'
    return mean_exponents, std_exponents

def perform_statistical_tests(data, mean_exponents, std_exponents):
    """
    Perform t-tests to compare the mean exponents.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and exponents.
    mean_exponents (pd.Series): Mean exponents for each terrain.
    std_exponents (pd.Series): Standard deviation of exponents for each terrain.
    
    Returns:
    pd.DataFrame: Results of the t-tests including p-values and t-statistics.
    """
    results = []
    for terrain in mean_exponents.index:
        terrain_mean = mean_exponents[terrain]
        
        if terrain != 'flat' and not terrain.startswith('predefined'):
            # T-test between terrain and flat
            t_stat_flat, p_val_flat = ttest_rel(
                data[data['Terrain'] == terrain]['Exponent'],
                data[data['Terrain'] == 'flat']['Exponent'],
            )
            
            results.append({
                'Terrain': terrain,
                'Mean Exponent': terrain_mean,
                'Std Exponent': std_exponents[terrain],
                'P-value (vs flat)': p_val_flat,
                'T-statistic (vs flat)': t_stat_flat
            })
        else:
            results.append({
                'Terrain': terrain,
                'Mean Exponent': terrain_mean,
                'Std Exponent': std_exponents[terrain],
                'P-value (vs flat)': np.nan,
                'T-statistic (vs flat)': np.nan
            })
    
    return pd.DataFrame(results)

# Null Hypothesis: The means of each of the terrains is not statistically different from flat terrain.
# Alternative Hypothesis: The means of each of the terrains is statistically different from flat terrain.




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



def plot_boxplot(data, base_filename):
    """
    Plot a box plot of exponents for each terrain and save the plot.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and exponents.
    base_filename (str): The base filename for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    data.boxplot(column='Exponent', by='Terrain', grid=False, patch_artist=True, boxprops=dict(facecolor='skyblue'), ax=ax)
    ax.set_xlabel('Terrain')
    ax.set_ylabel('Exponents')
    ax.set_title('Box Plot of Exponents by Terrain')
    plt.suptitle('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(fig, base_filename, 'boxplot/boxplot_exponents')

def plot_scatter(data, base_filename):
    """
    Plot a scatter plot of all exponents against the terrains and save the plot.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and exponents.
    base_filename (str): The base filename for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for terrain in data['Terrain'].unique():
        terrain_data = data[data['Terrain'] == terrain]
        ax.scatter(terrain_data['Terrain'], terrain_data['Exponent'], label=terrain)
    ax.set_xlabel('Terrain')
    ax.set_ylabel('Exponents')
    ax.set_title('Scatter Plot of Exponents by Terrain')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(fig, base_filename, 'scatter/scatter_exponents')

def plot_exponents(mean_exponents, std_exponents, base_filename):
    """
    Plot the mean exponents with standard deviation as error bars and save the plot.
    
    Parameters:
    mean_exponents (pd.Series): Mean exponents for each terrain.
    std_exponents (pd.Series): Standard deviation of exponents for each terrain.
    base_filename (str): The base filename for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_exponents.plot(kind='bar', yerr=std_exponents, capsize=4, color='skyblue', edgecolor='black', ax=ax)
    ax.set_xlabel('Terrain')
    ax.set_ylabel('Mean Exponents')
    ax.set_title('Mean Exponents with Standard Deviation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(fig, base_filename, 'bar_graph/mean_exponents')

def plot_graphs(data, mean_exponents, std_exponents, base_filename):
    """
    Plot all three graphs: mean exponents, box plot, and scatter plot, and save the plots.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and exponents.
    mean_exponents (pd.Series): Mean exponents for each terrain.
    std_exponents (pd.Series): Standard deviation of exponents for each terrain.
    base_filename (str): The base filename for saving the plots.
    """
    plot_exponents(mean_exponents, std_exponents, base_filename)
    plot_boxplot(data, base_filename)
    plot_scatter(data, base_filename)

def running(file_path, output_file, base_filename):
    """
    Main function to read data, calculate statistics, save results, and plot graphs.
    
    Parameters:
    file_path (str): The path to the CSV file containing the data.
    output_file (str): The path to save the results CSV file.
    base_filename (str): The base filename for saving the plots.
    
    Returns:
    pd.DataFrame: The results of the statistical calculations.
    """
    data = read_data(file_path)
    mean_exponents, std_exponents = calculate_statistics(data)
    results_df = perform_statistical_tests(data, mean_exponents, std_exponents)
    results_df.to_csv(output_file, index=False)
    
    # Plot the exponents and save the plots
    plot_graphs(data, mean_exponents,std_exponents, base_filename)
    
    return results_df

def main():
    file_path = 'sheets/exponents.csv'
    output_file = 'sheets/exponents_table.csv'
    base_filename = 'graphs/'
    results_df = running(file_path, output_file, base_filename)

if __name__ == '__main__':
    main()
