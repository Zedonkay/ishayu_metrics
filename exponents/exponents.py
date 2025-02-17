import pandas as pd
from scipy.stats import ks_2samp
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
    mean_exponents.index.name = 'Terrain'
    return mean_exponents, std_exponents

def perform_statistical_tests(data, mean_exponents, std_exponents):
    """
    Perform Kolmogorov-Smirnov tests to compare the distributions.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and exponents.
    mean_exponents (pd.Series): Mean exponents for each terrain.
    std_exponents (pd.Series): Standard deviation of exponents for each terrain.
    
    Returns:
    pd.DataFrame: Results of the K-S tests including p-values and test statistics.
    """
    results = []
    flat_data = data[data['Terrain'] == 'flat']['Exponent']
    
    for terrain in mean_exponents.index:
        terrain_mean = mean_exponents[terrain]
        terrain_data = data[data['Terrain'] == terrain]['Exponent']
        
        if terrain != 'flat' and not terrain.startswith('predefined'):
            # K-S test between terrain and flat
            ks_stat, p_val = ks_2samp(terrain_data, flat_data)
            
            results.append({
                'Terrain': terrain,
                'Mean Exponent': terrain_mean,
                'Std Exponent': std_exponents[terrain],
                'P-value (vs flat)': p_val,
                'KS-statistic (vs flat)': ks_stat
            })
        else:
            results.append({
                'Terrain': terrain,
                'Mean Exponent': terrain_mean,
                'Std Exponent': std_exponents[terrain],
                'P-value (vs flat)': np.nan,
                'KS-statistic (vs flat)': np.nan
            })
    
    return pd.DataFrame(results)

# Null Hypothesis: The distributions of exponents for each terrain are different from the flat terrain.
# Alternative Hypothesis: The distributions of exponents for each terrain are not different from the flat terrain.

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

def plot_distributions(data, base_filename):
    """
    Plot kernel density estimates for each terrain's distribution compared to flat terrain.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and exponents.
    base_filename (str): The base filename for saving the plot.
    """
    flat_data = data[data['Terrain'] == 'flat']['Exponent']
    terrains = [t for t in data['Terrain'].unique() if t != 'flat' and not t.startswith('predefined')]
    
    for terrain in terrains:
        fig, ax = plt.subplots(figsize=(10, 6))
        terrain_data = data[data['Terrain'] == terrain]['Exponent']
        
        # Plot kernel density estimates
        flat_data.plot.kde(ax=ax, label='Flat Terrain', color='blue')
        terrain_data.plot.kde(ax=ax, label=f'{terrain} Terrain', color='red')
        
        ax.set_xlabel('Exponent')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution Comparison: {terrain} vs Flat Terrain')
        ax.legend()
        plt.tight_layout()
        save_plot(fig, base_filename, f'distributions/distribution_{terrain}_vs_flat')

def plot_all_distributions(data, base_filename):
    """
    Plot all distributions in different shades of blue and the flat distribution in red.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and exponents.
    base_filename (str): The base filename for saving the plot.
    """
    flat_data = data[data['Terrain'] == 'flat']['Exponent']
    terrains = [t for t in data['Terrain'].unique() if t != 'flat' and not t.startswith('predefined')]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot flat terrain in red
    flat_data.plot.kde(ax=ax, label='Flat Terrain', color='red')
    
    # Plot other terrains in different shades of blue
    colors = plt.cm.viridis(np.linspace(0, 1, len(terrains)))
    for terrain, color in zip(terrains, colors):
        terrain_data = data[data['Terrain'] == terrain]['Exponent']
        terrain_data.plot.kde(ax=ax, label=f'{terrain} Terrain', color=color)
    
    ax.set_xlabel('Exponent')
    ax.set_ylabel('Density')
    ax.set_title('All Terrain Distributions Compared to Flat Terrain')
    ax.legend()
    plt.tight_layout()
    save_plot(fig, base_filename, 'distributions/all_terrain_distributions')

def plot_graphs(data, mean_exponents, std_exponents, base_filename):
    """
    Plot all graphs and save the plots.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and exponents.
    mean_exponents (pd.Series): Mean exponents for each terrain.
    std_exponents (pd.Series): Standard deviation of exponents for each terrain.
    base_filename (str): The base filename for saving the plots.
    """
    plot_boxplot(data, base_filename)
    plot_scatter(data, base_filename)
    plot_distributions(data, base_filename)
    plot_all_distributions(data, base_filename)

def running(file_path, output_file, base_filename):
    """
    Main function to read data, calculate statistics, save results, and plot graphs.
    
    Parameters:
    file_path (str): The path to the CSV file containing the data.
    output_file (str): The path to save the results CSV file.
    base_filename (str): The base filename for saving the plots.
    
    Returns:
    None
    """
    data = read_data(file_path)
    mean_exponents, std_exponents = calculate_statistics(data)
    results_df = perform_statistical_tests(data, mean_exponents, std_exponents)
    results_df.to_csv(output_file, index=False)
    
    # Plot the graphs and save them
    plot_graphs(data, mean_exponents, std_exponents, base_filename)
    
    return None

def main():
    file_path = 'sheets/exponents.csv'
    output_file = 'sheets/exponents_table.csv'
    base_filename = 'graphs/'
    running(file_path, output_file, base_filename)

if __name__ == '__main__':
    main()