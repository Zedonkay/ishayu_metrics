import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import ast
from scipy.stats import mannwhitneyu

def read_data(file_path):
    """
    Read the CSV data into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file containing the data.
    
    Returns:
    pd.DataFrame: The data read from the CSV file.
    """
    return pd.read_csv(file_path)

def calculate_statistics(df):
    """
    Calculate statistics for velocity data.

    Args:
        df (pandas.DataFrame): DataFrame containing velocity data.

    Returns:
        tuple: A tuple containing the following:
            - df (pandas.DataFrame): The original DataFrame with an additional column '% of Flat Velocities' 
              containing normalized velocities.
            - means (pandas.Series): Series containing the mean values for each '% of Flat Velocities'.
            - stds (pandas.Series): Series containing the standard deviation values for each '% of Flat Velocities'.
    """
    # Function to safely convert strings to lists
    def safe_eval(value, row):
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
        else:
            # If the value is numeric, print the relevant details
            print(f"Numeric value found in 'Forward Velocities' on Date: {row['Date']}, Terrain: {row['Terrain']}, Trial: {row['Trial']}")
            return value

    # Convert the 'Forward Velocities' column if necessary
    df['Forward Velocities'] = df.apply(lambda row: safe_eval(row['Forward Velocities'], row), axis=1)

    # Group by 'Terrain' and concatenate 'Forward Velocities' lists
    df = df.groupby('Terrain').agg({
        'Forward Velocities': lambda x: sum(x, []),
        'Date': 'first',  # Keep the first date for each terrain
        'Trial': 'first'  # Keep the first trial for each terrain
    }).reset_index()

    # Calculate the mean velocity for the flat terrain
    flat_mean_velocity = df[df['Terrain'] == 'flat']['Forward Velocities'].explode().mean()
    predefined_flat_mean_velocity = df[df['Terrain'] == 'predefined_flat']['Forward Velocities'].explode().mean()

    # Function to normalize velocities
    def normalize(row):
        velocities = row['Forward Velocities']
        if isinstance(velocities, list):
            if row['Terrain'].startswith('predefined'):
                normalized_velocities = [(v / predefined_flat_mean_velocity) * 100 for v in velocities]
            else:
                normalized_velocities = [(v / flat_mean_velocity) * 100 for v in velocities]
            return normalized_velocities
        else:
            # Handle as a single value if it's not a list
            return (velocities / predefined_flat_mean_velocity if row['Terrain'].startswith('predefined') 
                    else velocities / flat_mean_velocity) * 100

    # Apply the normalization function to each row
    df['Velocity % of Flat'] = df.apply(normalize, axis=1)

    # Calculate mean and standard deviation for each % of Flat Velocities
    means = df['Velocity % of Flat'].apply(lambda x: pd.Series(x).mean() if isinstance(x, list) else x)
    stds = df['Velocity % of Flat'].apply(lambda x: pd.Series(x).std() if isinstance(x, list) else 0)
    means.index = df['Terrain']

    return df, means, stds

def perform_statistical_tests(data, mean_velocities, std_velocities):
    """
    Perform paired permutation tests to compare the percentage velocities.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and percentage velocities.
    mean_velocities (pd.Series): Mean percentage velocities for each terrain.
    std_velocities (pd.Series): Standard deviation of percentage velocities for each terrain.
    
    Returns:
    pd.DataFrame: Results of the tests including U values and p-values.
    """
    results = []
    for terrain in mean_velocities.index:
        predefined_terrain = f'predefined_{terrain}'
        print(f"terrain = {terrain}, type: {type(terrain)}")
        if not terrain.startswith('predefined'):
            if predefined_terrain in mean_velocities.index:
                if terrain != "flat":
                    # Ensure data is numeric and handle NaNs
                    flat_data = pd.to_numeric(data[data['Terrain'] == 'flat']['Velocity % of Flat'].explode(), errors='coerce').dropna()
                    terrain_data = pd.to_numeric(data[data['Terrain'] == terrain]['Velocity % of Flat'].explode(), errors='coerce').dropna()
                    u_val_flat, p_val_flat = mannwhitneyu(flat_data, terrain_data, nan_policy='omit')
                else:
                    u_val_flat, p_val_flat = np.nan, np.nan
                # Ensure data is numeric and handle NaNs
                terrain_data = pd.to_numeric(data[data['Terrain'] == terrain]['Velocity % of Flat'].explode(), errors='coerce').dropna()
                predefined_data = pd.to_numeric(data[data['Terrain'] == predefined_terrain]['Velocity % of Flat'].explode(), errors='coerce').dropna()
                u_val_predefined, p_val_predefined = mannwhitneyu(terrain_data, predefined_data, nan_policy='omit')
                
                results.append({
                    'Terrain': terrain,
                    'Mean Velocity % of Flat': mean_velocities[terrain],
                    'Std Velocity % of Flat': std_velocities.get(terrain, np.nan),  # Use get to avoid KeyError
                    'U-value (vs flat)': u_val_flat,
                    'P-value (vs flat)': p_val_flat,
                    'U-value (vs predefined)': u_val_predefined,
                    'P-value (vs predefined)': p_val_predefined
                })
        else:
            if 'flat' not in terrain:
                flat_data = pd.to_numeric(data[data['Terrain'] == 'predefined_flat']['Velocity % of Flat'].explode(), errors='coerce').dropna()
                terrain_data = pd.to_numeric(data[data['Terrain'] == terrain]['Velocity % of Flat'].explode(), errors='coerce').dropna()
                u_val_flat, p_val_flat = mannwhitneyu(flat_data, terrain_data, nan_policy='omit')
                results.append({
                    'Terrain': terrain,
                    'Mean Velocity % of Flat': mean_velocities[terrain],
                    'Std Velocity % of Flat': std_velocities.get(terrain, np.nan),  # Use get to avoid KeyError
                    'U-value (vs flat)': u_val_flat,
                    'P-value (vs flat)': p_val_flat,
                    'U-value (vs predefined)': np.nan,
                    'P-value (vs predefined)': np.nan
                })
            else:
                results.append({
                    'Terrain': terrain,
                    'Mean Velocity % of Flat': mean_velocities[terrain],
                    'Std Velocity % of Flat': std_velocities.get(terrain, np.nan),  # Use get to avoid KeyError
                    'U-value (vs flat)': np.nan,
                    'P-value (vs flat)': np.nan,
                    'U-value (vs predefined)': np.nan,
                    'P-value (vs predefined)': np.nan
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

def plot_velocities(mean_velocities, std_velocities, base_filename):
    """
    Plot the mean percentage velocities with standard deviation as error bars and save the plot.
    
    Parameters:
    mean_velocities (pd.Series): Mean percentage velocities for each terrain.
    std_velocities (pd.Series): Standard deviation of percentage velocities for each terrain.
    base_filename (str): The base filename for saving the plot.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Separate predefined and non-predefined terrains
    predefined_mask = mean_velocities.index.str.startswith('predefined')
    non_predefined_mask = ~predefined_mask
    
    # Plot predefined terrains
    mean_velocities_predefined = mean_velocities[predefined_mask]
    std_velocities_predefined = std_velocities[predefined_mask]
    
    # Plot positive and negative bars separately for predefined terrains
    mean_velocities_predefined_positive = mean_velocities_predefined[mean_velocities_predefined >= 0]
    mean_velocities_predefined_negative = mean_velocities_predefined[mean_velocities_predefined < 0]
    
    mean_velocities_predefined_positive.plot(kind='bar', yerr=std_velocities_predefined[mean_velocities_predefined_positive.index], capsize=4, color='purple', edgecolor='black', ax=ax1, position=1, width=0.4)
    mean_velocities_predefined_negative.plot(kind='bar', yerr=std_velocities_predefined[mean_velocities_predefined_negative.index], capsize=4, color='purple', edgecolor='black', ax=ax1, position=1, width=0.4)
    
    # Create a second y-axis for non-predefined terrains
    ax2 = ax1.twinx()
    mean_velocities[non_predefined_mask].plot(kind='bar', yerr=std_velocities[non_predefined_mask], capsize=4, color='orange', edgecolor='black', ax=ax2, position=0, width=0.4)
    
    # Set labels and title
    ax1.set_xlabel('Terrain')
    ax1.set_ylabel('Mean Velocity % of Flat (Predefined)')
    ax2.set_ylabel('Mean Velocity % of Flat (Non-Predefined)')
    ax1.set_title('Mean Velocity % of Flat with Standard Deviation')
    
    # Adjust x-ticks to show both predefined and non-predefined terrains
    ax1.set_xticks(range(len(mean_velocities)))
    ax1.set_xticklabels(mean_velocities.index, rotation=45)
    
    plt.tight_layout()
    save_plot(fig, base_filename, 'bar graph/mean_velocity_percent_flat')
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
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Separate predefined and normal terrains
    add_label(ax.violinplot(velocities_neural, side='high', showmeans=False, showmedians=True, showextrema=False), "Neural")
    add_label(ax.violinplot(velocities_predefined, side='low', showmeans=False, showmedians=True, showextrema=False), "Predefined")
    ax.legend(*zip(*labels), loc=9)
    
    # Customize the plot
    ax.set_xlabel('Terrain')
    ax.set_ylabel('Velocity % of Predefined Flat')
    ax.set_title('Violin Plot of Velocity % of Flat by Terrain')
    set_axis_style(ax, terrains)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add a second y-axis on the right side
    ax2 = ax.twinx()
    ax2.set_ylabel('Velocity % of Neural Flat')
    
    # Set y-axis limits
    ax.set_ylim(0, 100)  # Adjust the limits as needed
    ax2.set_ylim(0, 100)  # Adjust the limits as needed
    
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
    # desired_order  =['amputate_L2','amputate_R2','amputate_R2_L2','amputate_R3_L3','terrain','flat','predefined_amputate_L2','predefined_amputate_R2','predefined_amputate_R2_L2','predefined_amputate_R3_L3','predefined_terrain','predefined_flat']
    # results_df['Terrain']=pd.Categorical(results_df['Terrain'],categories=desired_order,ordered=True)
    # results_df= results_df.sort_values('Terrain')
    results_df.to_csv(output_file, index=False)
    
    # Plot the velocities and save the plots
    #plot_velocities(mean_velocities, std_velocities, base_filename)
    velocities_neural = data[data['Terrain'].str.startswith('predefined') == False]['Velocity % of Flat'].tolist()
    velocities_predefined = data[data['Terrain'].str.startswith('predefined')]['Velocity % of Flat'].tolist()
    terrains = data[data['Terrain'].str.startswith('predefined') == False]['Terrain'].tolist()
    plot_violinplot(velocities_neural, velocities_predefined, terrains, base_filename)
    
    return results_df

def main():
    file_path = 'sheets/velocities.csv'
    output_file = 'sheets/velocities_table.csv'
    base_filename = 'graphs/all/'
    results_df = running(file_path, output_file, base_filename)

if __name__ == '__main__':
    main()