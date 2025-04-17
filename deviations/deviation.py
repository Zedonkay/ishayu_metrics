import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
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

def plot_deviations_with_separate_amputations(deviations_terrain, deviations_flat, amputation_data_dict, base_filename):
    """
    Plot a violin plot for flat and each amputation trial separately and save the plot.

    Parameters:
    deviations_flat (list): List of deviations for flat terrain.
    amputation_data_dict (dict): Dictionary with amputation terrain names as keys and deviation lists as values.
    base_filename (str): The base filename for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create empty list for labels
    labels = []
    
    # Define a function to add labels
    def add_label(violin, label):
        """
        Function to add a label to the violin plot.
        Args:
            violin (matplotlib.container.ViolinContainer): The violin plot object.
            label (str): The label to be added.
        """
        color = violin["bodies"][0].get_facecolor().flatten()  # Get the color of the violin plot
        labels.append((mpatches.Patch(color=color), label))  # Append the color and label to the labels list
    
    # Calculate positions for all plots
    num_violins = 1 + len(amputation_data_dict)  # flat and each amputation trial
    positions = list(range(1, num_violins + 1))
    
    # Plot flat deviations
    v1 = ax.violinplot([deviations_flat], [positions[0]], showmeans=True, showmedians=False, showextrema=False)
    add_label(v1, "Flat")
    
    # Plot each amputation trial separately
    amputation_labels = []
    for i, (terrain_name, deviations) in enumerate(amputation_data_dict.items(), 1):
        v = ax.violinplot([deviations], [positions[i]], showmeans=True, showmedians=False, showextrema=False)
        # Use a shortened/cleaned version of the terrain name for the label
        label = terrain_name.replace('amputate', 'Amp').replace('_', ' ')
        add_label(v, label)
        amputation_labels.append(label)
    
    # Add legend
    ax.legend(*zip(*labels), loc='upper right', fontsize='small')

    # Customize the plot
    ax.set_xlabel('Terrain Type')
    ax.set_ylabel('% Deviation')
    ax.set_title('% Deviation by Terrain Type with Separate Amputation Trials')
    
    # Set x-axis ticks and labels
    ax.set_xticks(positions)
    all_labels = ["Flat"] + amputation_labels
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    
    # Set y-axis limits
    # ax.set_ylim(0.075, 0.155)  # Adjust the limits as needed

    plt.tight_layout()

    # Save the plot
    save_plot(fig, base_filename, 'violinplot_deviation_separate_amputations')

def process_csv(input_file):
    """
    Reads a CSV file, extracts deviations for each category,
    and returns data for plotting with separate amputation trials.
    """
    data = pd.read_csv(input_file)
    
    # Filter out predefined data for each category
    terrain_data = data[(data['Terrain'] == 'terrain') & (~data['Terrain'].str.contains('predefined'))]
    flat_data = data[(data['Terrain'] == 'flat') & (~data['Terrain'].str.contains('predefined'))]
    
    # Extract amputation data with separate entries for each unique terrain
    amputation_data = data[data['Terrain'].str.contains('amputate') & (~data['Terrain'].str.contains('predefined'))]
    
    # Create a dictionary to store deviations for each unique amputation terrain
    amputation_data_dict = {}
    for terrain in amputation_data['Terrain'].unique():
        # Get deviations for this specific amputation terrain
        terrain_deviations = amputation_data[amputation_data['Terrain'] == terrain]['Deviation'].tolist()
        amputation_data_dict[terrain] = terrain_deviations
    
    # Extract deviation values for terrain and flat
    deviations_terrain = terrain_data['Deviation'].tolist()
    deviations_flat = flat_data['Deviation'].tolist()
    
    return deviations_terrain, deviations_flat, amputation_data_dict
    
if __name__ == "__main__":
    input_csv = "sheets/deviation.csv"  # Adjust path as needed
    output_image = "graphs/"     # Adjust path as needed
    
    deviations_terrain, deviations_flat, amputation_data_dict = process_csv(input_csv)
    
    # Print data lengths to verify
    print(f"Terrain data points: {len(deviations_terrain)}")
    print(f"Flat data points: {len(deviations_flat)}")
    print("Amputation data points by terrain:")
    for terrain, deviations in amputation_data_dict.items():
        print(f"  {terrain}: {len(deviations)}")
    
    # Create the plot with separate violins for each amputation trial
    plot_deviations_with_separate_amputations(
        deviations_terrain, 
        deviations_flat, 
        amputation_data_dict, 
        output_image
    )