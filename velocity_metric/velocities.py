import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    # Calculate the mean velocity for the flat terrain
    flat_mean_velocity = df[df['Terrain'] == 'flat']['Forward Velocities'].apply(eval).explode().mean()
    predefined_flat_mean_velocity = df[df['Terrain'] == 'predefined_flat']['Forward Velocities'].apply(eval).explode().mean()

    # Function to normalize velocities
    def normalize(row):
        velocities = eval(row['Forward Velocities'])
        if row['Terrain'].startswith('predefined'):
            normalized_velocities = [(v / predefined_flat_mean_velocity) * 100 for v in velocities]
        else:
            normalized_velocities = [(v / flat_mean_velocity) * 100 for v in velocities]
        return normalized_velocities

    # Apply the normalization function to each row
    df['% of Flat Velocities'] = df.apply(normalize, axis=1)

    # Calculate mean and standard deviation for each % of Flat Velocities
    means = df['% of Flat Velocities'].apply(lambda x: pd.Series(x).mean())
    stds = df['% of Flat Velocities'].apply(lambda x: pd.Series(x).std())

    return df, means, stds

def paired_permutation_test(x, y, num_permutations=100000):
    """
    Perform a paired permutation test.
    
    Parameters:
    x (array-like): The first sample.
    y (array-like): The second sample.
    num_permutations (int): The number of permutations to perform.
    
    Returns:
    float: The p-value of the test.
    """
    observed_diff = np.mean(x) - np.mean(y)
    combined = np.concatenate([x, y])
    count = 0
    
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        perm_x = combined[:len(x)]
        perm_y = combined[len(x):]
        perm_diff = np.mean(perm_x) - np.mean(perm_y)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    
    p_value = count / num_permutations
    return p_value

def perform_statistical_tests(data, mean_velocities, std_velocities):
    """
    Perform paired permutation tests to compare the percentage velocities.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and percentage velocities.
    mean_velocities (pd.Series): Mean percentage velocities for each terrain.
    std_velocities (pd.Series): Standard deviation of percentage velocities for each terrain.
    
    Returns:
    pd.DataFrame: Results of the tests including p-values.
    """
    results = []
    for terrain in mean_velocities.index:
        predefined_terrain = f'predefined_{terrain}'
        
        if not terrain.startswith('predefined'):
            if predefined_terrain in mean_velocities.index:
                if terrain!="flat":
                    # Paired permutation test between terrain and flat
                    p_val_flat = paired_permutation_test(
                        data[data['Terrain'] == 'flat']['Velocity % of Flat'],
                        data[data['Terrain'] == terrain]['Velocity % of Flat']
                    )
                else:
                    p_val_flat = np.nan
                # Paired permutation test between terrain and predefined terrain
                p_val_predefined = paired_permutation_test(
                    data[data['Terrain'] == terrain]['Velocity % of Flat'],
                    data[data['Terrain'] == predefined_terrain]['Velocity % of Flat']
                )
                
                results.append({
                    'Terrain': terrain,
                    'Mean Velocity % of Flat': mean_velocities[terrain],
                    'Std Velocity % of Flat': std_velocities[terrain],
                    'P-value (vs flat)': p_val_flat,
                    'P-value (vs predefined)': p_val_predefined
                })
        else:
            if 'flat' not in terrain:
                p_val_flat = paired_permutation_test(
                        data[data['Terrain'] == 'predefined_flat']['Velocity % of Flat'],
                        data[data['Terrain'] == terrain]['Velocity % of Flat']
                    )
                results.append({
                    'Terrain': terrain,
                    'Mean Velocity % of Flat': mean_velocities[terrain],
                    'Std Velocity % of Flat': std_velocities[terrain],
                    'P-value (vs flat)': p_val_flat,
                    'P-value (vs predefined)': np.nan
                })

            else:
                results.append({
                    'Terrain': terrain,
                    'Mean Velocity % of Flat': mean_velocities[terrain],
                    'Std Velocity % of Flat': std_velocities[terrain],
                    'P-value (vs flat)': np.nan,
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
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
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


def plot_violinplot(data, base_filename):
    """
    Plot a violin plot of percentage velocities for each terrain and its predefined version, and save the plot.
    
    Parameters:
    data (pd.DataFrame): The data containing terrain and percentage velocities.
    base_filename (str): The base filename for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Separate predefined and normal terrains
    data['Type'] = data['Terrain'].apply(lambda x: 'Predefined' if x.startswith('predefined_') else 'Normal')
    data['Terrain'] = data['Terrain'].apply(lambda x: x.replace('predefined_', ''))
    
    # Plot violin plots
    sns.violinplot(x='Terrain', y='Velocity % of Flat', hue='Type', data=data, split=True, inner='box', ax=ax, palette='muted')
    
    # Customize the plot
    ax.set_xlabel('Terrain')
    ax.set_ylabel('Velocity % of Flat')
    ax.set_title('Violin Plot of Velocity % of Flat by Terrain')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
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
    plot_violinplot(data, base_filename)
    
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