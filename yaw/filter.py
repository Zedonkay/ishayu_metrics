import pandas as pd

# Function to detect outliers using the IQR method
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Read the CSV file
df = pd.read_csv('sheets/velocities.csv')

# Group the data by the 'Terrain' column
grouped = df.groupby('Terrain')

# Create a new dataframe to store the filtered data
filtered_df = pd.DataFrame()

# Iterate through each group and filter out the outliers
for name, group in grouped:
    filtered_group = detect_outliers(group, 'Forward Velocities')
    filtered_df = pd.concat([filtered_df, filtered_group])

# Save the filtered dataframe to a new CSV file
filtered_df.to_csv('sheets/filtered_velocities.csv', index=False)

print("Filtered data saved to 'filtered_velocities.csv'")
