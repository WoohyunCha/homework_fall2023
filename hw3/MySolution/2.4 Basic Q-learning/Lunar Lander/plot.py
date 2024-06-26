import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_csv_files(directory):
    # List to store data frames
    data_frames = []

    # Iterate over all csv files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Construct full file path
            file_path = os.path.join(directory, filename)
            # Read the csv file
            df = pd.read_csv(file_path)
            # Set 'Steps' as the index if not already
            if 'Step' not in df.columns:
                print("Error: 'Step' column missing from", filename)
                continue
            df.set_index('Step', inplace=True)
            # Append the 'Value' column dataframe
            data_frames.append(df['Value'])  # Collect only 'Value' column
            # Plot each trial's return graph with faded lines
            plt.plot(df.index, df['Value'], label=f'Trial {len(data_frames)}', alpha=0.5)

    # Check if there are any data frames collected
    if data_frames:
        # Concatenate all Value columns along the columns
        combined_df = pd.concat(data_frames, axis=1)
        # Calculate the mean across all trials (mean for each row)
        mean_values = combined_df.mean(axis=1)
        # Plot the average graph with a bold line
        plt.plot(mean_values.index, mean_values, label='Average Value', linewidth=2, color='black')

    # Adding title and labels
    plt.title('Return Graphs of Trials')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

# Specify the directory containing the CSV files
directory_path = './csv/'
plot_csv_files(directory_path)
