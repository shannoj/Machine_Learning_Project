import torch
import yfinance as yf
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import math

# Set seaborn style
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
Colour_Palette = ['#01BEFE', '#FF7D00', '#FFDD00', '#FF006D', '#ADFF02', '#8F00FF']
sns.set_palette(sns.color_palette(Colour_Palette))

# Download stock data (Apple)
end_date = date.today().strftime("%Y-%m-%d")
start_date = '1990-01-01'

df = yf.download('AAPL', start=start_date, end=end_date)

# Function to plot data
def data_plot(df, columns=['Open']):
    # Check if the specified columns are present in the DataFrame
    if not all(col in df.columns for col in columns):
        raise ValueError(f"Some specified columns are not in the DataFrame: {columns}")
    
    # Plot line charts for selected columns
    df_plot = df[columns].copy()  # Only select specific columns if needed

    # If there are no columns to plot, raise an error
    if df_plot.empty:
        raise ValueError("No data available to plot. The DataFrame is empty.")

    # Calculate number of rows and columns for subplots
    ncols = 2
    nrows = int(np.ceil(df_plot.shape[1] / ncols))  # Use np.ceil to ensure at least 1 row if necessary
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(14, 7))

    for i, ax in enumerate(fig.axes):
        if i < df_plot.shape[1]:  # Avoid out of bounds error if there are fewer than expected subplots
            sns.lineplot(data=df_plot.iloc[:, i], ax=ax)
            ax.tick_params(axis="x", rotation=30, labelsize=10, length=0)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    fig.tight_layout()
    plt.show()

# Plot the 'Open' price data (you can add more columns here if needed)
data_plot(df, columns=['Open'])

# Train-test split
training_data_len = math.ceil(len(df) * 0.8)
print(f'Training data length: {training_data_len}')

# Splitting the dataset into training and test sets based on the 'Open' price
train_data = df[:training_data_len]['Open'].values  # Use the 'Open' column directly
test_data = df[training_data_len:]['Open'].values
print(f'Train data shape: {train_data.shape}, Test data shape: {test_data.shape}')

# Reshaping to 2D array (necessary for MinMaxScaler)
dataset_train = np.reshape(train_data, (-1, 1))
dataset_test = np.reshape(test_data, (-1, 1))

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale training data
scaled_train = scaler.fit_transform(dataset_train)
print(f'Scaled training data (first 5 rows):\n{scaled_train[:5]}')

# Scale testing data using the same scaler
scaled_test = scaler.transform(dataset_test)  # Use transform() for test data
print(f'Scaled testing data (first 5 rows):\n{scaled_test[:5]}')
