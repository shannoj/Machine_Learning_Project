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
from data_plot import plot
from get_data import df

# Plot the 'Open' price data (you can add more columns here if needed)
plot(df, columns=['Open'])

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

