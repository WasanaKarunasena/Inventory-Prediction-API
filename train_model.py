import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

# Load preprocessed data
df = pd.read_csv('processed_inventory.csv')
df['date'] = pd.to_datetime(df['date'])

grouped = df.groupby(['store', 'item'])


sequence_length = 10
X, y = [], []

for _, group in grouped:
    sales_data = group['sales'].values
    for i in range(len(sales_data) - sequence_length):
        X.append(sales_data[i:i + sequence_length])
        y.append(sales_data[i + sequence_length])

X, y = np.array(X), np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))


model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, batch_size=16, epochs=20)

model.save("lstm_inventory_model.h5")
print("LSTM Model trained and saved.")
