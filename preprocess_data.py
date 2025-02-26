import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


file_path = "train.csv" 
if not os.path.exists(file_path):
   raise FileNotFoundError("Dataset not found.")

df = pd.read_csv(file_path)

print("Dataset Columns:", df.columns)

required_columns = ['date', 'store', 'item', 'sales']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing columns in dataset: {missing_columns}")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['store', 'item', 'date'])

# Normalize sales data
scaler = MinMaxScaler(feature_range=(0, 1))
df['sales'] = scaler.fit_transform(df[['sales']])

# Save processed data and scaler
df.to_csv('processed_inventory.csv', index=False)
np.save('scaler.npy', scaler)
print("Preprocessing complete. Data saved.")
