from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import tensorflow as tf
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI()

model_path = "lstm_inventory_model.h5"
scaler_path = "scaler.npy"
data_path = "processed_inventory.csv"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file '{scaler_path}' not found!")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file '{data_path}' not found!")

model = tf.keras.models.load_model(model_path)
scaler = np.load(scaler_path, allow_pickle=True).item()

df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])

class PredictionRequest(BaseModel):
    store: int
    item: int

@app.get("/")
def home():
    return {"message": "Inventory Prediction API is running!"}

@app.post("/predict")
def predict_sales(data: PredictionRequest):
    store, item = data.store, data.item

    filtered = df[(df['store'] == store) & (df['item'] == item)]

    if len(filtered) < 10:
        raise HTTPException(status_code=400, detail="Not enough historical data for prediction.")


    sales_data = filtered['sales'].values[-10:].reshape(1, 10, 1)

   
    prediction = model.predict(sales_data)

    predicted_sales = scaler.inverse_transform([[prediction[0][0]]])[0][0]

    return {
        "store": store,
        "item": item,
        "predicted_sales": float(predicted_sales)  # Ensure JSON serialization
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
