# Inventory-Prediction-API
This repository contains an LSTM-based inventory sales prediction system. The project processes historical sales data and predicts future sales for a given store and item. It includes a FastAPI server to expose an HTTP endpoint for querying predictions

## 📌 Features
✅ Train an **LSTM model** on historical sales data.  
✅ Provide sales predictions via an HTTP API (`/predict`).  
✅ Support for multiple stores and items.  
✅ API documentation available via **Swagger UI** (`/docs`).  
✅ Postman collection provided for easy testing.


# Installation & Setup

### **01. Clone the Repository**
```sh
git clone https://github.com/WasanaKarunasena/Inventory-Prediction-API.git
cd inventory-prediction

### **02. Install Dependencies**

Make sure you have Python 3.8+ installed, then run:
pip install -r requirements.txt

03. Data Preprocessing

Run the preprocessing script to prepare the data for training:
python preprocess_data.py
This will generate the files,

processed_inventory.csv (cleaned dataset)
scaler.npy (saved data scaler for predictions)

04. Train the LSTM Model

Train the model using:
python train_model.py
This will create,

lstm_inventory_model.h5 (trained model)

05. Run the API
Start the FastAPI server using:
python app.py
You should see output like.

INFO:     Uvicorn running on http://127.0.0.1:8000

🌍 API Endpoints
POST	-  /predict	Predict sales for a given store & item
GET -	/docs	Open API documentation (Swagger UI)

Example Request,
🔹 POST /predict
URL: http://127.0.0.1:8000/predict

Body (JSON):

json
Copy
Edit
{
  "store": 1,
  "item": 100
}
Response (JSON):

json
Copy
Edit
{
  "store": 1,
  "item": 100,
  "predicted_sales": 235.67
}

06. Testing with Postman

Open Postman.
Click Import → Select postman_collection.json (attached).
Use the POST /predict request to test predictions.
Click Send to get predictions.
