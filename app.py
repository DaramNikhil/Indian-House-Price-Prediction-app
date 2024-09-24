from fastapi import FastAPI
import joblib
import pandas as pd
from house_app import Item
import uvicorn


app = FastAPI()

# Load the model and scaler
my_model = joblib.load("model/model_and_scaler.pkl")
scaler = my_model["scaler"]
model = my_model["model"]


@app.get("/")
def home_page():
    return {"message": "Welcome to the House Price Prediction website."}


@app.post("/predict")
def predict_price(house: Item):
    try:
        new_data = pd.DataFrame([house.dict()])
        scaled_data = scaler.transform(new_data)
        prediction = model.predict(scaled_data)
        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        return {"error": str(e)}


    