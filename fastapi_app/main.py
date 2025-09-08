from fastapi import FastAPI

import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "regression.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/predict")
async def get_predict():
    return {"y_pred": 2}

@app.post("/predict")
async def predict(size: float, nb_rooms: int, garden: int):
    y_pred = model.predict([[size, nb_rooms, garden]])
    return {"y_pred": y_pred.tolist()}

