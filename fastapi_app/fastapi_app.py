from fastapi import FastAPI

import joblib

import os
filename = '../regression.joblib'
if os.path.exists(filename):
    model = joblib.load(filename)
else:
    raise FileNotFoundError(f"Model file not found: {filename}")

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

