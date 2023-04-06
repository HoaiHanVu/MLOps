import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from typing import List

app = FastAPI(title="Precicting Traffic of Rented Bike")

class Bike(BaseModel):
    batches: List[conlist(item_type=float, min_items=24, max_items=24)]

@app.on_event("startup")
def load_reg():
    with open("app/bike-traffic.pkl", "rb") as file:
        global reg 
        reg = pickle.load(file)

@app.get("/")
def home():
    return "Welcome to Predicting Bike Traffic Rented"

@app.post("/predict")
def predict(bike: Bike):
    batches = bike.batches
    np_batches = np.array(batches)
    pred = np.round(reg.predict(np_batches), 3).tolist()
    return {'Prediction': pred}