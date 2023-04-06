import pickle
import numpy as np
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel, conlist

app = FastAPI(title="Predicting Response of Customer")

class Customer(BaseModel):
    batches: List[conlist(item_type=float, max_items=5, min_items=5)]

@app.on_event("startup")
def load_clf():
    with open("app/mkt.pkl", "rb") as file:
        global clf 
        clf = pickle.load(file)

@app.get("/")
def home():
    return "Congratulations! Your API to pull batches request was succesfully!"

@app.post("/predict")
def predict(customer: Customer):
    batches = customer.batches
    np_batches = np.array(batches)
    pred = clf.predict(np_batches).tolist()
    return {"Prediction": pred}    