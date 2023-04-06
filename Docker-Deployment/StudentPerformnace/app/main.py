import pickle
import numpy as np
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel, conlist

app = FastAPI(title = "Predict Student Performance")

class Student(BaseModel):
    batches: List[conlist(item_type=float, min_items=40, max_items=40)]

@app.on_event("startup")
def load_clf():
    with open("/app/lgbm_student.pkl", "rb") as file:
        global clf 
        clf = pickle.load(file)

@app.get("/")
def home():
    return "Welcome to Server predicting performance of student!"

@app.post("/predict")
def predict(student: Student):
    np_batches = np.array(student.batches)
    pred = clf.predict(np_batches).tolist()
    return {"Prediction": pred} 
