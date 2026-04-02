from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import logging
from src.predict import make_prediction

logging.basicConfig(level=logging.INFO)
# Initialize app
app = FastAPI()

# Load model
model = joblib.load("models/model.pkl")

class StudentInput(BaseModel):
    sem_present_count: int
    sem_absent_count: int
    sem_eval_lec_test_1_mark: int
    sem_eval_lab_test_1_mark: int
    semester_evaluation_mid_mark: int
    sem_eval_lec_test_2_mark: int
    sem_eval_lab_test_2_mark: int
    semester_evaluation_pre_gtu_mark: int
    semester_evaluation_internal_mark: int

@app.get("/")
def home():
    return {"message": "ML Model API is running 🚀"}

@app.post("/predict")
def predict(data: StudentInput):

    logging.info(f"Received input: {data}")
    logging.info(F"Prediction: {prediction}")
    
    input_data = [
        data.sem_present_count,
        data.sem_absent_count,
        data.sem_eval_lec_test_1_mark,
        data.sem_eval_lab_test_1_mark,
        data.semester_evaluation_mid_mark,
        data.sem_eval_lec_test_2_mark,
        data.sem_eval_lab_test_2_mark,
        data.semester_evaluation_pre_gtu_mark,
        data.semester_evaluation_internal_mark
    ]

    prediction = make_prediction(input_data)

    return {"prediction": prediction.tolist()}