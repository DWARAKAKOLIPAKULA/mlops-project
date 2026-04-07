from fastapi import FastAPI, Header, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
import logging
from src.predict import make_prediction
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

logging.basicConfig(level=logging.INFO)
# Initialize app
app = FastAPI()

def verify_api_key(x_api_key: str = Header(...)):
    api_key = os.getenv("API_KEY")

    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")

    if x_api_key != api_key:
        raise HTTPException(status_code=403, detail="Unauthorized")

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
def predict(data: StudentInput, x_api_key: str = Header(...)):
    
    verify_api_key(x_api_key)

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

print("CI/CD test")