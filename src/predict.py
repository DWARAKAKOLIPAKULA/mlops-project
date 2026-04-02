import joblib

model = joblib.load("models/model.pkl")

def make_prediction(input_data):
    return model.predict([input_data])