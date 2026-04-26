from fastapi import FastAPI
from pydantic import BaseModel, TypeAdapter
import joblib
import json
import pandas as pd

app = FastAPI(title="Titanic Survival Predictor API")

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# Loading the model and json file
filename = "knn_model.pkl"
model = joblib.load(filename)
with open('columns.json', 'r') as file:
    columns = json.load(file)

print(columns)

# Define input schema - expected by API
class Passenger(BaseModel):
    Pclass: int     # 1, 2, 3
    Sex: int        # 0 = female, 1 = male (encoded)
    Age: int
    SibSp: int
    Parch: int
    Embarked: int   # 0 = C, 1 = Q, 2 = S (encoded)

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "ok", "model": "KNN Titanic Classifier"}

# Prediction endpoint
@app.post("/predict")
def predict(passenger: Passenger):
    # Convert input to dataframe in correct column order
    input_df = pd.DataFrame([passenger.dict()])[columns]

    # Making predictions
    prediction = model.predict(input_df)[0]
    # Provides probability of predictions
    probability = model.predict_proba(input_df)[0]

    # print(input_df)
    return {
        "survived": bool(prediction),
        "survival_probability": round(float(probability[1]), 3),
        "death_probaility": round(float(probability[0]), 3)
    }