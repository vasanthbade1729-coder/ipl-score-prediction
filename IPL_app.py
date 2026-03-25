from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("ipl_model.pkl")

teams = [
    'Chennai Super Kings',
    'Delhi Daredevils',
    'Kings XI Punjab',
    'Kolkata Knight Riders',
    'Mumbai Indians',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

# ─── Define Input Schema (THIS FIXES YOUR PROBLEM) ───
class MatchInput(BaseModel):
    batting_team: str
    bowling_team: str
    runs: int
    wickets: int
    overs: float
    runs_last_5: int
    wickets_last_5: int

# The frontend is mounted at the root later

# Prediction API
@app.post("/predict")
def predict(data: MatchInput):

    prediction_array = []

    # Batting team encoding
    for team in teams:
        prediction_array.append(1 if data.batting_team == team else 0)

    # Bowling team encoding
    for team in teams:
        prediction_array.append(1 if data.bowling_team == team else 0)

    # Add numeric features
    prediction_array.extend([
        data.runs,
        data.wickets,
        data.overs,
        data.runs_last_5,
        data.wickets_last_5
    ])

    prediction_array = np.array([prediction_array])

    prediction = model.predict(prediction_array)[0]

    return {
        "predicted_score": int(round(prediction))
    }

# Mount the static frontend application at the root
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")