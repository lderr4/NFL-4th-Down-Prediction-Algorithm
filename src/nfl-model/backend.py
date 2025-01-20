from robo_coach import robo_coach
from preprocessing import get_dataset, prep_for_classifier, years
from fastapi import FastAPI
from pydantic import BaseModel
from pandas import DataFrame
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


app = FastAPI()
rc = robo_coach()
rc.load()


_, test = get_dataset(None)
print(test.columns)


class InputData(BaseModel):
    play: dict  

@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

@app.post("/get_classifier_predict_proba")
def get_classifier_predict_proba(input_data: InputData):
    
    df = DataFrame([input_data.dict()["play"]])
    probas, classes = rc.get_classifier_predict_proba(df)
    probas = {c: pred for c, pred in zip(classes, probas[0])}
    
    wpa_preds = {}
    for play_type, proba in probas.items():
        if proba > 0:

            wpa_pred = rc.get_wpa_pred(df, play_type)
            
            wpa_preds[play_type] = float(wpa_pred[0])

    response = {"predict_probabilities": probas, "wpa_predictions": wpa_preds}
    return {"response": response}

@app.get("/random_test_row")
def get_random_test_row():
    return test.sample(1).to_dict(orient="records")

     



