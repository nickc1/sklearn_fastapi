from fastapi import FastAPI
from joblib import load
from pydantic import  BaseModel


# define model for post request. Not needed if just implementing get
class ModelParams(BaseModel):
    param1: float
    param2: float


app = FastAPI()

clf = load('/model/model_1.joblib')

def get_prediction(param1, param2):
    
    x = [[param1, param2]]

    y = clf.predict(x)[0]  # just get single value
    prob = clf.predict_proba(x)[0].tolist()  # send to list for return

    return {'prediction': int(y), 'probability': prob}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict/{param1}/{param2}")
def predict(param1: float, param2: float):

    pred = get_prediction(param1, param2)

    return pred


@app.post("/predict-post/")
def post_predict(params: ModelParams):

    # param_dict = params.dict()
    # print(param_dict)
    pred = get_prediction(params.param1, params.param2)

    return pred

