import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from predict import run_prediction


app = FastAPI()
templates = Jinja2Templates(directory="templates/")


@app.get('/')
def read_root():
    return 'This is a Spam Detector'

@app.get('/predict')
def form_post(request: Request):
    result = "Type a word"
    return templates.TemplateResponse('form.html', context={'request': request, 'result':result})


@app.post("/predict")
def form_post(request: Request, word: str = Form(...)):
    result = run_prediction(word)
    return templates.TemplateResponse('form.html', context={'request': request, 'result': result})
    