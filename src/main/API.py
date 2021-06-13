from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from predict import run_prediction


app = FastAPI()
templates = Jinja2Templates(directory="templates/")


@app.get('/')
def read_root():
    return 'Spam detector'

@app.get('/predict')
def form_post(request: Request):
    result = "Type a phrase"
    return templates.TemplateResponse('form.html', context={'request': request, 'result':result})


@app.post("/predict")
def form_post(request: Request, word: str = Form(...)):
    result = run_prediction(word)
    return templates.TemplateResponse('form.html', context={'request': request, 'result': result})
    