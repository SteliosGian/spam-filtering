from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from predict import run_prediction
from preprocess.data_manager import load_pipeline
from tensorflow.keras.models import load_model


app = FastAPI()
templates = Jinja2Templates(directory="templates/")

# Load model and pipeline
pipe = load_pipeline(path='trained_pipe/', file_name='pipe.pkl')
model = load_model('trained_models/')

@app.get('/')
def read_root():
    return 'Spam detector'

@app.get('/predict')
def form_post(request: Request):
    result = "Type a phrase"
    return templates.TemplateResponse('form.html', context={'request': request, 'result':result})


@app.post("/predict")
def form_post(request: Request, word: str = Form(...)):
    result = run_prediction(word, model, pipe)
    return templates.TemplateResponse('form.html', context={'request': request, 'result': result})
    