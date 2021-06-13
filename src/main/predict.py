import pandas as pd
import numpy as np
from preprocess.data_manager import load_pipeline
from tensorflow.keras.models import load_model



def run_prediction(word):
    """Generate predictions"""
    data = pd.Series(word)
    
    # Load pipeline and model
    pipe = load_pipeline(path='trained_pipe/', file_name='pipe.pkl')
    model = load_model('trained_models/')

    # Transform the data
    proc_data = pipe.transform(data)

    # Generate predictions
    preds = model.predict(proc_data)
    preds_binary = np.where(preds > 0.5, 'Spam', 'Not spam')

    return preds_binary[0][0]
