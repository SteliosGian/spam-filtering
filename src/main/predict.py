import pandas as pd
import numpy as np



def run_prediction(word, model, pipe):
    """Generate predictions"""
    data = pd.Series(word)

    # Transform the data
    proc_data = pipe.transform(data)

    # Generate predictions
    preds = model.predict(proc_data)
    preds_binary = np.where(preds > 0.5, 'Spam', 'Not spam')

    return preds_binary[0][0]
