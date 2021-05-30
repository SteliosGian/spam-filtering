import argparse
import pandas as pd
import numpy as np
from preprocess.data_manager import load_dataset, load_pipeline
from config import config
from tensorflow.keras.models import load_model




# def run_prediction(opts):
#     """Generate predictions"""
    
#     # Load the dataset, pipeline, and model
#     data = load_dataset(path=opts.source)
#     pipe = load_pipeline(path=opts.pipe_path, file_name='pipe.pkl')
#     model = load_model(opts.model_path)
 
#     # Transform the data
#     proc_data = pipe.transform(data)
  
#     # Generate predictions
#     preds = model.predict(proc_data)
#     preds_binary = np.where(preds > 0.5, 'Spam', 'Not spam')
#     return preds_binary
    

def run_prediction(word):
    """Generate predictions"""
    data = pd.Series(word)
    # Load the dataset, pipeline, and model
    # data = load_dataset(path='data/spam.csv')
    pipe = load_pipeline(path='trained_pipe/', file_name='pipe.pkl')
    model = load_model('trained_models/')

    # Transform the data
    proc_data = pipe.transform(data)

    # Generate predictions
    preds = model.predict(proc_data)
    preds_binary = np.where(preds > 0.5, 'Spam', 'Not spam')

    print(preds_binary)
    return preds_binary[0][0]



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Prediction phase", prog="predict", usage="%(prog) [options]")
#     parser.add_argument("--source", required=True, type=str, help="Source path of the prediction dataset")
#     parser.add_argument("--pipe_path", required=True, type=str, help="Path to load the pipeline")
#     parser.add_argument("--model_path", required=True, type=str, help="Path to load the trained model")
#     args = parser.parse_args()
#     run_prediction(args)
