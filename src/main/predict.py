import argparse
from preprocess.data_manager import load_dataset, load_pipeline
from config import config
from tensorflow.keras.models import load_model


def run_prediction(opts):
    data = load_dataset(path=opts.source)
    
    pipe = load_pipeline(path=opts.pipe_path, file_name='pipe.pkl')
    
    model = load_model(opts.model_path)
    
    proc_data = pipe.transform(data[config.FEATURES])
    
    preds = model.predict(proc_data)
    return preds



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction phase", prog="predict", usage="%(prog) [options]")
    parser.add_argument("--source", required=True, type=str, help="Source path of the prediction dataset")
    parser.add_argument("--pipe_path", required=True, type=str, help="Path to load the pipeline")
    parser.add_argument("--model_path", required=True, type=str, help="Path to load the trained model")
    args = parser.parse_args()
    run_prediction(args)
