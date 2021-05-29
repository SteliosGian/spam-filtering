import argparse
from pipeline import pipe
from preprocess.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split
from config import config
from models.lstm_model import LSTMModel


def run_training(opts) -> None:
    """Train the model"""
    
    # Load the data
    data = load_dataset(path=opts.source)
    X_train, X_test, y_train, y_test = train_test_split(data[config.FEATURES],
                                                        data[config.TARGET],
                                                        test_size=config.TEST_SIZE,
                                                        random_state=config.RANDOM_STATE)

    # Fit and transform the pipeline
    pipe.fit(X_train)
    proc_data_x = pipe.transform(X_train)
    
    # Initialize and fit the model
    model = LSTMModel()
    model.fit(proc_data_x, y_train)
    
    # Save the trained model
    model.save(opts.model_path)
    
    # Save the pipeline
    save_pipeline(path=opts.pipe_path, pipeline_to_persist=pipe)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training phase", prog="train", usage="%(prog) [options]")
    parser.add_argument("--source", required=True, type=str, help="Source path of the training dataset")
    parser.add_argument("--pipe_path", required=True, type=str, help="Path to save the pipeline")
    parser.add_argument("--model_path", required=True, type=str, help="Path to save the trained model")
    args = parser.parse_args()
    run_training(args)
