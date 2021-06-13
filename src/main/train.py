import argparse
import numpy as np
from pipeline import pipe
from preprocess.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split
from config import config
from models.lstm_model import LSTMModel
from mlflow_funcs.logging import log_metrics


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
    proc_data_test = pipe.transform(X_test)

    # Initialize and fit the model
    model = LSTMModel(lstm_units=opts.lstm_units,
                      embedding_dim=opts.embedding_dim,
                      input_dim=opts.input_dim,
                      optimizer=opts.optimizer)
    model.fit(proc_data_x, y_train, epochs=opts.epochs, validation_data=(proc_data_test, y_test))
    
    params = model.get_numeric_params
    avg_metrics = {}
    for key, value in model.model.history.history.items():
        avg_metrics[key] = np.mean(value)
    
    log_metrics(metrics=avg_metrics, params=params, tracking_uri=config.TRACKING_URI, experiment_name='test_experiment')
    
    # Save the trained model
    model.save(opts.model_path)
    
    # Save the pipeline
    save_pipeline(path=opts.pipe_path, pipeline_to_persist=pipe)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training phase", prog="train", usage="%(prog) [options]")
    parser.add_argument("--source", required=True, type=str, help="Source path of the training dataset")
    parser.add_argument("--pipe_path", required=True, type=str, help="Path to save the pipeline")
    parser.add_argument("--model_path", required=True, type=str, help="Path to save the trained model")
    parser.add_argument("--lstm_units", required=False, default=15, type=int, help="Number of units in a layer")
    parser.add_argument("--embedding_dim", required=False, default=20, type=int, help="Embedding dimension")
    parser.add_argument("--input_dim", required=False, default=7295, type=int, help="Input dimension")
    parser.add_argument("--optimizer", required=False, default='adam', type=str, help="Model optimizer")
    parser.add_argument("--epochs", required=False, default=2, type=int, help="Number of epochs")
    args = parser.parse_args()
    run_training(args)
