from pipeline import pipe
from preprocess.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split
from config import config


def run_training(opts) -> None:
    """Train the model"""
    
    # Load the data
    data = load_dataset(file_name=opts.source)
    
    X_train, X_test, y_train, y_test = train_test_split(X=data[config.FEATURES],
                                                        y=data[config.TARGET],
                                                        test_size=config.TEST_SIZE,
                                                        random_state=config.RANDOM_STATE)
    
    # Fit the pipeline
    pipe.fit(X_train, y_train)
    
    # Save the pipeline
    save_pipeline(path=opts.path, pipeline_to_persist=pipe)


if __name__ == "__main__":
    pass
    