import typing as t

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from config import config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    df = pd.read_csv(Path(f"{config.DATASET_DIR}/{file_name}"))
    df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1) # TODO Add unnamed to config
    df = df.rename(columns=config.FEATURES_TO_RENAME) # TODO Add FEATURES_TO_RENAME
    # df.columns = ['labels', 'data']
    df['b_labels'] = df['labels'].map(config.LABEL_MAPPING) # TODO Add LABEL_MAPPING
    # df['b_labels'] = df['labels'].map({'ham':0, 'spam': 1})
    return df

def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    save_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl" # TODO Import _version, add PIPELINE_SAVE_FILE
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    
    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)

def load_pipeline(*, file_name: str) -> Pipeline:
    file_path = config.TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model

def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
