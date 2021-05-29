import joblib
import typing as t
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from config import config


def load_dataset(*, path: str) -> pd.DataFrame:
    df = pd.read_csv(Path(f"{path}"))
    df = df.drop(config.FEATURES_TO_DROP, axis=1)
    df = df.rename(mapper=config.FEATURES_TO_RENAME, axis=1)
    df['b_labels'] = df['labels'].map(config.LABEL_MAPPING)
    return df


def save_pipeline(*, path: str, pipeline_to_persist: Pipeline) -> None:
    save_file_name = f"{config.PIPELINE_SAVE_FILE}.pkl" # TODO Import _version, add PIPELINE_SAVE_FILE
    save_path = f"{path}/{save_file_name}"
    
    remove_old_pipelines(path=path, files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, path: str, file_name: str) -> Pipeline:
    file_path = f"{path}/{file_name}"
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, path: str, files_to_keep: t.List[str]) -> None:
    do_not_delete = files_to_keep + ["__init__.py"]
    path = Path(path)
    for model_file in path.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
