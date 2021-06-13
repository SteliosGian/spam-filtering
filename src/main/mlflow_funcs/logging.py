import mlflow


def log_metrics(metrics: dict, 
                params: dict, 
                tracking_uri: str, 
                experiment_name: str) -> None:
    """
    Logs model metrics and hyperparameters.
    :param metrics: Metrics dictionary
    :param params: Parameter dictionary to use
    :param tracking_uri: The tracking uri. If local, it is the tracking folder to use
    :return: None
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
