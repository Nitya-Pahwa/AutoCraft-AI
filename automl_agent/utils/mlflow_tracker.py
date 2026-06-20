import mlflow


def get_runs(experiment_name: str = "agentic_automl"):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    runs = mlflow.search_runs([experiment.experiment_id])
    return runs

