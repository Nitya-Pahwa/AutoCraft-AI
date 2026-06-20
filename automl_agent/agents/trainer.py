import os
import time

import joblib
import mlflow
import numpy as np
import optuna

from automl_agent.core.models import (
    ALL_MODELS,
    build_model,
    save_model,
)
from automl_agent.core.state import AgentState, TrainingResult
from automl_agent.utils.preprocessing import build_splits, save_preprocessor

optuna.logging.set_verbosity(optuna.logging.WARNING)

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def _optuna_space(trial, architecture: str) -> dict:
    """Define hyperparameter search space per architecture."""

    if architecture == "LogisticRegression":
        return {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}

    if architecture == "LinearSVC":
        return {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}

    if architecture == "DecisionTree":
        return {
            "max_depth":         trial.suggest_int("max_depth", 2, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }

    if architecture in ("RandomForest", "ExtraTrees",
                         "RandomForestRegressor", "ExtraTreesRegressor"):
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
            "max_depth":         trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }

    if architecture in ("GradientBoosting", "GradientBoostingRegressor"):
        return {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":     trial.suggest_int("max_depth", 2, 8),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
        }

    if architecture in ("HistGradientBoosting", "HistGradientBoostingRegressor"):
        return {
            "max_iter":          trial.suggest_int("max_iter", 50, 300),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", 2, 10),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
        }

    if architecture in ("XGBoost", "XGBoostRegressor"):
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 2, 8),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    if architecture in ("LightGBM", "LightGBMRegressor"):
        return {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":     trial.suggest_int("max_depth", -1, 12),
            "num_leaves":    trial.suggest_int("num_leaves", 15, 127),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
        }

    if architecture in ("KNN", "KNNRegressor"):
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 2, 20),
            "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),
        }

    if architecture == "GaussianNB":
        return {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)
        }

    if architecture == "Ridge":
        return {"alpha": trial.suggest_float("alpha", 0.001, 100.0, log=True)}

    if architecture in ("Lasso", "ElasticNet"):
        hp = {"alpha": trial.suggest_float("alpha", 0.0001, 1.0, log=True)}
        if architecture == "ElasticNet":
            hp["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 0.9)
        return hp

    if architecture in ("DecisionTreeRegressor",):
        return {
            "max_depth":         trial.suggest_int("max_depth", 2, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }

    return {}


def _score(model, X_val, y_val, task_type: str) -> float:
    """Higher is always better — returns a maximization score."""
    preds = model.predict(X_val)
    if task_type == "classification":
        from sklearn.metrics import f1_score
        return float(f1_score(y_val, preds, average="weighted", zero_division=0))
    else:
        from sklearn.metrics import r2_score
        score = float(r2_score(y_val, preds))
        return max(score, -1.0)   # floor at -1 so study doesn't explode


def trainer_agent(state: AgentState) -> AgentState:
    state.current_step = "training"
    state.add_log(f"Trainer: starting. architecture={state.ml_config.architecture}.")

    try:
        arch = state.ml_config.architecture
        task = state.dataset_info.task_type

        if arch not in ALL_MODELS:
            raise ValueError(
                f"Unknown architecture '{arch}'. "
                f"Valid: {sorted(ALL_MODELS)}"
            )

        # preprocessing 
        X_train, X_val, y_train, y_val, preprocessor, label_encoder = build_splits(
            state.dataset_info
        )
        state.ml_config.input_dim = int(X_train.shape[1])
        prep_path = os.path.join(
            ARTIFACT_DIR, f"{state.dataset_info.name}_preprocessor.joblib"
        )
        state.add_log(
            f"Trainer: train={X_train.shape}, val={X_val.shape}."
        )

        # Optuna hyperparameter search 
        n_trials = (
            state.planner_decision.optuna_trials
            if state.planner_decision else 10
        )
        # use fewer trials on retry good params already found
        if state.retry_count > 0:
            n_trials = max(5, n_trials // 2)

        def objective(trial):
            hp     = _optuna_space(trial, arch)
            model  = build_model(arch, hp, task)
            model.fit(X_train, y_train)
            return _score(model, X_val, y_val, task)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        trial_scores = [
            float(trial.value)
            for trial in study.trials
            if trial.value is not None
        ]
        state.ml_config.hyperparams.update(best_params)
        state.add_log(f"Trainer: Optuna best_params={best_params}.")

        # final training with best params
        start = time.time()
        final_model = build_model(arch, state.ml_config.hyperparams, task)
        final_model.fit(X_train, y_train)
        training_time = time.time() - start

        val_score    = _score(final_model, X_val, y_val, task)
        train_score  = _score(final_model, X_train, y_train, task)
        overfit_gap  = float(train_score - val_score)

        # save model and preprocessor 
        model_path = os.path.join(
            ARTIFACT_DIR, f"{state.dataset_info.name}_best.joblib"
        )
        save_model(final_model, model_path)
        save_preprocessor(prep_path, preprocessor, label_encoder)

        # MLflow logging 
        mlflow.set_experiment(state.experiment_name)
        with mlflow.start_run(
            run_name=f"{state.dataset_info.name}_retry_{state.retry_count}"
        ) as run:
            state.mlflow_run_id = run.info.run_id
            mlflow.log_params({"architecture": arch, **state.ml_config.hyperparams})
            mlflow.log_metric("val_score",      val_score)
            mlflow.log_metric("train_score",    train_score)
            mlflow.log_metric("overfit_gap",    overfit_gap)
            mlflow.log_metric("training_time_s", training_time)

        # sklearn has no epoch losses store single-point curves
        # so evaluator and visualizer don't crash
        state.training_result = TrainingResult(
            train_losses=[float(1 - train_score)],
            val_losses=[float(1 - val_score)],
            train_scores=[train_score],
            val_scores=[val_score],
            trial_scores=trial_scores,
            best_epoch=0,
            best_val_loss=float(1 - val_score),
            best_params=best_params,
            model_path=model_path,
            preprocessor_path=prep_path,
            training_time_s=training_time,
        )

        state.add_log(
            f"Trainer: done. val_score={val_score:.4f}, "
            f"train_score={train_score:.4f}, "
            f"overfit_gap={overfit_gap:.4f}, "
            f"time={training_time:.1f}s."
        )

    except Exception as exc:
        state.error = f"Trainer failed: {exc}"
        state.add_log(state.error)

    return state
