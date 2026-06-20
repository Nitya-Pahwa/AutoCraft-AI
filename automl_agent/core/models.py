import joblib
import numpy as np
from typing import Any, Dict

# sklearn imports 
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# optional boosting libraries
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# model registry 
# all valid architecture names the LLM can pick
CLASSIFICATION_MODELS = {
    "LogisticRegression",
    "LinearSVC",
    "DecisionTree",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "HistGradientBoosting",
    "KNN",
    "GaussianNB",
    *( {"XGBoost"}   if XGBOOST_AVAILABLE  else set() ),
    *( {"LightGBM"}  if LIGHTGBM_AVAILABLE else set() ),
}

REGRESSION_MODELS = {
    "Ridge",
    "Lasso",
    "ElasticNet",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "KNNRegressor",
    *( {"XGBoostRegressor"}  if XGBOOST_AVAILABLE  else set() ),
    *( {"LightGBMRegressor"} if LIGHTGBM_AVAILABLE else set() ),
}

ALL_MODELS = CLASSIFICATION_MODELS | REGRESSION_MODELS


def get_available_models(task_type: str) -> list[str]:
    """Return sorted list of available models for a given task."""
    if task_type == "classification":
        return sorted(CLASSIFICATION_MODELS)
    return sorted(REGRESSION_MODELS)


def build_model(architecture: str, hyperparams: Dict[str, Any], task_type: str):
    """
    Build and return an untrained sklearn-compatible model.
    Raises ValueError if architecture is unknown.
    """
    hp = hyperparams

    # classification
    if architecture == "LogisticRegression":
        return LogisticRegression(
            C=hp.get("C", 1.0),
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )

    if architecture == "LinearSVC":
        # LinearSVC doesn't support predict_proba natively —
        # wrap with CalibratedClassifierCV for probability estimates
        base = LinearSVC(
            C=hp.get("C", 1.0),
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
        return CalibratedClassifierCV(base, cv=3)

    if architecture == "DecisionTree":
        return DecisionTreeClassifier(
            max_depth=hp.get("max_depth", 6),
            min_samples_split=hp.get("min_samples_split", 5),
            class_weight="balanced",
            random_state=42,
        )

    if architecture == "RandomForest":
        return RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 200),
            max_depth=hp.get("max_depth", None),
            min_samples_split=hp.get("min_samples_split", 2),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    if architecture == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=hp.get("n_estimators", 200),
            max_depth=hp.get("max_depth", None),
            min_samples_split=hp.get("min_samples_split", 2),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    if architecture == "GradientBoosting":
        return GradientBoostingClassifier(
            n_estimators=hp.get("n_estimators", 100),
            learning_rate=hp.get("learning_rate", 0.1),
            max_depth=hp.get("max_depth", 3),
            subsample=hp.get("subsample", 0.8),
            random_state=42,
        )

    if architecture == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            max_iter=hp.get("max_iter", 100),
            learning_rate=hp.get("learning_rate", 0.1),
            max_depth=hp.get("max_depth", None),
            l2_regularization=hp.get("l2_regularization", 0.0),
            random_state=42,
        )

    if architecture == "KNN":
        return KNeighborsClassifier(
            n_neighbors=hp.get("n_neighbors", 5),
            weights=hp.get("weights", "uniform"),
            metric=hp.get("metric", "minkowski"),
            n_jobs=-1,
        )

    if architecture == "GaussianNB":
        return GaussianNB(
            var_smoothing=hp.get("var_smoothing", 1e-9),
        )

    if architecture == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost is not installed. Run: pip install xgboost")
        return XGBClassifier(
            n_estimators=hp.get("n_estimators", 200),
            learning_rate=hp.get("learning_rate", 0.1),
            max_depth=hp.get("max_depth", 4),
            subsample=hp.get("subsample", 0.8),
            colsample_bytree=hp.get("colsample_bytree", 0.8),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    if architecture == "LightGBM":
        if not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM is not installed. Run: pip install lightgbm")
        return LGBMClassifier(
            n_estimators=hp.get("n_estimators", 200),
            learning_rate=hp.get("learning_rate", 0.1),
            max_depth=hp.get("max_depth", -1),
            num_leaves=hp.get("num_leaves", 31),
            subsample=hp.get("subsample", 0.8),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    # regression
    if architecture == "Ridge":
        return Ridge(alpha=hp.get("alpha", 1.0))

    if architecture == "Lasso":
        return Lasso(alpha=hp.get("alpha", 0.01), max_iter=2000)

    if architecture == "ElasticNet":
        return ElasticNet(
            alpha=hp.get("alpha", 0.01),
            l1_ratio=hp.get("l1_ratio", 0.5),
            max_iter=2000,
        )

    if architecture == "DecisionTreeRegressor":
        return DecisionTreeRegressor(
            max_depth=hp.get("max_depth", 6),
            min_samples_split=hp.get("min_samples_split", 5),
            random_state=42,
        )

    if architecture == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=hp.get("n_estimators", 200),
            max_depth=hp.get("max_depth", None),
            random_state=42,
            n_jobs=-1,
        )

    if architecture == "ExtraTreesRegressor":
        return ExtraTreesRegressor(
            n_estimators=hp.get("n_estimators", 200),
            max_depth=hp.get("max_depth", None),
            random_state=42,
            n_jobs=-1,
        )

    if architecture == "GradientBoostingRegressor":
        return GradientBoostingRegressor(
            n_estimators=hp.get("n_estimators", 100),
            learning_rate=hp.get("learning_rate", 0.1),
            max_depth=hp.get("max_depth", 3),
            subsample=hp.get("subsample", 0.8),
            random_state=42,
        )

    if architecture == "HistGradientBoostingRegressor":
        return HistGradientBoostingRegressor(
            max_iter=hp.get("max_iter", 100),
            learning_rate=hp.get("learning_rate", 0.1),
            max_depth=hp.get("max_depth", None),
            l2_regularization=hp.get("l2_regularization", 0.0),
            random_state=42,
        )

    if architecture == "KNNRegressor":
        return KNeighborsRegressor(
            n_neighbors=hp.get("n_neighbors", 5),
            weights=hp.get("weights", "uniform"),
            n_jobs=-1,
        )

    if architecture == "XGBoostRegressor":
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost is not installed. Run: pip install xgboost")
        return XGBRegressor(
            n_estimators=hp.get("n_estimators", 200),
            learning_rate=hp.get("learning_rate", 0.1),
            max_depth=hp.get("max_depth", 4),
            subsample=hp.get("subsample", 0.8),
            random_state=42,
            n_jobs=-1,
        )

    if architecture == "LightGBMRegressor":
        if not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM is not installed. Run: pip install lightgbm")
        return LGBMRegressor(
            n_estimators=hp.get("n_estimators", 200),
            learning_rate=hp.get("learning_rate", 0.1),
            max_depth=hp.get("max_depth", -1),
            num_leaves=hp.get("num_leaves", 31),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    raise ValueError(
        f"Unknown architecture: '{architecture}'. "
        f"Valid options: {sorted(ALL_MODELS)}"
    )


def save_model(model, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)