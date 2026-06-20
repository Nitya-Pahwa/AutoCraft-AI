import json

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from automl_agent.core.llm import get_llm
from automl_agent.core.state import (
    AgentState,
    EvaluationResult,
    EvaluatorInsight,
)
from automl_agent.utils.preprocessing import read_table


def _overfitting_threshold(architecture: str) -> float:
    """
    Each model family has a different normal train/val gap.
    Random forests always have high train scores by design —
    penalising them with a tight threshold causes false positives.
    """
    if architecture in (
        "RandomForest", "RandomForestRegressor",
        "ExtraTrees",   "ExtraTreesRegressor",
    ):
        return 0.35

    if architecture in (
        "GradientBoosting",         "GradientBoostingRegressor",
        "HistGradientBoosting",     "HistGradientBoostingRegressor",
        "XGBoost",                  "XGBoostRegressor",
        "LightGBM",                 "LightGBMRegressor",
    ):
        return 0.20

    if architecture in ("DecisionTree", "DecisionTreeRegressor"):
        return 0.25

    # linear models, KNN, NaiveBayes
    return 0.15


def _validation_data(state: AgentState):
    """Rebuild the same val split the trainer used."""
    info  = state.dataset_info
    saved = joblib.load(state.training_result.preprocessor_path)
    preprocessor  = saved["preprocessor"]
    label_encoder = saved["label_encoder"]

    df    = read_table(info.path).dropna(subset=[info.target_column])
    X     = df.drop(columns=[info.target_column])
    y_raw = df[info.target_column]

    if info.task_type == "classification":
        y = label_encoder.transform(y_raw).astype(int)
        min_class = int(np.bincount(y).min())
        stratify  = y if min_class >= 2 else None
    else:
        y        = y_raw.astype(np.float32).to_numpy()
        stratify = None

    _, X_val_raw, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    X_val = preprocessor.transform(X_val_raw).astype(np.float32)
    return X_val, y_val


def _get_predictions(state: AgentState):
    X_val, y_val = _validation_data(state)
    model        = joblib.load(state.training_result.model_path)
    preds        = model.predict(X_val)

    probs = None
    if state.dataset_info.task_type == "classification":
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)

    return y_val, preds, probs


def _llm_insight(
    state: AgentState,
    metrics: dict,
    is_overfitting: bool,
    passed: bool,
) -> EvaluatorInsight:
    llm = get_llm()
    if not llm:
        return _rule_based_insight(state, metrics, is_overfitting, passed)

    prompt = f"""
You are an ML evaluation expert interpreting model results.

Dataset context:
- Name: {state.dataset_info.name}
- Task: {state.dataset_info.task_type}
- Rows: {state.dataset_info.n_samples}
- Imbalanced: {state.dataset_info.is_imbalanced}
- Class balance: {state.dataset_info.class_balance}

Model: {state.ml_config.architecture if state.ml_config else "unknown"}

Metrics achieved:
{json.dumps(metrics, indent=2)}

Overfitting detected: {is_overfitting}
Overfitting gap (train_score - val_score): {state.training_result.overfit_gap if hasattr(state.training_result, 'overfit_gap') else 'N/A'}
Passed quality threshold: {passed}

Important context:
- For regression, RMSE must be interpreted relative to the target 
  variable range — do NOT call RMSE "high" without knowing the scale.
- For RandomForest/ExtraTrees, a train/val gap up to 0.35 is normal 
  and does NOT mean the model is broken.
- For medical classification datasets, recall on positive class 
  is more important than accuracy.

Analyst identified risks: {state.analyst_insight.risks if state.analyst_insight else []}

Respond ONLY in this exact JSON format, no markdown, no extra text:
{{
    "interpretation": "2-3 sentence interpretation referencing actual metric values in context",
    "strengths": [
        "specific strength with actual metric value",
        "specific strength 2"
    ],
    "critical_failures": [
        "specific failure with actual metric value, or empty list if none"
    ]
}}
"""
    try:
        response = llm.invoke(prompt).content.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        parsed = json.loads(response.strip())
        return EvaluatorInsight(metrics=metrics, **parsed)

    except Exception as exc:
        return _rule_based_insight(state, metrics, is_overfitting, passed)


def _rule_based_insight(
    state: AgentState,
    metrics: dict,
    is_overfitting: bool,
    passed: bool,
) -> EvaluatorInsight:
    strengths         = []
    critical_failures = []

    if state.dataset_info.task_type == "classification":
        acc = metrics.get("accuracy", 0)
        f1  = metrics.get("f1_weighted", 0)
        rec = metrics.get("recall_class_1", 0)
        auc = metrics.get("roc_auc", 0)

        if acc >= 0.75:
            strengths.append(f"Accuracy {acc:.1%} is above 75% baseline.")
        if f1 >= 0.70:
            strengths.append(f"Weighted F1 {f1:.2f} shows balanced class performance.")
        if auc >= 0.80:
            strengths.append(f"ROC-AUC {auc:.2f} shows strong discrimination ability.")
        if rec < 0.60:
            critical_failures.append(
                f"Recall on positive class is {rec:.1%} — "
                "many true positives are being missed."
            )
        if is_overfitting:
            critical_failures.append(
                "Train/val score gap exceeds threshold — "
                "model may not generalise well."
            )
        if not passed:
            critical_failures.append(
                "Model did not meet minimum quality threshold."
            )

        interpretation = (
            f"Model achieved {acc:.1%} accuracy and {f1:.2f} weighted F1."
            + (f" ROC-AUC={auc:.2f}." if auc else "")
            + (" Overfitting detected." if is_overfitting else "")
            + (" Passed threshold." if passed else " Did not pass threshold.")
        )

    else:
        r2   = metrics.get("r2",   0)
        rmse = metrics.get("rmse", 0)
        mae  = metrics.get("mae",  0)

        if r2 >= 0.75:
            strengths.append(f"R2={r2:.2f} indicates strong predictive power.")
        elif r2 >= 0.50:
            strengths.append(f"R2={r2:.2f} is acceptable with room to improve.")
        else:
            critical_failures.append(
                f"R2={r2:.2f} is below acceptable threshold of 0.50."
            )
        if is_overfitting:
            critical_failures.append(
                "Train/val score gap exceeds model-specific threshold."
            )

        interpretation = (
            f"Regression model: R2={r2:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}."
            + (" Overfitting detected." if is_overfitting else "")
            + (" Passed threshold." if passed else " Did not pass threshold.")
        )

    return EvaluatorInsight(
        metrics=metrics,
        interpretation=interpretation,
        strengths=strengths,
        critical_failures=critical_failures,
    )


def evaluator_agent(state: AgentState) -> AgentState:
    state.current_step = "evaluation"
    state.add_log("Evaluator: computing metrics.")

    try:
        # Step 1: get predictions 
        y_val, preds, probs = _get_predictions(state)

        # Step 2: compute metrics 
        if state.dataset_info.task_type == "classification":
            is_binary = state.dataset_info.n_classes == 2

            metrics = {
                "accuracy":           round(float(accuracy_score(y_val, preds)), 4),
                "precision_macro":    round(float(precision_score(y_val, preds, average="macro",    zero_division=0)), 4),
                "precision_weighted": round(float(precision_score(y_val, preds, average="weighted", zero_division=0)), 4),
                "recall_macro":       round(float(recall_score(y_val, preds,    average="macro",    zero_division=0)), 4),
                "recall_weighted":    round(float(recall_score(y_val, preds,    average="weighted", zero_division=0)), 4),
                "f1_macro":           round(float(f1_score(y_val, preds,        average="macro",    zero_division=0)), 4),
                "f1_weighted":        round(float(f1_score(y_val, preds,        average="weighted", zero_division=0)), 4),
            }

            # per-class recall on positive class
            report_dict = classification_report(
                y_val, preds, output_dict=True, zero_division=0
            )
            metrics["recall_class_1"] = round(
                float(
                    report_dict.get("1", report_dict.get(1, {})).get("recall", 0.0)
                ),
                4,
            )

            if probs is not None:
                try:
                    if is_binary:
                        metrics["roc_auc"] = round(
                            float(roc_auc_score(y_val, probs[:, 1])), 4
                        )
                    else:
                        metrics["roc_auc_ovr"] = round(
                            float(
                                roc_auc_score(
                                    y_val, probs,
                                    multi_class="ovr",
                                    average="macro",
                                )
                            ),
                            4,
                        )
                except Exception:
                    pass

            cm     = confusion_matrix(y_val, preds).tolist()
            report = classification_report(y_val, preds, zero_division=0)
            passed = (
                metrics["accuracy"] >= 0.70
                or metrics["f1_weighted"] >= 0.70
                or metrics.get("roc_auc", 0) >= 0.75
            )

        else:
            preds  = preds.squeeze() if hasattr(preds, "squeeze") else preds
            mse    = mean_squared_error(y_val, preds)
            r2     = r2_score(y_val, preds)
            metrics = {
                "rmse": round(float(np.sqrt(mse)), 4),
                "mae":  round(float(mean_absolute_error(y_val, preds)), 4),
                "r2":   round(float(r2), 4),
            }
            cm     = None
            report = ""
            passed = metrics["r2"] >= 0.50

        # Step 3: overfitting check 
        train_scores = state.training_result.train_scores
        val_scores   = state.training_result.val_scores
        overfit_gap  = float(train_scores[-1] - val_scores[-1]) if train_scores else 0.0

        threshold      = _overfitting_threshold(
            state.ml_config.architecture if state.ml_config else ""
        )
        is_overfitting = overfit_gap > threshold

        # Step 4: regression approval  don't penalise good R2 
        if state.dataset_info.task_type == "regression":
            r2_val = metrics.get("r2", 0)
            if r2_val >= 0.75:
                # strong R2: approve even with mild overfitting
                passed = True
            elif r2_val >= 0.60:
                # acceptable R2: approve only if overfitting is within threshold
                passed = not is_overfitting
            else:
                passed = False

        else:
            passed = bool(passed and not is_overfitting)

        state.evaluation_result = EvaluationResult(
            metrics=metrics,
            confusion_matrix=cm,
            classification_report=report,
            is_overfitting=is_overfitting,
            overfit_gap=round(overfit_gap, 4),
            passed=passed,
        )
        state.add_log(
            f"Evaluator: metrics={metrics}, "
            f"overfit_gap={overfit_gap:.3f} (threshold={threshold}), "
            f"overfitting={is_overfitting}, passed={passed}."
        )

        # Step 5: LLM interprets the metrics
        state.evaluator_insight = _llm_insight(
            state, metrics, is_overfitting, passed
        )
        state.add_log(
            f"Evaluator insight: {state.evaluator_insight.interpretation}"
        )

    except Exception as exc:
        state.error = f"Evaluator failed: {exc}"
        state.add_log(state.error)

    return state