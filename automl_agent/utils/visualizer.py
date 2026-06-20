import plotly.graph_objects as go
import pandas as pd
import numpy as np


def confusion_matrix_figure(cm):
    arr = np.array(cm)
    fig = go.Figure(data=go.Heatmap(z=arr, text=arr, texttemplate="%{text}", colorscale="Blues"))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    return fig

def feature_importance_figure(model, feature_names: list):
    """
    Works for RandomForest, ExtraTrees, GradientBoosting,
    XGBoost, LightGBM — any model with feature_importances_.
    Returns None if model doesn't support it.
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        return None

    # sort by importance descending, show top 20
    pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )[:20]

    features, values = zip(*pairs)

    fig = go.Figure(go.Bar(
        x=list(values),
        y=list(features),
        orientation="h",
        marker_color="#636EFA",
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=max(300, len(features) * 28),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=150),
    )
    return fig


def optuna_trial_figure(best_params: dict, trial_scores: list = None):
    """
    Shows Optuna search results.
    trial_scores is optional — if provided shows convergence curve.
    """
    if not best_params:
        return None

    # show best hyperparameters as a simple table
    params_df = pd.DataFrame([
        {"Parameter": k, "Best Value": str(v)}
        for k, v in best_params.items()
    ])

    fig = go.Figure(go.Table(
        header=dict(
            values=["Parameter", "Best Value"],
            fill_color="#262730",
            font=dict(color="white", size=13),
            align="left",
        ),
        cells=dict(
            values=[params_df["Parameter"], params_df["Best Value"]],
            fill_color="#0E1117",
            font=dict(color="white", size=12),
            align="left",
            height=30,
        ),
    ))
    fig.update_layout(
        title="Best Hyperparameters Found by Optuna",
        height=max(200, len(best_params) * 40 + 100),
        margin=dict(t=50, b=10),
    )
    return fig


def loss_curve(train_losses: list, val_losses: list, best_epoch: int = 0):
    if not train_losses and not val_losses:
        return None

    fig = go.Figure()
    if train_losses:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(train_losses) + 1)),
            y=train_losses,
            mode="lines+markers",
            name="Train Loss",
        ))
    if val_losses:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(val_losses) + 1)),
            y=val_losses,
            mode="lines+markers",
            name="Validation Loss",
        ))
    if best_epoch:
        fig.add_vline(
            x=best_epoch,
            line_dash="dash",
            line_color="#16A34A",
            annotation_text="Best epoch",
        )

    fig.update_layout(
        title="Training Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
    )
    return fig


def score_curve(train_scores: list, val_scores: list):
    if not train_scores and not val_scores:
        return None

    fig = go.Figure()
    if train_scores:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(train_scores) + 1)),
            y=train_scores,
            mode="lines+markers",
            name="Train Score",
        ))
    if val_scores:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(val_scores) + 1)),
            y=val_scores,
            mode="lines+markers",
            name="Validation Score",
        ))

    fig.update_layout(
        title="Training Score",
        xaxis_title="Epoch",
        yaxis_title="Score",
        hovermode="x unified",
    )
    return fig