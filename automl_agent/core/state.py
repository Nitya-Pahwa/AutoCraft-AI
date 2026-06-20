from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic import field_validator

class DatasetInfo(BaseModel):
    path: str
    name: str = "dataset"
    target_column: str
    task_type: str = ""
    n_samples: int = 0
    n_features: int = 0
    n_classes: Optional[int] = None
    class_balance: Dict[str, float] = Field(default_factory=dict)
    is_imbalanced: bool = False
    feature_types: Dict[str, str] = Field(default_factory=dict)
    missing_ratio: float = 0.0


class ModelConfig(BaseModel):
    architecture: str = "MLP"
    reason: str = ""
    input_dim: int = 0
    output_dim: int = 1
    hyperparams: Dict[str, Any] = Field(default_factory=dict)


class TrainingResult(BaseModel):
    train_losses: List[float] = Field(default_factory=list)
    val_losses: List[float] = Field(default_factory=list)
    train_scores: List[float] = Field(default_factory=list)
    val_scores: List[float] = Field(default_factory=list)
    trial_scores: List[float] = Field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = 0.0
    best_params: Dict[str, Any] = Field(default_factory=dict)
    model_path: str = ""
    preprocessor_path: str = ""
    training_time_s: float = 0.0


class EvaluationResult(BaseModel):
    metrics: Dict[str, float] = Field(default_factory=dict)
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: str = ""
    is_overfitting: bool = False
    overfit_gap: float = 0.0
    passed: bool = False


class CriticFeedback(BaseModel):
    approved: bool = False
    score: float = 0.0
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    retry_reason: str = ""


class PlannerDecision(BaseModel):
    optuna_trials: int = 10
    max_retries: int = 1
    risk_flags: List[str] = Field(default_factory=list)
    strategy_notes: str = ""

class AnalystInsight(BaseModel):
    task_type: str = ""
    interpretation: str = ""
    risks: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

class SelectorDecision(BaseModel):
    architecture: str = ""
    reasoning: str = ""
    hyperparams: Dict[str, Any] = Field(default_factory=dict)
    alternatives_considered: List[str] = Field(default_factory=list)

class EvaluatorInsight(BaseModel):
    metrics: Dict[str, float] = Field(default_factory=dict)
    interpretation: str = ""
    critical_failures: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)

class CriticDecision(BaseModel):
    approved: bool = False
    score: float = 0.0
    reasoning: str = ""
    retry_strategy: str = ""
    suggested_hyperparams: Dict[str, Any] = Field(default_factory=dict)

class FinalReport(BaseModel):
    full_markdown: str = ""


class ColumnProfile(BaseModel):
    name:         str
    dtype:        str   = ""
    col_type:     str   = ""
    is_converted: bool  = False
    n_missing:    int   = 0
    missing_pct:  float = 0.0
    n_unique:     int   = 0
    unique_pct:   float = 0.0
    stats:        Dict[str, Any] = Field(default_factory=dict)
    top_values:   Dict[str, int] = Field(default_factory=dict)
    outlier_count: int = 0
    outlier_pct:   float = 0.0
    outlier_lower_bound: Optional[float] = None
    outlier_upper_bound: Optional[float] = None

    @field_validator("top_values", mode="before")
    @classmethod
    def ensure_dict(cls, v):
        if isinstance(v, dict):
            return v
        if isinstance(v, (list, tuple)) and len(v) == 0:
            return {}
        return {}

    @field_validator("stats", mode="before")
    @classmethod
    def ensure_stats_dict(cls, v):
        if isinstance(v, dict):
            return v
        return {}


class DatasetProfile(BaseModel):
    n_rows:            int   = 0
    n_cols:            int   = 0
    n_duplicates:      int   = 0
    duplicate_pct:     float = 0.0
    total_missing:     int   = 0
    total_missing_pct: float = 0.0
    null_summary:      Dict[str, Any]              = Field(default_factory=dict)
    outlier_summary:   Dict[str, Any]              = Field(default_factory=dict)
    column_profiles:   Dict[str, ColumnProfile]    = Field(default_factory=dict)
    target_profile:    Optional[ColumnProfile]     = None


class AgentState(BaseModel):
    dataset_info: DatasetInfo
    ml_config: Optional[ModelConfig] = None
    training_result: Optional[TrainingResult] = None
    evaluation_result: Optional[EvaluationResult] = None
    critic_feedback: Optional[CriticFeedback] = None
    similar_experiments: List[Dict[str, Any]] = Field(default_factory=list)
    experiment_name: str = "agentic_automl"
    retry_count: int = 0
    max_retries: int = 1
    should_retry: bool = False
    mlflow_run_id: str = ""
    current_step: str = "idle"
    logs: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    planner_decision: Optional[PlannerDecision] = None
    analyst_insight: Optional[AnalystInsight] = None
    selector_decision: Optional[SelectorDecision] = None
    evaluator_insight: Optional[EvaluatorInsight] = None
    critic_decision: Optional[CriticDecision] = None
    final_report: Optional[FinalReport] = None
    dataset_profile: Optional[DatasetProfile] = None
    preprocessing_plan: List[Dict[str, Any]] = Field(default_factory=list)

    def add_log(self, message: str) -> "AgentState":
        self.logs.append(message)
        return self
