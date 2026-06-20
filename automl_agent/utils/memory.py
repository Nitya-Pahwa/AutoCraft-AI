import json
import os
from datetime import datetime
from typing import Any, Dict, List

from automl_agent.core.state import AgentState

MEMORY_PATH = os.path.join("artifacts", "experiment_memory.jsonl")


def _load_all() -> List[Dict[str, Any]]:
    if not os.path.exists(MEMORY_PATH):
        return []
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception:
        return []


def _similarity_score(exp: Dict, dataset_name: str, task_type: str) -> float:
    """
    Score how similar a past experiment is to the current one.
    Higher is more similar.

    Scoring:
    - Same dataset name          : +10  (most important signal)
    - Same task type             : +3
    - Similar row count (±30%)   : +2
    - Similar feature count (±30%): +1
    """
    score = 0.0

    # same dataset name — strongest signal
    past_name    = exp.get("dataset_name", "").lower().strip()
    current_name = dataset_name.lower().strip()
    if past_name == current_name:
        score += 10.0
    elif past_name and current_name:
        # partial match — e.g. "heart_disease" matches "heart_disease_data"
        if past_name in current_name or current_name in past_name:
            score += 5.0

    # same task type
    if exp.get("task_type") == task_type:
        score += 3.0

    return score


def _find_similar(
    rows: List[Dict],
    dataset_name: str,
    task_type: str,
    n: int = 3,
    min_score: float = 3.0,
) -> List[Dict]:
    """
    Return up to n most similar past experiments.
    Only returns experiments with similarity score >= min_score.
    min_score=3.0 means at minimum the task_type must match.
    """
    scored = []
    for row in rows:
        score = _similarity_score(row, dataset_name, task_type)
        if score >= min_score:
            scored.append((score, row))

    # sort by score descending, then by recency
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:n]]


def memory_retrieve_agent(state: AgentState) -> AgentState:
    state.current_step = "memory_retrieve"
    state.add_log("Memory: loading past experiments.")

    rows = _load_all()

    if not rows:
        state.similar_experiments = []
        state.add_log("Memory: no past experiments on record yet.")
        return state

    similar = _find_similar(
        rows,
        dataset_name=state.dataset_info.name,
        task_type=state.dataset_info.task_type,
    )

    state.similar_experiments = similar

    if not similar:
        state.add_log(
            f"Memory: {len(rows)} total experiments found but none "
            f"similar to '{state.dataset_info.name}' ({state.dataset_info.task_type})."
        )
        return state

    # find best past result for logging
    approved_exps = [e for e in similar if e.get("approved")]
    same_dataset  = [e for e in similar if e.get("dataset_name", "").lower() == state.dataset_info.name.lower()]

    if same_dataset:
        best = max(
            same_dataset,
            key=lambda e: e.get("metrics", {}).get("accuracy", 0)
                          or e.get("metrics", {}).get("r2", 0),
        )
        state.add_log(
            f"Memory: found {len(same_dataset)} past run(s) on "
            f"'{state.dataset_info.name}'. "
            f"Best: {best.get('architecture')} → "
            f"metrics={best.get('metrics', {})} "
            f"approved={best.get('approved')}."
        )
    elif approved_exps:
        best = max(
            approved_exps,
            key=lambda e: e.get("metrics", {}).get("accuracy", 0)
                          or e.get("metrics", {}).get("r2", 0),
        )
        state.add_log(
            f"Memory: found {len(similar)} similar experiment(s) "
            f"(same task, different dataset). "
            f"Best approved: {best.get('architecture')} on "
            f"'{best.get('dataset_name')}' → metrics={best.get('metrics', {})}."
        )
    else:
        state.add_log(
            f"Memory: found {len(similar)} similar experiment(s) "
            f"but none were approved."
        )

    return state


def memory_store_agent(state: AgentState) -> AgentState:
    state.current_step = "memory_store"
    state.add_log("Memory: storing experiment summary.")

    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)

    try:
        row = {
            "created_at":       datetime.utcnow().isoformat(),
            "dataset_name":     state.dataset_info.name,
            "task_type":        state.dataset_info.task_type,
            "n_samples":        state.dataset_info.n_samples,
            "n_features":       state.dataset_info.n_features,
            "is_imbalanced":    state.dataset_info.is_imbalanced,
            "architecture":     state.ml_config.architecture  if state.ml_config  else "",
            "hyperparams":      state.ml_config.hyperparams   if state.ml_config  else {},
            "metrics":          state.evaluator_insight.metrics if state.evaluator_insight else {},
            "approved":         state.critic_decision.approved  if state.critic_decision  else False,
            "critic_score":     state.critic_decision.score     if state.critic_decision  else 0.0,
            "retry_count":      state.retry_count,
            "mlflow_run_id":    state.mlflow_run_id,
            "best_val_loss":    state.training_result.best_val_loss   if state.training_result else None,
            "training_time":    state.training_result.training_time_s if state.training_result else None,
            "planner_flags":    state.planner_decision.risk_flags      if state.planner_decision  else [],
            "analyst_risks":    state.analyst_insight.risks            if state.analyst_insight   else [],
            "critic_reasoning": state.critic_decision.reasoning        if state.critic_decision   else "",
        }

        with open(MEMORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        state.add_log(
            f"Memory: stored '{state.dataset_info.name}' — "
            f"approved={row['approved']}, "
            f"metrics={row['metrics']}."
        )

    except Exception as exc:
        state.add_log(f"Memory: storage failed ({exc}).")

    return state


def get_experiment_history() -> List[Dict[str, Any]]:
    """Public helper for Streamlit UI."""
    return _load_all()


def clear_memory() -> None:
    """Wipe all experiment memory — useful for testing."""
    if os.path.exists(MEMORY_PATH):
        os.remove(MEMORY_PATH)