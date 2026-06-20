import json
import os
from datetime import datetime

from automl_agent.core.state import AgentState

MEMORY_PATH = os.path.join("artifacts", "experiment_memory.jsonl")


def memory_retrieve_agent(state: AgentState) -> AgentState:
    state.current_step = "memory_retrieve"
    state.add_log("Memory: checking previous experiments.")
    if not os.path.exists(MEMORY_PATH):
        state.similar_experiments = []
        return state

    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        same_task = [r for r in rows if r.get("task_type") == state.dataset_info.task_type]
        state.similar_experiments = same_task[-3:]
        state.add_log(f"Memory: found {len(state.similar_experiments)} previous same-task runs.")
    except Exception as exc:
        state.add_log(f"Memory: retrieval skipped ({exc}).")
    return state


def memory_store_agent(state: AgentState) -> AgentState:
    state.current_step = "memory_store"
    state.add_log("Memory: storing experiment summary.")
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    try:
        row = {
            "created_at": datetime.utcnow().isoformat(),
            "dataset_name": state.dataset_info.name,
            "task_type": state.dataset_info.task_type,
            "architecture": state.ml_config.architecture if state.ml_config else "",
            "metrics": state.evaluation_result.metrics if state.evaluation_result else {},
            "approved": state.critic_feedback.approved if state.critic_feedback else False,
            "retry_count": state.retry_count,
            "mlflow_run_id": state.mlflow_run_id,
        }
        with open(MEMORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as exc:
        state.add_log(f"Memory: storage skipped ({exc}).")
    return state

