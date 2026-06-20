import json

from automl_agent.core.llm import get_llm
from automl_agent.core.models import (
    ALL_MODELS,
    CLASSIFICATION_MODELS,
    LIGHTGBM_AVAILABLE,
    REGRESSION_MODELS,
    XGBOOST_AVAILABLE,
)
from automl_agent.core.state import AgentState, ModelConfig, SelectorDecision


def _rule_based_selector(state: AgentState) -> tuple:
    """Pure fallback — only runs when LLM is unavailable."""
    info = state.dataset_info
    task = info.task_type
    n    = info.n_samples

    # reuse past approved architecture if available
    for exp in state.similar_experiments:
        if exp.get("approved") and exp.get("task_type") == task:
            arch = exp.get("architecture", "")
            if arch in ALL_MODELS:
                return (
                    arch,
                    exp.get("hyperparams", {}),
                    f"Reusing approved architecture from past experiment: {arch}.",
                    ["Other models — past run already succeeded with this one"],
                )

    if task == "classification":
        if n < 1000:
            arch = "XGBoost" if XGBOOST_AVAILABLE else "RandomForest"
        elif n < 10000:
            arch = "LightGBM" if LIGHTGBM_AVAILABLE else "HistGradientBoosting"
        else:
            arch = "LightGBM" if LIGHTGBM_AVAILABLE else "HistGradientBoosting"
    else:
        if n < 1000:
            arch = "XGBoostRegressor" if XGBOOST_AVAILABLE else "RandomForestRegressor"
        elif n < 10000:
            arch = "LightGBMRegressor" if LIGHTGBM_AVAILABLE else "HistGradientBoostingRegressor"
        else:
            arch = "LightGBMRegressor" if LIGHTGBM_AVAILABLE else "HistGradientBoostingRegressor"

    return (
        arch,
        {},
        f"Rule-based fallback selected {arch} for {n}-row {task} dataset.",
        [],
    )


def selector_agent(state: AgentState) -> AgentState:
    state.current_step = "selection"
    state.add_log("Selector: choosing model architecture.")

    info = state.dataset_info
    task = info.task_type

    available = (
        sorted(CLASSIFICATION_MODELS)
        if task == "classification"
        else sorted(REGRESSION_MODELS)
    )

    past_summary = [
        {
            "architecture":     e.get("architecture"),
            "metrics":          e.get("metrics", {}),
            "approved":         e.get("approved", False),
            "n_samples":        e.get("n_samples"),
            "n_features":       e.get("n_features"),
            "critic_reasoning": e.get("critic_reasoning", ""),
            "analyst_risks":    e.get("analyst_risks", []),
        }
        for e in state.similar_experiments
    ]

    analyst_risks = state.analyst_insight.risks           if state.analyst_insight else []
    analyst_recs  = state.analyst_insight.recommendations if state.analyst_insight else []
    analyst_interp= state.analyst_insight.interpretation  if state.analyst_insight else ""

    llm = get_llm()

    if llm:
        prompt = f"""
You are a senior ML engineer choosing a model for a tabular dataset.
Study the full context below and make the best choice you can.

Dataset profile:
- Rows: {info.n_samples}
- Features: {info.n_features}
- Task: {task}
- Is imbalanced: {info.is_imbalanced}
- Class balance: {info.class_balance}
- Missing ratio: {info.missing_ratio}
- Feature types: {info.feature_types}

Analyst interpretation: {analyst_interp}
Analyst risks: {analyst_risks}
Analyst recommendations: {analyst_recs}
Planner strategy: {state.planner_decision.strategy_notes if state.planner_decision else "N/A"}
Planner risk flags: {state.planner_decision.risk_flags if state.planner_decision else []}

Past experiments on similar tasks:
{json.dumps(past_summary, indent=2) if past_summary else "None — this is the first run."}

Available models for {task}:
{json.dumps(available, indent=2)}

Available boosting libraries:
- XGBoost available: {XGBOOST_AVAILABLE}
- LightGBM available: {LIGHTGBM_AVAILABLE}

Think carefully about:
- What properties of this dataset make certain models better or worse?
- Does the missing ratio affect which models can handle the data?
- Does class imbalance change what model properties matter?
- What did past experiments reveal — which architectures failed and why?
- Would a simpler model be safer given the dataset size?
- What starting hyperparameters make sense for this specific dataset size?

Also consider what the analyst flagged — if the analyst said
overfitting is a risk, pick a model with built-in regularization.
If the analyst flagged high cardinality, pick a model that handles
it natively rather than relying on encoding.

Provide complete starting hyperparameters appropriate for the 
dataset size. These will be the starting point for Optuna search,
so sensible initial values help the search converge faster.

Respond ONLY in this exact JSON format, no markdown, no extra text:
{{
    "architecture": "one name from the available list above",
    "reasoning": "3-4 sentences explaining your choice, referencing specific dataset properties and past experiment outcomes",
    "alternatives_considered": [
        "ModelName — specific reason it was considered but not chosen"
    ],
    "hyperparams": {{
        "n_estimators": 200
    }}
}}
"""
        try:
            response = llm.invoke(prompt).content.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()

            parsed       = json.loads(response)
            architecture = parsed.get("architecture", "")

            # guardrail in code — validate name, don't list rules in prompt
            if architecture not in ALL_MODELS:
                state.add_log(
                    f"Selector: LLM returned unknown model '{architecture}' "
                    f"— falling back to rule-based."
                )
                architecture, hyperparams, reasoning, alternatives = \
                    _rule_based_selector(state)
            else:
                hyperparams  = parsed.get("hyperparams", {})
                reasoning    = parsed.get("reasoning", "")
                alternatives = parsed.get("alternatives_considered", [])

            state.selector_decision = SelectorDecision(
                architecture=architecture,
                reasoning=reasoning,
                hyperparams=hyperparams,
                alternatives_considered=alternatives,
            )
            state.add_log(f"Selector LLM: chose {architecture}. {reasoning}")

        except Exception as exc:
            state.add_log(f"Selector: LLM failed ({exc}), using rule-based fallback.")
            architecture, hyperparams, reasoning, alternatives = \
                _rule_based_selector(state)
            state.selector_decision = SelectorDecision(
                architecture=architecture,
                reasoning=reasoning,
                hyperparams=hyperparams,
                alternatives_considered=alternatives,
            )
    else:
        state.add_log("Selector: no API key, using rule-based fallback.")
        architecture, hyperparams, reasoning, alternatives = \
            _rule_based_selector(state)
        state.selector_decision = SelectorDecision(
            architecture=architecture,
            reasoning=reasoning,
            hyperparams=hyperparams,
            alternatives_considered=alternatives,
        )

    output_dim = info.n_classes if task == "classification" else 1
    state.ml_config = ModelConfig(
        architecture=state.selector_decision.architecture,
        reason=state.selector_decision.reasoning,
        input_dim=info.n_features,
        output_dim=output_dim or 1,
        hyperparams=state.selector_decision.hyperparams,
    )

    state.add_log(
        f"Selector: final={state.selector_decision.architecture}, "
        f"hyperparams={state.selector_decision.hyperparams}."
    )
    return state