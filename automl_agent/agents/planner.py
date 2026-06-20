import json
import time

from automl_agent.core.llm import get_llm
from automl_agent.core.state import AgentState, PlannerDecision


def _rule_based_planner(state: AgentState) -> PlannerDecision:
    """Pure fallback — only runs when LLM is unavailable."""
    risk_flags    = []
    optuna_trials = 10
    max_retries   = 1

    if state.similar_experiments:
        past_approved = [e for e in state.similar_experiments if e.get("approved")]
        if past_approved:
            optuna_trials = 5
            risk_flags.append(
                f"{len(past_approved)} past approved run(s) found — "
                "starting with fewer Optuna trials."
            )
        failed = [e for e in state.similar_experiments if not e.get("approved")]
        if len(failed) >= 2:
            max_retries = 2
            risk_flags.append(
                "Multiple past failures detected — increasing retry budget."
            )

    return PlannerDecision(
        optuna_trials=optuna_trials,
        max_retries=max_retries,
        risk_flags=risk_flags,
        strategy_notes=(
            f"Rule-based fallback: {optuna_trials} trials, "
            f"{max_retries} retries."
        ),
    )


def _call_llm_with_retry(llm, prompt: str, max_attempts: int = 3) -> str:
    for attempt in range(max_attempts):
        try:
            response = llm.invoke(prompt).content.strip()
            if not response:
                raise ValueError("Empty response from LLM.")
            return response
        except Exception:
            if attempt == max_attempts - 1:
                raise
            time.sleep(1.5)
    raise ValueError("LLM failed after all retries.")


def planner_agent(state: AgentState) -> AgentState:
    state.current_step = "planning"
    state.add_log("Planner: forming execution strategy.")

    past_summary = [
        {
            "dataset":        e.get("dataset_name"),
            "architecture":   e.get("architecture"),
            "metrics":        e.get("metrics", {}),
            "approved":       e.get("approved", False),
            "retry_count":    e.get("retry_count", 0),
            "critic_reasoning": e.get("critic_reasoning", ""),
            "analyst_risks":  e.get("analyst_risks", []),
            "planner_flags":  e.get("planner_flags", []),
        }
        for e in state.similar_experiments
    ]

    llm = get_llm()

    if llm:
        prompt = f"""
You are an experienced AutoML pipeline strategist.
Your job is to decide how much effort to invest in this experiment
before training begins.

What you know right now:
- Dataset name: {state.dataset_info.name}
- Target column: {state.dataset_info.target_column}

Note: Dataset statistics like row count and feature count are not
yet available — the analyst runs after you. Do not invent them.

Past experiments on similar tasks:
{json.dumps(past_summary, indent=2) if past_summary else "None — this is the first run on this type of problem."}

Think about:
- Do past experiments suggest this problem is easy or hard?
- Did past runs fail repeatedly? That signals we need more retries.
- Did a past run succeed quickly? That signals fewer trials are needed.
- Does the dataset name hint at anything? (medical, fraud, spam, churn
  are typically harder and benefit from more search effort)
- What risks can you actually infer without seeing the data yet?

Your output controls three things:
- optuna_trials: how many hyperparameter search trials (range 5-50)
- max_retries: how many critic-triggered retrains to allow (range 1-5)
- risk_flags: genuine risks you can infer right now — not guesses
- strategy_notes: your reasoning in plain language

Respond ONLY in this exact JSON format, no markdown, no extra text:
{{
    "optuna_trials": 10,
    "max_retries": 1,
    "risk_flags": [],
    "strategy_notes": "your reasoning here"
}}
"""
        try:
            response = _call_llm_with_retry(llm, prompt)
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            parsed = json.loads(response.strip())

            # guardrails in code — not in the prompt
            parsed["optuna_trials"] = max(5,  min(50, int(parsed.get("optuna_trials", 10))))
            parsed["max_retries"]   = max(1,  min(5,  int(parsed.get("max_retries", 1))))

            state.planner_decision = PlannerDecision(**parsed)
            state.max_retries      = state.planner_decision.max_retries
            state.add_log(f"Planner LLM: {state.planner_decision.strategy_notes}")

        except Exception as exc:
            state.add_log(f"Planner: LLM failed ({exc}), using rule-based fallback.")
            state.planner_decision = _rule_based_planner(state)
            state.max_retries      = state.planner_decision.max_retries
    else:
        state.add_log("Planner: no API key, using rule-based fallback.")
        state.planner_decision = _rule_based_planner(state)
        state.max_retries      = state.planner_decision.max_retries

    state.add_log(
        f"Planner: optuna_trials={state.planner_decision.optuna_trials}, "
        f"max_retries={state.planner_decision.max_retries}, "
        f"risk_flags={state.planner_decision.risk_flags}."
    )
    return state