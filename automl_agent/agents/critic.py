import json
import time

from automl_agent.core.llm import get_llm
from automl_agent.core.state import (
    AgentState,
    CriticDecision,
    CriticFeedback,
)


def _rule_based_critic(state: AgentState) -> CriticDecision:
    """Pure fallback — only runs when LLM is unavailable."""
    ev   = state.evaluation_result
    arch = state.ml_config.architecture if state.ml_config else ""
    hp   = state.ml_config.hyperparams  if state.ml_config else {}
    task = state.dataset_info.task_type

    issues                = []
    score                 = 8.0
    retry_strategy        = ""
    suggested_hyperparams = {}

    # architecture-aware overfitting thresholds in code not in prompt
    thresholds = {
        "Forest": 0.35, "Trees": 0.35,
        "Boosting": 0.20, "XGBoost": 0.20, "LightGBM": 0.20,
        "DecisionTree": 0.25,
    }
    threshold = 0.15
    for key, val in thresholds.items():
        if key in arch:
            threshold = val
            break

    real_overfit = ev.overfit_gap > threshold

    if real_overfit:
        issues.append(f"Overfitting gap {ev.overfit_gap:.3f} exceeds {threshold} threshold.")
        score -= 1.5
        if "Forest" in arch or "Trees" in arch:
            suggested_hyperparams = {
                "max_depth":         max(3, hp.get("max_depth", 10) - 3),
                "min_samples_split": hp.get("min_samples_split", 2) + 3,
                "min_samples_leaf":  hp.get("min_samples_leaf", 1) + 1,
            }
            retry_strategy = "Constrain tree depth and increase min samples."
        elif any(k in arch for k in ("Boosting", "XGBoost", "LightGBM")):
            suggested_hyperparams = {
                "learning_rate": round(hp.get("learning_rate", 0.1) * 0.7, 4),
                "max_depth":     max(2, hp.get("max_depth", 4) - 1),
            }
            retry_strategy = "Lower learning rate and reduce depth."

    if not ev.passed:
        issues.append("Did not pass quality threshold.")
        score -= 1.5
        if not retry_strategy:
            retry_strategy = "Increase model capacity."

    # regression: strong R2 overrides everything
    if task == "regression" and ev.metrics.get("r2", 0) >= 0.75:
        approved       = True
        score          = max(score, 7.5)
        retry_strategy = ""
        issues         = [i for i in issues if "threshold" not in i.lower()]
    else:
        approved = score >= 6.5 and ev.passed and not real_overfit

    return CriticDecision(
        approved=approved,
        score=max(0.0, round(score, 2)),
        reasoning=(
            f"Score {score}/10. "
            + (f"Issues: {'; '.join(issues)}." if issues else "No major issues.")
        ),
        retry_strategy=retry_strategy,
        suggested_hyperparams=suggested_hyperparams,
    )


def _call_llm_with_retry(llm, prompt: str, max_attempts: int = 3) -> str:
    for attempt in range(max_attempts):
        try:
            response = llm.invoke(prompt).content.strip()
            if not response:
                raise ValueError("Empty response.")
            return response
        except Exception:
            if attempt == max_attempts - 1:
                raise
            time.sleep(1.5)
    raise ValueError("LLM failed after all retries.")


def critic_agent(state: AgentState) -> AgentState:
    state.current_step = "critic"
    state.add_log("Critic: reviewing result and deciding retry.")

    ev = state.evaluation_result

    if ev is None:
        state.critic_decision = CriticDecision(
            approved=False, score=0.0,
            reasoning="No evaluation result produced.",
            retry_strategy="Fix training and evaluation pipeline.",
            suggested_hyperparams={},
        )
        state.critic_feedback = CriticFeedback(
            approved=False, score=0,
            issues=["No evaluation result."],
            suggestions=["Fix pipeline."],
            retry_reason="Evaluation missing.",
        )
        state.should_retry = state.retry_count < state.max_retries
        return state

    llm = get_llm()

    if llm:
        arch = state.ml_config.architecture if state.ml_config else "unknown"
        task = state.dataset_info.task_type

        # model-specific context — facts, not rules
        model_context = (
            f"Architecture: {arch}\n"
            f"Current hyperparameters: {state.ml_config.hyperparams if state.ml_config else {}}\n"
        )

        if "Forest" in arch or "Trees" in arch:
            model_context += (
                "Context: Tree ensemble models (Random Forest, Extra Trees) "
                "always achieve near-perfect train scores because each tree "
                "sees a random subset. A train/val gap up to 0.35 is normal "
                f"behaviour for this model family. Current gap: {ev.overfit_gap:.4f}."
            )
        elif any(k in arch for k in ("Boosting", "XGBoost", "LightGBM")):
            model_context += (
                f"Context: {arch} is a boosting model. Gaps up to 0.20 are "
                f"typical. Current gap: {ev.overfit_gap:.4f}."
            )
        else:
            model_context += (
                f"Context: Current train/val gap: {ev.overfit_gap:.4f}."
            )

        prompt = f"""
You are a senior ML engineer doing a post-training review.
Make a holistic judgement about whether this result is good enough
or whether retraining with different hyperparameters would help.

Dataset:
- Name: {state.dataset_info.name}
- Task: {task}
- Rows: {state.dataset_info.n_samples}
- Features: {state.dataset_info.n_features}
- Imbalanced: {state.dataset_info.is_imbalanced}
- Target column: {state.dataset_info.target_column}

{model_context}

Evaluation results:
- Metrics: {ev.metrics}
- Overfitting detected by evaluator: {ev.is_overfitting}
- Overfitting gap (train - val score): {ev.overfit_gap:.4f}
- Passed quality threshold: {ev.passed}

Analyst identified risks: {state.analyst_insight.risks if state.analyst_insight else []}
Evaluator interpretation: {state.evaluator_insight.interpretation if state.evaluator_insight else "N/A"}
Evaluator critical failures: {state.evaluator_insight.critical_failures if state.evaluator_insight else []}

Past experiments on similar problems:
{state.similar_experiments[-2:] if state.similar_experiments else "None"}

Retries used so far: {state.retry_count} / {state.max_retries}

Think about:
- Is the overfitting gap genuinely harmful or just normal for this model?
- Are the metrics good enough for practical use in this domain?
- For medical/health datasets: is recall on the positive class acceptable?
  Missing a positive case is often worse than a false alarm.
- For regression: does the R2 explain enough variance to be useful?
  RMSE is only meaningful relative to the target variable range.
- If a retry is warranted, what SPECIFIC hyperparameter changes would
  actually address the root cause? Don't just suggest generic changes.
- Would a different architecture help more than hyperparameter tuning?

If you suggest a retry, reference the current hyperparameter values
and explain exactly what you're changing and why that specific change
addresses the problem you identified.

Respond ONLY in this exact JSON format, no markdown, no extra text:
{{
    "approved": true or false,
    "score": 7.5,
    "reasoning": "3-4 sentences with specific metric values, domain context, and clear justification",
    "retry_strategy": "specific explanation of what to change and the expected effect (empty string if approved)",
    "suggested_hyperparams": {{
        "param_name": value
    }}
}}
"""
        try:
            response = _call_llm_with_retry(llm, prompt)
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            parsed   = json.loads(response.strip())

            # guardrails in code — not in prompt
            if not isinstance(parsed.get("approved"), bool):
                parsed["approved"] = False
            parsed["score"] = max(0.0, min(10.0, float(parsed.get("score", 5.0))))

            decision = CriticDecision(**parsed)
            state.critic_decision = decision
            state.add_log(
                f"Critic LLM: approved={decision.approved}, "
                f"score={decision.score}. {decision.reasoning}"
            )

        except Exception as exc:
            state.add_log(f"Critic: LLM failed ({exc}), using rule-based fallback.")
            state.critic_decision = _rule_based_critic(state)
    else:
        state.add_log("Critic: no API key, using rule-based fallback.")
        state.critic_decision = _rule_based_critic(state)

    decision = state.critic_decision

    # retry logic pure code, no LLM 
    if not decision.approved and state.retry_count < state.max_retries:
        state.retry_count += 1
        state.should_retry = True

        if decision.suggested_hyperparams and state.ml_config:
            state.ml_config.hyperparams.update(decision.suggested_hyperparams)
            state.add_log(
                f"Critic: applied hyperparams: {decision.suggested_hyperparams}."
            )
        elif state.ml_config:
            hp   = state.ml_config.hyperparams
            arch = state.ml_config.architecture
            if ev.is_overfitting:
                if "Forest" in arch or "Trees" in arch:
                    hp["max_depth"]         = max(3,  hp.get("max_depth", 10) - 3)
                    hp["min_samples_split"] = min(10, hp.get("min_samples_split", 2) + 2)
                    hp["min_samples_leaf"]  = min(5,  hp.get("min_samples_leaf", 1) + 1)
                elif any(k in arch for k in ("Boosting", "XGBoost", "LightGBM")):
                    hp["learning_rate"] = round(hp.get("learning_rate", 0.1) * 0.7, 4)
                    hp["max_depth"]     = max(2, hp.get("max_depth", 4) - 1)
            else:
                if "Forest" in arch or "Trees" in arch:
                    hp["n_estimators"] = min(500, hp.get("n_estimators", 200) + 100)
                elif any(k in arch for k in ("Boosting", "XGBoost", "LightGBM")):
                    hp["n_estimators"]  = min(500, hp.get("n_estimators", 100) + 100)
                    hp["learning_rate"] = round(hp.get("learning_rate", 0.1) * 0.8, 4)

        state.add_log(
            f"Critic: retry {state.retry_count}/{state.max_retries}. "
            f"Strategy: {decision.retry_strategy}"
        )
    else:
        state.should_retry = False
        state.add_log(
            f"Critic: approved={decision.approved}, score={decision.score}."
        )

    state.critic_feedback = CriticFeedback(
        approved=decision.approved,
        score=decision.score,
        issues=[decision.reasoning],
        suggestions=[decision.retry_strategy] if decision.retry_strategy else [],
        retry_reason=decision.retry_strategy,
    )
    return state