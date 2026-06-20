import os
import sys
import pandas as pd
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from automl_agent.utils.data_loader import get_columns, save_uploaded_file, validate_dataset
from automl_agent.utils.memory import get_experiment_history
from automl_agent.utils.visualizer import (
    confusion_matrix_figure,
    feature_importance_figure,
    loss_curve,
    optuna_trial_figure,
    score_curve,
)


st.set_page_config(page_title="Agentic AutoML", layout="wide")
st.title("Agentic AutoML Pipeline")


@st.cache_resource(show_spinner="Loading ML pipeline...")
def get_pipeline_runner():
    from automl_agent.core.graph import run_pipeline
    return run_pipeline


# sidebar 
with st.sidebar:
    st.header("Past Experiments")
    history = get_experiment_history()
    if history:
        for exp in reversed(history[-5:]):
            icon = "✅" if exp.get("approved") else "❌"
            st.markdown(
                f"{icon} **{exp.get('dataset_name', 'unknown')}**  \n"
                f"`{exp.get('architecture', '?')}` | "
                f"`{exp.get('task_type', '?')}`"
            )
            metrics = exp.get("metrics", {})
            if metrics:
                st.json(metrics)
            st.divider()
    else:
        st.caption("No past experiments yet.")


# upload
uploaded = st.file_uploader(
    "Upload a CSV or Excel dataset",
    type=["csv", "xlsx", "xls"],
)

if not uploaded:
    st.caption(
        "Upload a CSV or Excel dataset to begin. "
        "Version 1 supports classification and regression on tabular data."
    )
    st.stop()


# config
path          = save_uploaded_file(uploaded)
columns       = get_columns(path)
target_column = st.selectbox("Target column", columns)
max_retries   = st.slider("Critic retries", 0, 2, 1)

valid, message, preview = validate_dataset(path, target_column)
st.info(message)
if preview is not None:
    st.dataframe(preview.head(10), use_container_width=True)

if not valid:
    st.stop()

if not st.button("Run AutoML Pipeline", type="primary"):
    st.stop()


# tabs
run_pipeline = get_pipeline_runner()

st.subheader("Live Pipeline Logs")
log_box   = st.empty()
live_logs = []

(
    tab_plan, tab_analyst, tab_selector,
    tab_train, tab_eval, tab_critic, tab_report
) = st.tabs([
    "🗺 Planner", "🔬 Analyst", "🧠 Selector",
    "🏋 Training", "📊 Evaluator", "⚖ Critic", "📄 Report",
])

# one placeholder per tab each agent writes only to its own
plan_ph     = tab_plan.empty()
analyst_ph  = tab_analyst.empty()
selector_ph = tab_selector.empty()
train_ph    = tab_train.empty()
eval_ph     = tab_eval.empty()
critic_ph   = tab_critic.empty()
report_ph   = tab_report.empty()

for ph, name in [
    (plan_ph,     "Planner"),
    (analyst_ph,  "Analyst"),
    (selector_ph, "Selector"),
    (train_ph,    "Trainer"),
    (eval_ph,     "Evaluator"),
    (critic_ph,   "Critic"),
    (report_ph,   "Reporter"),
]:
    ph.caption(f"⏳ Waiting for {name} to run...")


def on_progress(node_name: str, state):
    # live log
    if state.logs:
        live_logs.append(f"[{node_name}]  {state.logs[-1]}")
        log_box.code("\n".join(live_logs[-30:]))

    # planner 
    if node_name == "planner" and state.planner_decision:
        d = state.planner_decision
        with plan_ph.container():
            st.subheader("Planner Decision")
            c1, c2 = st.columns(2)
            c1.metric("Optuna Trials", d.optuna_trials)
            c2.metric("Max Retries",   d.max_retries)
            st.markdown("**Strategy**")
            st.write(d.strategy_notes)
            

    # analyst 
    elif node_name == "analyst":
        with analyst_ph.container():
            # LLM interpretation
            if state.analyst_insight:
                a = state.analyst_insight
                st.subheader("Dataset Interpretation")
                st.write(a.interpretation)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Risks Identified**")
                    for r in (a.risks or ["None"]):
                        st.markdown(f"- {r}")
                with col2:
                    st.markdown("**Recommendations**")
                    for r in (a.recommendations or ["None"]):
                        st.markdown(f"- {r}")

            # dataset quality
            if state.dataset_profile:
                dp = state.dataset_profile
                st.divider()
                st.subheader("Dataset Quality")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows",           dp.n_rows)
                c2.metric("Columns",        dp.n_cols)
                c3.metric("Duplicates",     f"{dp.n_duplicates} ({dp.duplicate_pct}%)")
                c4.metric("Missing Cells",  f"{dp.total_missing} ({dp.total_missing_pct}%)")

                if dp.null_summary:
                    st.markdown("**Columns with Missing Values**")
                    st.dataframe(
                        pd.DataFrame([
                            {
                                "Column":        col,
                                "Missing Count": v["count"],
                                "Missing %":     f"{v['pct']}%",
                            }
                            for col, v in dp.null_summary.items()
                        ]),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.success("No missing values in any column.")

                # column analysis table
                if dp.column_profiles:
                    st.divider()
                    st.subheader("Column Analysis")
                    col_rows = []
                    for col, profile in dp.column_profiles.items():
                        row = {
                            "Column":  col,
                            "Type":    profile.col_type
                                       + (" ← from string" if profile.is_converted else ""),
                            "Missing": f"{profile.n_missing} ({profile.missing_pct}%)",
                            "Unique":  f"{profile.n_unique} ({profile.unique_pct}%)",
                        }
                        if profile.col_type == "numeric" and profile.stats:
                            row["Mean"]   = profile.stats.get("mean", "-")
                            row["Median"] = profile.stats.get("median", "-")
                            row["Min"]    = profile.stats.get("min", "-")
                            row["Max"]    = profile.stats.get("max", "-")
                        else:
                            row["Top Values"] = ", ".join(
                                list(profile.top_values.keys())[:3]
                            )
                        col_rows.append(row)
                    st.dataframe(
                        pd.DataFrame(col_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                # target profile
                if dp.target_profile:
                    st.divider()
                    st.subheader(f"Target Column: `{state.dataset_info.target_column}`")
                    tp = dp.target_profile
                    t1, t2, t3 = st.columns(3)
                    t1.metric("Type",    tp.col_type)
                    t2.metric("Unique",  tp.n_unique)
                    t3.metric("Missing", f"{tp.missing_pct}%")
                    if tp.stats:
                        st.json(tp.stats)
                    elif tp.top_values:
                        st.bar_chart(tp.top_values)

            # preprocessing plan
            if hasattr(state, "preprocessing_plan") and state.preprocessing_plan:
                st.divider()
                st.subheader("Preprocessing Plan")
                st.caption(
                    "Exact transformations applied to each column before training."
                )
                plan_df = pd.DataFrame(state.preprocessing_plan)

                def color_treatment(val):
                    colors = {
                        "Numeric":             "background-color: #1a472a; color: white",
                        "Numeric (extracted)": "background-color: #2d6a4f; color: white",
                        "One-Hot Encoded":     "background-color: #1d3461; color: white",
                        "Frequency Encoded":   "background-color: #3d2b56; color: white",
                        "TF-IDF":              "background-color: #6b2737; color: white",
                    }
                    return colors.get(val, "")

                styled = plan_df.style.map(
                    color_treatment, subset=["treatment"]
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)

    # selector 
    elif node_name == "selector" and state.selector_decision:
        s = state.selector_decision
        with selector_ph.container():
            st.subheader("Model Selection")
            st.metric("Selected Architecture", s.architecture)
            st.markdown("**Reasoning**")
            st.write(s.reasoning)
            if s.alternatives_considered:
                st.markdown("**Alternatives Considered**")
                for a in s.alternatives_considered:
                    st.markdown(f"- {a}")
            st.markdown("**Starting Hyperparameters**")
            st.json(s.hyperparams)

    # trainer
    elif node_name == "trainer" and state.training_result:
        t = state.training_result
        with train_ph.container():
            st.subheader("Training Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Architecture",
                      state.ml_config.architecture if state.ml_config else "N/A")
            c2.metric("Train Score",
                      f"{t.train_scores[-1]:.4f}" if t.train_scores else "N/A")
            c3.metric("Val Score",
                      f"{t.val_scores[-1]:.4f}"   if t.val_scores   else "N/A")
            c4.metric("Training Time", f"{t.training_time_s:.1f}s")

            st.markdown("**Best Hyperparameters Found by Optuna**")
            if t.best_params:
                fig_optuna = optuna_trial_figure(t.best_params)
                if fig_optuna:
                    st.plotly_chart(fig_optuna, use_container_width=True)
                else:
                    st.json(t.best_params)

            # epoch curves for neural nets only
            is_sklearn = len(t.train_losses) <= 1
            if not is_sklearn:
                fig_loss = loss_curve(t.train_losses, t.val_losses, t.best_epoch)
                if fig_loss:
                    st.plotly_chart(fig_loss, use_container_width=True)
                if state.dataset_info.task_type == "classification":
                    fig_score = score_curve(t.train_scores, t.val_scores)
                    if fig_score:
                        st.plotly_chart(fig_score, use_container_width=True)

    # evaluator
    elif node_name == "evaluator":
        with eval_ph.container():
            if state.evaluator_insight:
                e = state.evaluator_insight
                st.subheader("Evaluation Insight")
                st.write(e.interpretation)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Strengths**")
                    for s_ in (e.strengths or ["None identified"]):
                        st.markdown(f"- {s_}")
                with col2:
                    st.markdown("**Critical Failures**")
                    for f_ in (e.critical_failures or ["None identified"]):
                        st.markdown(f"- {f_}")

            if state.evaluation_result:
                st.divider()
                st.subheader("Metrics")
                st.json(state.evaluation_result.metrics)
                ev = state.evaluation_result
                c1, c2, c3 = st.columns(3)
                if (
                    ev.confusion_matrix
                    and state.dataset_info.task_type == "classification"
                ):
                    fig_cm = confusion_matrix_figure(ev.confusion_matrix)
                    if fig_cm:
                        st.plotly_chart(fig_cm, use_container_width=True)
                c1.metric("Overfit Gap",      f"{ev.overfit_gap:.4f}")
                c2.metric("Overfitting",      "Yes ⚠" if ev.is_overfitting else "No ✅")
                c3.metric("Passed Threshold", "Yes ✅" if ev.passed        else "No ❌")

    # critic 
    elif node_name == "critic" and state.critic_decision:
        c = state.critic_decision
        with critic_ph.container():
            st.subheader("Critic Decision")
            status = "✅ Approved" if c.approved else "🔄 Retry Triggered"
            c1, c2 = st.columns(2)
            c1.metric("Decision", status)
            c2.metric("Score",    f"{c.score} / 10")
            st.markdown("**Reasoning**")
            st.write(c.reasoning)
            if c.retry_strategy:
                st.markdown("**Retry Strategy**")
                st.warning(c.retry_strategy)
                if c.suggested_hyperparams:
                    st.markdown("**Suggested Hyperparameter Changes**")
                    st.json(c.suggested_hyperparams)
            else:
                st.success("Model approved — no retry needed.")

    # reporter 
    elif node_name == "reporter" and state.final_report:
        if state.final_report.full_markdown:
            with report_ph.container():
                st.subheader("Experiment Report")
                st.markdown(state.final_report.full_markdown)


# ── run ───────────────────────────────────────────────────────────────────────
with st.spinner("Running agentic ML pipeline..."):
    result = run_pipeline(
        dataset_path=path,
        target_column=target_column,
        dataset_name=os.path.splitext(uploaded.name)[0],
        max_retries=max_retries,
        progress_callback=on_progress,
    )


# ── post-run ──────────────────────────────────────────────────────────────────
if result.error:
    st.error(f"Pipeline failed: {result.error}")
    with st.expander("Full logs"):
        st.code("\n".join(result.logs))
    st.stop()

st.success("Pipeline complete.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Task",         result.dataset_info.task_type.capitalize())
c2.metric("Model",        result.ml_config.architecture if result.ml_config else "N/A")
c3.metric("Critic Score", f"{result.critic_decision.score:.1f} / 10" if result.critic_decision else "N/A")
c4.metric("Retries",      result.retry_count)

# ── post-run: feature importance in training tab ──────────────────────────────
if result.training_result:
    is_sklearn = len(result.training_result.train_losses) <= 1
    if is_sklearn:
        with tab_train:
            st.divider()
            st.subheader("Feature Importance")
            try:
                import joblib as jl
                model      = jl.load(result.training_result.model_path)
                saved      = jl.load(result.training_result.preprocessor_path)
                preproc    = saved["preprocessor"]
                feat_names = preproc.get_feature_names_out().tolist()
                fig_imp    = feature_importance_figure(model, feat_names)
                if fig_imp:
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.caption(
                        f"{result.ml_config.architecture} does not "
                        "support feature importance."
                    )
            except Exception as e:
                st.caption(f"Feature importance unavailable: {e}")

# ── post-run: report download in report tab ───────────────────────────────────
if result.final_report and result.final_report.full_markdown:
    with tab_report:
        st.divider()
        st.download_button(
            label="Download Report as Markdown",
            data=result.final_report.full_markdown,
            file_name=f"{os.path.splitext(uploaded.name)[0]}_report.md",
            mime="text/markdown",
        )

# ── full logs ─────────────────────────────────────────────────────────────────
with st.expander("Full pipeline logs"):
    st.code("\n".join(result.logs))
