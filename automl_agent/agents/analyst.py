import json

import pandas as pd
import numpy as np
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from automl_agent.core.llm import get_llm
from automl_agent.core.state import (
    AgentState,
    AnalystInsight,
    DatasetInfo,
    ColumnProfile,
    DatasetProfile,
)
from automl_agent.utils.preprocessing import read_table, _try_extract_numeric


def _profile_column(series: pd.Series, col: str) -> ColumnProfile:
    """
    Compute detailed statistics for a single column.
    """
    n_total   = len(series)
    n_missing = int(series.isna().sum())
    n_unique  = int(series.nunique())

    # detect if string column is actually numeric
    dtype_raw    = str(series.dtype)
    is_num       = is_numeric_dtype(series)
    is_converted = False

    if not is_num and series.dtype == object:
        converted = _try_extract_numeric(series)
        if converted is not None:
            series       = converted
            is_num       = True
            is_converted = True

    if is_num:
        col_type = "numeric"
        clean = pd.to_numeric(series, errors="coerce").dropna()
        outlier_count = 0
        outlier_pct = 0.0
        outlier_lower_bound = None
        outlier_upper_bound = None
        if len(clean) > 0:
            q1 = float(clean.quantile(0.25))
            q3 = float(clean.quantile(0.75))
            iqr = q3 - q1
            if iqr > 0:
                outlier_lower_bound = round(q1 - 1.5 * iqr, 4)
                outlier_upper_bound = round(q3 + 1.5 * iqr, 4)
                mask = (clean < outlier_lower_bound) | (clean > outlier_upper_bound)
                outlier_count = int(mask.sum())
                outlier_pct = round(outlier_count / n_total * 100, 2) if n_total > 0 else 0.0
        stats = {
            "mean":   round(float(series.mean()),   4) if series.notna().any() else None,
            "std":    round(float(series.std()),    4) if series.notna().any() else None,
            "min":    round(float(series.min()),    4) if series.notna().any() else None,
            "25%":    round(float(series.quantile(0.25)), 4) if series.notna().any() else None,
            "median": round(float(series.median()), 4) if series.notna().any() else None,
            "75%":    round(float(series.quantile(0.75)), 4) if series.notna().any() else None,
            "max":    round(float(series.max()),    4) if series.notna().any() else None,
        }
        top_values = {}
    else:
        col_type   = "categorical"
        stats      = {}
        vc = series.value_counts()
        top_values = {str(k): int(v) for k, v in vc.head(5).items()} if len(vc) > 0 else {}
        outlier_count = 0
        outlier_pct = 0.0
        outlier_lower_bound = None
        outlier_upper_bound = None

    return ColumnProfile(
        name=col,
        dtype=dtype_raw,
        col_type=col_type,
        is_converted=is_converted,
        n_missing=n_missing,
        missing_pct=round(n_missing / n_total * 100, 2) if n_total > 0 else 0.0,
        n_unique=n_unique,
        unique_pct=round(n_unique / n_total * 100, 2) if n_total > 0 else 0.0,
        stats=stats,
        top_values=top_values,
        outlier_count=outlier_count,
        outlier_pct=outlier_pct,
        outlier_lower_bound=outlier_lower_bound,
        outlier_upper_bound=outlier_upper_bound,
    )


def _build_dataset_profile(df: pd.DataFrame, target_col: str) -> DatasetProfile:
    """
    Full dataset-level profile including column analysis,
    null summary, and duplicate detection.
    """
    n_rows, n_cols = df.shape
    n_duplicates   = int(df.duplicated().sum())
    total_cells    = n_rows * n_cols
    total_missing  = int(df.isna().sum().sum())

    # null summary per column
    null_summary = {}
    for col in df.columns:
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            null_summary[col] = {
                "count": n_null,
                "pct":   round(n_null / n_rows * 100, 2),
            }

    # column profiles (excluding target)
    column_profiles = {}
    for col in df.columns:
        if col == target_col:
            continue
        column_profiles[col] = _profile_column(df[col].copy(), col)

    # target profile
    target_profile = _profile_column(df[target_col].copy(), target_col)
    outlier_summary = {
        col: {
            "count": profile.outlier_count,
            "pct": profile.outlier_pct,
            "lower_bound": profile.outlier_lower_bound,
            "upper_bound": profile.outlier_upper_bound,
        }
        for col, profile in column_profiles.items()
        if profile.outlier_count > 0
    }

    return DatasetProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        n_duplicates=n_duplicates,
        duplicate_pct=round(n_duplicates / n_rows * 100, 2) if n_rows > 0 else 0.0,
        total_missing=total_missing,
        total_missing_pct=round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0.0,
        null_summary=null_summary,
        outlier_summary=outlier_summary,
        column_profiles=column_profiles,
        target_profile=target_profile,
    )


def analyst_agent(state: AgentState) -> AgentState:
    state.current_step = "analysis"
    state.add_log("Analyst: reading and profiling dataset.")

    try:
        info = state.dataset_info
        df   = read_table(info.path)

        if info.target_column not in df.columns:
            raise ValueError(
                f"Target column '{info.target_column}' not found."
            )

        df = df.dropna(subset=[info.target_column])

        # drop obvious index columns
        index_cols = [
            c for c in df.columns
            if c.lower() in ("unnamed: 0", "id", "index", "row_id", "rowid")
        ]
        if index_cols:
            df = df.drop(columns=index_cols)

        target    = df[info.target_column]
        n_unique  = target.nunique()

        # task type detection
        task_type = "classification"
        if (
            is_numeric_dtype(target)
            and not is_bool_dtype(target)
            and n_unique > min(20, len(target) * 0.05)
        ):
            task_type = "regression"

        # feature types
        feature_types = {}
        for col in df.columns:
            if col == info.target_column:
                continue
            if is_numeric_dtype(df[col]):
                feature_types[col] = "numeric"
            elif _try_extract_numeric(df[col]) is not None:
                feature_types[col] = "numeric_string"
            else:
                feature_types[col] = "categorical"

        # class balance
        class_balance = {}
        is_imbalanced = False
        n_classes     = None

        if task_type == "classification":
            counts        = target.value_counts(normalize=True)
            class_balance = {str(k): round(float(v), 4) for k, v in counts.items()}
            n_classes     = int(n_unique)
            if len(counts) > 1:
                is_imbalanced = bool(counts.min() / counts.max() < 0.25)

        # build detailed profile 
        dataset_profile = _build_dataset_profile(df, info.target_column)

        state.dataset_info = DatasetInfo(
            path=info.path,
            name=info.name,
            target_column=info.target_column,
            task_type=task_type,
            n_samples=int(df.shape[0]),
            n_features=int(df.shape[1] - 1),
            n_classes=n_classes,
            class_balance=class_balance,
            is_imbalanced=is_imbalanced,
            feature_types=feature_types,
            missing_ratio=round(float(df.isna().mean().mean()), 4),
        )
        state.dataset_profile = dataset_profile

        state.add_log(
            f"Analyst: task={task_type}, rows={df.shape[0]}, "
            f"features={df.shape[1]-1}, "
            f"missing={state.dataset_info.missing_ratio:.2%}, "
            f"duplicates={dataset_profile.n_duplicates}."
        )

    except Exception as exc:
        state.error = f"Analyst failed: {exc}"
        state.add_log(state.error)
        state.analyst_insight = AnalystInsight(
            task_type="unknown",
            interpretation=f"Analysis failed: {exc}",
        )
        return state

    # LLM interprets the statistics 
    llm = get_llm()

    # build column summary for LLM  top 15 columns to avoid token overflow
    col_summary = []
    for col, profile in list(state.dataset_profile.column_profiles.items())[:15]:
        entry = {
            "column":      col,
            "type":        profile.col_type,
            "missing_pct": profile.missing_pct,
            "n_unique":    profile.n_unique,
        }
        if profile.col_type == "numeric" and profile.stats:
            entry["range"] = f"{profile.stats.get('min')} – {profile.stats.get('max')}"
            entry["mean"]  = profile.stats.get("mean")
            entry["outliers"] = {
                "count": profile.outlier_count,
                "pct": profile.outlier_pct,
                "lower_bound": profile.outlier_lower_bound,
                "upper_bound": profile.outlier_upper_bound,
            }
        else:
            entry["top_values"] = list(profile.top_values.keys())[:3]
        if profile.is_converted:
            entry["note"] = "originally string, extracted as numeric"
        col_summary.append(entry)

    if llm:
        prompt = f"""
You are an expert ML data analyst reviewing a dataset before training.

Dataset overview:
- Name: {state.dataset_info.name}
- Rows: {state.dataset_info.n_samples}
- Features: {state.dataset_info.n_features}
- Task: {state.dataset_info.task_type}
- Target column: {state.dataset_info.target_column}
- Class balance: {state.dataset_info.class_balance}
- Is imbalanced: {state.dataset_info.is_imbalanced}
- Overall missing ratio: {state.dataset_info.missing_ratio:.2%}
- Duplicate rows: {state.dataset_profile.n_duplicates} ({state.dataset_profile.duplicate_pct}%)
- Columns with nulls: {list(state.dataset_profile.null_summary.keys())}
- Columns with outliers: {state.dataset_profile.outlier_summary}

Column-level summary (top {len(col_summary)} features):
{json.dumps(col_summary, indent=2)}

Target profile:
- type: {state.dataset_profile.target_profile.col_type}
- missing: {state.dataset_profile.target_profile.missing_pct}%
- unique values: {state.dataset_profile.target_profile.n_unique}
- stats: {state.dataset_profile.target_profile.stats}

Planner risk flags: {state.planner_decision.risk_flags if state.planner_decision else []}

Think about:
- Which columns have quality issues (high missing, suspicious values)?
- Are there columns that were originally strings but contain numbers?
  (e.g. "8GB", "1.37kg") — these need special preprocessing.
- Does the class balance or target distribution suggest challenges?
- Are duplicate rows significant enough to affect training?
- Are numeric outliers significant enough to affect training or model choice?
- What does this dataset represent and what are the domain-specific risks?

Respond ONLY in this exact JSON format, no markdown, no extra text:
{{
    "task_type": "{state.dataset_info.task_type}",
    "interpretation": "3-4 sentence summary covering dataset quality, domain context, and key challenges",
    "risks": [
        "specific risk referencing actual column names and numbers"
    ],
    "recommendations": [
        "specific actionable recommendation referencing actual columns"
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
            state.analyst_insight = AnalystInsight(**parsed)
            state.add_log(
                f"Analyst LLM: {state.analyst_insight.interpretation}"
            )
        except Exception as exc:
            state.add_log(
                f"Analyst: LLM failed ({exc}), using rule-based insight."
            )
            state.analyst_insight = _rule_based_insight(state)
    else:
        state.add_log("Analyst: no API key, using rule-based insight.")
        state.analyst_insight = _rule_based_insight(state)


    # build preprocessing plan 
    try:
        state.preprocessing_plan = _build_preprocessing_plan(
            df, info.target_column, state.dataset_profile
        )
        state.add_log(
            f"Analyst: preprocessing plan built for "
            f"{len(state.preprocessing_plan)} columns."
        )
    except Exception as e:
        state.add_log(f"Analyst: preprocessing plan skipped ({e}).")
        state.preprocessing_plan = []

    return state

    
    return state


def _rule_based_insight(state: AgentState) -> AnalystInsight:
    info    = state.dataset_info
    profile = state.dataset_profile
    risks   = []
    recs    = []

    if info.is_imbalanced:
        risks.append("Class imbalance — minority class may be underlearned.")
        recs.append("Use class_weight='balanced' in model.")

    if info.missing_ratio > 0.05:
        high_null = [
            f"{col} ({v['pct']}%)"
            for col, v in profile.null_summary.items()
            if v["pct"] > 10
        ]
        if high_null:
            risks.append(f"High missing values in: {', '.join(high_null)}.")
            recs.append("Consider dropping or imputing high-null columns.")

    if profile.n_duplicates > 0:
        risks.append(
            f"{profile.n_duplicates} duplicate rows ({profile.duplicate_pct}%) detected."
        )
        recs.append("Duplicates will be removed before training.")

    if profile.outlier_summary:
        worst = sorted(
            profile.outlier_summary.items(),
            key=lambda item: item[1]["pct"],
            reverse=True,
        )[:5]
        risks.append(
            "Numeric outliers detected in: "
            + ", ".join(f"{col} ({meta['pct']}%)" for col, meta in worst)
            + "."
        )
        recs.append(
            "Review high-outlier columns; consider clipping, transformations, "
            "or robust tree-based models when values are valid extremes."
        )

    converted = [
        col for col, p in profile.column_profiles.items()
        if p.is_converted
    ]
    if converted:
        risks.append(
            f"Columns {converted} contain numeric values stored as strings "
            "(e.g. '8GB', '1.37kg') — will be extracted automatically."
        )

    interpretation = (
        f"Dataset has {info.n_samples} rows, {info.n_features} features, "
        f"task={info.task_type}. "
        f"Missing={info.missing_ratio:.1%}, "
        f"duplicates={profile.n_duplicates}."
    )

    return AnalystInsight(
        task_type=info.task_type,
        interpretation=interpretation,
        risks=risks,
        recommendations=recs,
    )

def _build_preprocessing_plan(
    df: pd.DataFrame,
    target_col: str,
    dataset_profile,
) -> list[dict]:
    """
    Show exactly what preprocessing will be applied to each column.
    Matches the logic in build_splits exactly.
    """
    from automl_agent.utils.preprocessing import _try_extract_numeric, _is_text_column

    plan = []
    for col in df.columns:
        if col == target_col:
            continue

        series   = df[col]
        n_unique = series.nunique()

        if pd.api.types.is_numeric_dtype(series):
            plan.append({
                "column":     col,
                "raw_type":   str(series.dtype),
                "treatment":  "Numeric",
                "steps":      "Median impute → StandardScaler",
                "reason":     "Already numeric",
            })
            continue

        converted = _try_extract_numeric(series)
        if converted is not None:
            plan.append({
                "column":     col,
                "raw_type":   "string",
                "treatment":  "Numeric (extracted)",
                "steps":      f"Extract number from string → Median impute → StandardScaler",
                "reason":     f"e.g. '{series.dropna().iloc[0]}' → {converted.dropna().iloc[0]:.2f}",
            })
            continue

        if _is_text_column(series):
            plan.append({
                "column":     col,
                "raw_type":   "free text",
                "treatment":  "TF-IDF",
                "steps":      "TfidfVectorizer (max 300 features, unigrams+bigrams)",
                "reason":     "High avg word count — free text column",
            })
            continue

        if n_unique <= 15:
            plan.append({
                "column":     col,
                "raw_type":   "categorical",
                "treatment":  "One-Hot Encoded",
                "steps":      f"Mode impute → OneHotEncoder ({n_unique} categories)",
                "reason":     f"Low cardinality: {n_unique} unique values",
            })
        else:
            plan.append({
                "column":     col,
                "raw_type":   "categorical",
                "treatment":  "Frequency Encoded",
                "steps":      "Replace category with its frequency in training set → StandardScaler",
                "reason":     f"High cardinality: {n_unique} unique values (>15)",
            })

    return plan
