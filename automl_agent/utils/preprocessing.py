import joblib
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from scipy.sparse import issparse

from automl_agent.core.state import DatasetInfo


# file reading 

def read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if path.lower().endswith(".csv"):
        encodings = ["utf-8", "latin-1", "cp1252", "utf-8-sig", "iso-8859-1"]
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except (UnicodeDecodeError, Exception):
                continue
        return pd.read_csv(path, encoding="utf-8", errors="ignore")
    raise ValueError("Only CSV and Excel files are supported.")


def save_preprocessor(path: str, preprocessor, label_encoder) -> None:
    joblib.dump(
        {"preprocessor": preprocessor, "label_encoder": label_encoder},
        path,
    )


# helper functions (used by analyst too)

def _try_extract_numeric(series: pd.Series) -> pd.Series | None:
    """
    Attempt to convert a string column to numeric.
    Handles: units (8GB→8, 1.37kg→1.37), ranges (1000-1500→1250),
    leading numbers (2 BHK→2).
    Returns numeric Series if >70% converts successfully, else None.
    """
    s = series.dropna().astype(str).str.strip()

    # ranges: "1000-1500" or "100 - 200"
    range_mask = s.str.contains(
        r"^\d+\.?\d*\s*[-–]\s*\d+\.?\d*$", regex=True
    )
    if range_mask.sum() > len(s) * 0.1:
        def parse_range(val):
            parts = re.split(r"[-–]", str(val))
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except Exception:
                return np.nan
        converted = series.apply(
            lambda x: parse_range(x) if pd.notna(x) else np.nan
        )
        if converted.notna().sum() / max(len(series), 1) > 0.7:
            return converted

    # strip units: 8GB→8, 1.37kg -> 1.37
    unit_pattern = r"^(\d+\.?\d*)\s*(GB|MB|TB|kg|g|cm|mm|GHz|MHz|W|m|inch|\")?$"
    extracted = s.str.extract(unit_pattern, expand=False)[0]
    converted = pd.to_numeric(extracted, errors="coerce")
    if converted.notna().sum() / max(len(s), 1) > 0.7:
        return pd.to_numeric(
            series.astype(str).str.strip()
                  .str.extract(unit_pattern, expand=False)[0],
            errors="coerce",
        )

    # leading number: "2 BHK" -> 2
    leading   = s.str.extract(r"^(\d+\.?\d*)", expand=False)
    converted = pd.to_numeric(leading, errors="coerce")
    if converted.notna().sum() / max(len(s), 1) > 0.7:
        return pd.to_numeric(
            series.astype(str).str.strip()
                  .str.extract(r"^(\d+\.?\d*)", expand=False),
            errors="coerce",
        )

    return None


def _is_text_column(series: pd.Series, min_avg_words: float = 4.0) -> bool:
    non_null  = series.dropna().astype(str)
    if len(non_null) == 0:
        return False
    return non_null.str.split().str.len().mean() >= min_avg_words


# Option C: custom sklearn transformer 

class NumericStringExtractor(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer that extracts numbers from string columns.

    Fitted during training: determines the extraction pattern per column.
    Applied during transform: converts strings to floats the same way.

    This makes the preprocessor self-contained — any caller that runs
    preprocessor.transform() gets correct numeric output automatically,
    regardless of whether the input contains '8GB' or 8.0.

    Fits into a sklearn Pipeline like any other transformer:
        Pipeline([
            ("extractor", NumericStringExtractor()),
            ("imputer",   SimpleImputer(strategy="median")),
            ("scaler",    StandardScaler()),
        ])
    """

    def __init__(self):
        self.patterns_  = {}   # col_index -> extraction pattern type
        self.range_     = {}   # col_index -> True if range pattern
        self.unit_pat_  = r"^(\d+\.?\d*)\s*(GB|MB|TB|kg|g|cm|mm|GHz|MHz|W|m|inch|\")?$"
        self.lead_pat_  = r"^(\d+\.?\d*)"

    def fit(self, X, y=None):
        """
        Learn which extraction pattern works for each column.
        X can be a DataFrame or numpy array.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.patterns_ = {}
        self.range_    = {}

        for i, col in enumerate(X.columns if hasattr(X, "columns")
                                else range(X.shape[1])):
            series = X.iloc[:, i] if isinstance(X, pd.DataFrame) else X[:, i]
            series = pd.Series(series)

            # already numeric — no extraction needed
            if pd.api.types.is_numeric_dtype(series):
                self.patterns_[i] = "numeric"
                continue

            s = series.dropna().astype(str).str.strip()

            # check range pattern
            range_mask = s.str.contains(
                r"^\d+\.?\d*\s*[-–]\s*\d+\.?\d*$", regex=True
            )
            if range_mask.sum() > len(s) * 0.1:
                self.patterns_[i] = "range"
                continue

            # check unit pattern
            extracted = s.str.extract(self.unit_pat_, expand=False)[0]
            converted = pd.to_numeric(extracted, errors="coerce")
            if converted.notna().sum() / max(len(s), 1) > 0.7:
                self.patterns_[i] = "unit"
                continue

            # check leading number
            leading   = s.str.extract(r"^(\d+\.?\d*)", expand=False)
            converted = pd.to_numeric(leading, errors="coerce")
            if converted.notna().sum() / max(len(s), 1) > 0.7:
                self.patterns_[i] = "leading"
                continue

            self.patterns_[i] = "numeric"  # pass-through

        return self

    def transform(self, X, y=None):
        """Apply the fitted extraction patterns to new data."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X.copy())
        else:
            X = X.copy()

        result = np.zeros((len(X), X.shape[1]), dtype=np.float32)

        for i in range(X.shape[1]):
            series  = pd.Series(X.iloc[:, i] if isinstance(X, pd.DataFrame)
                                else X[:, i])
            pattern = self.patterns_.get(i, "numeric")

            if pattern == "numeric" or pd.api.types.is_numeric_dtype(series):
                result[:, i] = pd.to_numeric(
                    series, errors="coerce"
                ).astype(np.float32).values
                continue

            s = series.astype(str).str.strip()

            if pattern == "range":
                def parse_range(val):
                    parts = re.split(r"[-–]", str(val))
                    try:
                        return (float(parts[0]) + float(parts[1])) / 2
                    except Exception:
                        return np.nan
                vals = series.apply(
                    lambda x: parse_range(x) if pd.notna(x) else np.nan
                )

            elif pattern == "unit":
                vals = pd.to_numeric(
                    s.str.extract(self.unit_pat_, expand=False)[0],
                    errors="coerce",
                )

            elif pattern == "leading":
                vals = pd.to_numeric(
                    s.str.extract(r"^(\d+\.?\d*)", expand=False),
                    errors="coerce",
                )
            else:
                vals = pd.to_numeric(series, errors="coerce")

            result[:, i] = vals.astype(np.float32).values

        return result

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        return np.array([f"x{i}" for i in range(len(self.patterns_))])


# main preprocessing function 

def _frequency_encode(
    train: pd.Series,
    val: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    freq_map  = train.value_counts(normalize=True).to_dict()
    train_enc = train.map(freq_map).fillna(0.0).astype(np.float32)
    val_enc   = val.map(freq_map).fillna(0.0).astype(np.float32)
    return train_enc, val_enc


def build_splits(info: DatasetInfo):
    """
    Column handling strategy:
    ┌──────────────────────────┬──────────────────────────────────────────┐
    │ Column type              │ Treatment                                │
    ├──────────────────────────┼──────────────────────────────────────────┤
    │ Numeric                  │ NumericStringExtractor →                 │
    │ Numeric-looking string   │   median impute → StandardScaler         │
    │  (8GB, 1.37kg, 1000-1500)│ (extractor is a no-op for true numerics) │
    │ Low-card cat (≤15 unique)│ mode impute → OneHotEncoder              │
    │ High-card cat (>15)      │ frequency encoding → numeric pipeline    │
    │ Free text (primary)      │ TF-IDF (max 300 features)                │
    │ Free text (secondary)    │ dropped                                  │
    │ Index/ID columns         │ dropped                                  │
    └──────────────────────────┴──────────────────────────────────────────┘

    Key change (Option C):
    NumericStringExtractor is the first step in the numeric pipeline.
    This means preprocessor.transform() on raw data (with "8GB" strings)
    works correctly without any manual pre-processing by the caller.
    """
    df = read_table(info.path).dropna(subset=[info.target_column])

    # drop obvious index columns
    index_cols = [
        c for c in df.columns
        if c.lower() in ("unnamed: 0", "id", "index", "row_id", "rowid")
        or (df[c].nunique() == len(df) and pd.api.types.is_integer_dtype(df[c]))
    ]
    if index_cols:
        df = df.drop(columns=index_cols)

    # drop duplicates
    df = df.drop_duplicates()

    X     = df.drop(columns=[info.target_column]).copy()
    y_raw = df[info.target_column]

    # encode target 
    label_encoder = None
    if info.task_type == "classification":
        label_encoder = LabelEncoder()
        y        = label_encoder.fit_transform(y_raw).astype(np.int64)
        min_cls  = int(np.bincount(y).min())
        stratify = y if min_cls >= 2 else None
    else:
        if y_raw.dtype == object:
            converted = _try_extract_numeric(y_raw)
            if converted is not None:
                y_raw = converted
            else:
                raise ValueError(
                    f"Target '{info.target_column}' could not be converted "
                    "to numeric for regression."
                )
        y        = y_raw.astype(np.float32).to_numpy()
        stratify = None

    # train / val split 
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    #  classify columns using TRAIN data only
    num_cols       = []   # numeric + numeric-string -> extractor pipeline
    low_card_cols  = []   # categorical ≤15 -> one-hot
    high_card_cols = []   # categorical >15 -> frequency encode
    text_cols      = []   # free text -> tfidf

    for col in X_train.columns:
        series = X_train[col]

        if pd.api.types.is_numeric_dtype(series):
            num_cols.append(col)
            continue

        # numeric-looking strings go into the numeric pipeline
        # NumericStringExtractor handles the conversion inside the pipeline
        if _try_extract_numeric(series) is not None:
            num_cols.append(col)
            continue

        if _is_text_column(series):
            text_cols.append(col)
            continue

        n_unique = series.nunique()
        if n_unique <= 15:
            low_card_cols.append(col)
        else:
            high_card_cols.append(col)

    # frequency-encode high-cardinality columns 
    # these are done outside the ColumnTransformer because frequency maps
    # must be fitted on train only and reused consistently
    for col in high_card_cols:
        X_train[col], X_val[col] = _frequency_encode(
            X_train[col], X_val[col]
        )
        num_cols.append(col)   # after encoding, treat as numeric

    # text columns 
    primary_text_col = text_cols[0] if text_cols else None
    secondary_text   = text_cols[1:] if len(text_cols) > 1 else []

    if secondary_text:
        X_train = X_train.drop(columns=secondary_text)
        X_val   = X_val.drop(columns=secondary_text)

    if not (num_cols or low_card_cols) and not primary_text_col:
        raise ValueError("No usable feature columns found after preprocessing.")

    # build ColumnTransformer 
    transformers = []

    if num_cols:
        # NumericStringExtractor is first — converts "8GB"->8 before imputer
        num_pipeline = Pipeline([
            ("extractor", NumericStringExtractor()),
            ("imputer",   SimpleImputer(strategy="median")),
            ("scaler",    StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if low_card_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )),
        ])
        transformers.append(("cat", cat_pipeline, low_card_cols))

    if primary_text_col:
        X_train[primary_text_col] = (
            X_train[primary_text_col].fillna("").astype(str)
        )
        X_val[primary_text_col] = (
            X_val[primary_text_col].fillna("").astype(str)
        )
        tfidf_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=300,
                strip_accents="unicode",
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
            )),
        ])
        transformers.append(("text", tfidf_pipeline, primary_text_col))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0,
    )

    X_train_out = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_out   = preprocessor.transform(X_val).astype(np.float32)

    if issparse(X_train_out):
        X_train_out = X_train_out.toarray().astype(np.float32)
    if issparse(X_val_out):
        X_val_out = X_val_out.toarray().astype(np.float32)

    return X_train_out, X_val_out, y_train, y_val, preprocessor, label_encoder