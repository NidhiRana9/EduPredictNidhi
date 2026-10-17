"""Data Preprocessing and Feature Preparation Module for EduPredict.

Phase 3 Implementation:
- Defines clean separation between features and target variables.
- Sets up targets for both planned ML tasks:
    1. Binary Classification: 'at_risk' (1 if G3 < 10, else 0).
    2. Regression: numerical 'G3' (final grade).
- Excludes interim grades ('G1', 'G2'), target ('G3', 'at_risk'), and database
  metadata ('_id', 'source_row_id', 'dataset_source') to prevent data leakage.
- Identifies and partitions numerical and categorical feature sets.
- Constructs scikit-learn ColumnTransformer pipelines (un-fitted).
- Provides train/test splitting utilities with reproducible random seeds and
  stratification for classification.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==============================================================================
# Feature & Target Column Definitions
# ==============================================================================
TARGET_COLUMN_REGRESSION: str = "G3"
TARGET_COLUMN_CLASSIFICATION: str = "at_risk"
AT_RISK_THRESHOLD: int = 10  # Passing threshold is 10/20 in standard Portuguese academic scale

# Columns strictly excluded from feature sets to avoid target leakage & metadata bias
EXCLUDED_FROM_FEATURES: List[str] = [
    "_id",
    "source_row_id",
    "dataset_source",
    "G1",
    "G2",
    "G3",
    "at_risk",
]

# Numeric feature definitions (13 features)
NUMERIC_FEATURES: List[str] = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
]

# Categorical feature definitions (17 features)
CATEGORICAL_FEATURES: List[str] = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]


def identify_feature_types(
    df: pd.DataFrame,
    excluded_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical feature columns present in the DataFrame.

    Excludes metadata identifiers and grade columns to prevent target leakage.

    Args:
        df: Input student DataFrame.
        excluded_cols: Optional custom list of column names to exclude.

    Returns:
        Tuple of (numeric_column_names, categorical_column_names).
    """
    excluded = set(excluded_cols if excluded_cols is not None else EXCLUDED_FROM_FEATURES)
    candidate_cols = [c for c in df.columns if c not in excluded]

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def prepare_features_and_targets(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate input features from classification and regression targets.

    Creates the binary classification target:
        at_risk = 1 if G3 < 10 else 0
    And extracts the numeric regression target:
        G3 (0-20 scale)

    Excludes:
      - 'G1', 'G2', 'G3' (interim and final grades)
      - 'at_risk' (target variable)
      - '_id', 'source_row_id', 'dataset_source' (database identifiers)

    Args:
        df: Complete raw or cleaned student DataFrame.

    Returns:
        Tuple of:
          - X: Feature DataFrame (30 predictor variables)
          - y_classification: Binary target Series (1: at-risk, 0: not at-risk)
          - y_regression: Numeric target Series (G3 final grade)

    Raises:
        KeyError: If G3 target column is missing from the DataFrame.
    """
    if TARGET_COLUMN_REGRESSION not in df.columns:
        raise KeyError(
            f"Required target column '{TARGET_COLUMN_REGRESSION}' is missing from the dataset."
        )

    # 1. Create the classification target
    y_classification = (df[TARGET_COLUMN_REGRESSION] < AT_RISK_THRESHOLD).astype(int)
    y_classification.name = TARGET_COLUMN_CLASSIFICATION

    # 2. Extract the regression target
    y_regression = df[TARGET_COLUMN_REGRESSION].astype(int)
    y_regression.name = TARGET_COLUMN_REGRESSION

    # 3. Filter feature columns
    excluded_set = set(EXCLUDED_FROM_FEATURES)
    feature_cols = [c for c in df.columns if c not in excluded_set]
    X = df[feature_cols].copy()

    return X, y_classification, y_regression


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build an un-fitted scikit-learn ColumnTransformer preprocessing pipeline.

    Applies:
      - Numeric features: Median imputation (preserves integer medians and handles missing values),
        with optional StandardScaler.
      - Categorical features: Most-frequent imputation and OneHotEncoder(handle_unknown='ignore').

    IMPORTANT: This function returns an UN-FITTED pipeline. Preprocessing must
    only be fitted on the training split in later ML phases to prevent data leakage.

    Args:
        numeric_features: List of numeric column names (defaults to NUMERIC_FEATURES).
        categorical_features: List of categorical column names (defaults to CATEGORICAL_FEATURES).
        scale_numeric: Whether to append StandardScaler to numeric features.

    Returns:
        An un-fitted sklearn.compose.ColumnTransformer instance.
    """
    num_cols = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat_cols = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES

    # Numeric pipeline
    num_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=num_steps)

    # Categorical pipeline
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split feature and target datasets into train and test partitions.

    Uses a fixed random seed for reproducibility. When stratify is True,
    ensures the target class distribution is preserved across splits.

    Args:
        X: Feature DataFrame.
        y: Target Series (classification or regression).
        test_size: Fraction of samples to allocate to test partition (default: 0.2).
        random_state: Random seed for deterministic splitting (default: 42).
        stratify: Whether to perform stratified sampling on y.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    stratify_target = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    return X_train, X_test, y_train, y_test
