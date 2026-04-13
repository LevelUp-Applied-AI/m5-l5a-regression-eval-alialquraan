"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score,ConfusionMatrixDisplay)

import matplotlib.pyplot as plt


def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset.

    Returns:
        DataFrame with all columns.
    """
    # TODO: Load the CSV and return the DataFrame
    df = pd.read_csv(filepath)

    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())

    if "churned" in df.columns:
        print("\nChurn distribution:\n", df["churned"].value_counts(normalize=True))

    return df



def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # TODO: Separate features and target, then split with stratification
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.nunique() <= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

    print("\nTrain size:", X_train.shape)
    print("Test size:", X_test.shape)

    if y.nunique() <= 10:
        print("Train churn rate:", y_train.mean())
        print("Test churn rate:", y_test.mean())

    return X_train, X_test, y_train, y_test


def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression.

    Returns:
        sklearn Pipeline object.
    """
    # TODO: Create and return a Pipeline with two steps
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])
    return pipe


def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression.

    Returns:
        sklearn Pipeline object.
    """
    # TODO: Create and return a Pipeline for Ridge regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    return pipe


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics.

    Args:
        pipeline: sklearn Pipeline with a classifier.
        X_train, X_test: Feature arrays.
        y_train, y_test: Label arrays.

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    # TODO: Fit the pipeline on training data, predict on test, compute metrics
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"]
    }



def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics.

    Args:
        pipeline: sklearn Pipeline with a regressor.
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.

    Returns:
        Dictionary with keys: 'mae', 'r2'.
    """
    # TODO: Fit the pipeline, predict, and compute MAE and R²
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nMAE:", mae)
    print("R2:", r2)

    return {
        "mae": mae,
        "r2": r2
    }



def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of folds.

    Returns:
        Array of cross-validation scores.
    """
    # TODO: Run cross_val_score with StratifiedKFold
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv_splitter,
        scoring="accuracy"
    )

    print("\nCV scores:", scores)
    print("Mean:", scores.mean())
    print("Std:", scores.std())

    return scores



if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

        # Select numeric features for classification
        numeric_features = ["tenure", "monthly_charges", "total_charges",
                           "num_support_calls", "senior_citizen",
                           "has_partner", "has_dependents"]

        # Classification: predict churn
        df_cls = df[numeric_features + ["churned"]].dropna()
        split = split_data(df_cls, "churned")
        if split:
            X_train, X_test, y_train, y_test = split
            pipe = build_logistic_pipeline()
            if pipe:
                metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
                print(f"Logistic Regression: {metrics}")

                scores = run_cross_validation(pipe, X_train, y_train)
                if scores is not None:
                    print(f"CV: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Regression: predict monthly_charges
        df_reg = df[["tenure", "total_charges", "num_support_calls",
                     "senior_citizen", "has_partner", "has_dependents",
                     "monthly_charges"]].dropna()
        split_reg = split_data(df_reg, "monthly_charges")
        if split_reg:
            X_tr, X_te, y_tr, y_te = split_reg
            ridge_pipe = build_ridge_pipeline()
            if ridge_pipe:
                reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
                print(f"Ridge Regression: {reg_metrics}")



# =========================
# TASK 7: SUMMARY OF FINDINGS
# =========================

# 1. Most important features for predicting churn:
# Based on the dataset and model behavior, the most influential features are likely:
# - tenure (how long the customer stayed)
# - contract_type (contract stability)
# - monthly_charges (cost sensitivity)
# - num_support_calls (customer dissatisfaction indicator)
#
# These features are typically strong predictors because churn is often driven by
# customer dissatisfaction, contract flexibility, and cost.

# 2. Logistic Regression performance:
# - Accuracy ≈ 0.63 (moderate performance)
# - Precision for class 1 (churn) is low (~0.23), meaning many predicted churns are incorrect
# - Recall for class 1 is better (~0.51), meaning the model catches about half of actual churners
#
# IMPORTANT:
# Recall is more critical in this problem than precision because:
# - The goal is to identify customers who are likely to leave
# - Missing a churned customer (false negative) is worse than a false alarm
#
# Therefore, improving recall should be a priority.

# 3. Recommendations for next steps:
# - Try more powerful models (Random Forest, Gradient Boosting, XGBoost)
# - Improve feature engineering (interaction features, customer behavior patterns)
# - Handle class imbalance more aggressively (SMOTE or tuning class weights)
# - Hyperparameter tuning for Logistic Regression (C, penalty)
# - Add more behavioral features if available (usage patterns, tenure trends)
# - Try threshold tuning instead of default 0.5 for better recall                
