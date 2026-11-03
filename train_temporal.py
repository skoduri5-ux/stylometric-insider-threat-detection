#!/usr/bin/env python3
"""
Train and evaluate a temporal insider-threat model using the generated,
employee-disjoint train/validation/test CSV files.

Run from the project directory:
    python train_temporal.py --data-dir data --artifact-dir artifacts

Predictors exclude IDs, username, department, role, label, scenario and split.
The supervised threshold is selected on validation data only.
Isolation Forest is fitted on normal training records only; its threshold is
selected from normal validation scores to target a 5% validation false-positive
rate. Test data is used only for final evaluation.
"""

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

SEED = 42

FEATURE_COLUMNS = [
    "hour",
    "day_of_week_num",
    "is_weekend",
    "is_after_hours",
    "recipient_count",
    "external_recipient",
    "attachment_count",
    "size",
    "word_count",
    "char_count",
    "sentence_count",
    "activity_frequency_1h",
    "activity_frequency_24h",
    "unique_recipients_24h",
    "off_hours_events_7d",
    "external_recipient_events_7d",
]

LABEL_MAP = {"normal": 0, "suspicious": 1}


def load_split(path):
    df = pd.read_csv(path)
    # day_of_week_num is derived below from timestamp, so it need not exist in the CSV.
    required = {"timestamp", "label", *(c for c in FEATURE_COLUMNS if c != "day_of_week_num")}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.copy()
    labels = df["label"].astype(str).str.strip().str.lower()
    unknown = sorted(set(labels) - set(LABEL_MAP))
    if unknown:
        raise ValueError(f"Unknown labels in {path}: {unknown}")

    # Derive weekday number from the timestamp to match the generator's
    # Monday=0 ... Sunday=6 convention.
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any():
        raise ValueError(f"Invalid timestamp(s) found in {path}")
    df["day_of_week_num"] = ts.dt.dayofweek

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing/non-numeric predictor values in {path}: {bad}")

    y = labels.map(LABEL_MAP).astype(int)
    if y.nunique() != 2:
        raise ValueError(f"{path} must contain both label classes.")
    return df, X, y


def confusion_dict(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    return {
        "true_normal_predicted_normal": int(tn),
        "true_normal_predicted_suspicious": int(fp),
        "true_suspicious_predicted_normal": int(fn),
        "true_suspicious_predicted_suspicious": int(tp),
    }


def metrics_at_threshold(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_dict(y_true, pred),
        "classification_report": classification_report(
            y_true,
            pred,
            labels=[0, 1],
            target_names=["normal", "suspicious"],
            output_dict=True,
            zero_division=0,
        ),
    }


def choose_f1_threshold(y_true, probabilities):
    """Choose threshold with maximum validation F1; never use test labels."""
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )
    if len(thresholds) == 0:
        return 0.5
    # precision/recall contain one more entry than thresholds.
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    best = int(np.nanargmax(f1))
    return float(thresholds[best])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()

    os.makedirs(args.artifact_dir, exist_ok=True)

    train_path = os.path.join(args.data_dir, "temporal_train.csv")
    val_path = os.path.join(args.data_dir, "temporal_validation.csv")
    test_path = os.path.join(args.data_dir, "temporal_test.csv")

    train_df, X_train, y_train = load_split(train_path)
    val_df, X_val, y_val = load_split(val_path)
    test_df, X_test, y_test = load_split(test_path)

    # Confirm the generator's employee-disjoint split property.
    employee_sets = {
        "train": set(train_df["username"]),
        "validation": set(val_df["username"]),
        "test": set(test_df["username"]),
    }
    if (
        employee_sets["train"] & employee_sets["validation"]
        or employee_sets["train"] & employee_sets["test"]
        or employee_sets["validation"] & employee_sets["test"]
    ):
        raise ValueError("Employee overlap detected between dataset splits.")

    print("=" * 68)
    print("TEMPORAL MODEL TRAINING")
    print("=" * 68)
    for name, frame, y in [
        ("Train", train_df, y_train),
        ("Validation", val_df, y_val),
        ("Test", test_df, y_test),
    ]:
        print(
            f"{name:10s}: {len(frame):5d} rows | "
            f"normal={(y == 0).sum():4d} | suspicious={(y == 1).sum():4d} | "
            f"employees={frame['username'].nunique()}"
        )
    print("Employee-disjoint split check: PASSED")

    # ------------------------------------------------------------
    # 1. Supervised model
    # ------------------------------------------------------------
    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    classifier.fit(X_train, y_train)

    val_prob = classifier.predict_proba(X_val)[:, 1]
    threshold = choose_f1_threshold(y_val, val_prob)
    val_metrics = metrics_at_threshold(y_val, val_prob, threshold)

    test_prob = classifier.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at_threshold(y_test, test_prob, threshold)
    test_metrics["roc_auc"] = float(roc_auc_score(y_test, test_prob))
    test_metrics["pr_auc"] = float(average_precision_score(y_test, test_prob))

    print("\n--- SUPERVISED RANDOM FOREST ---")
    print(f"Validation-selected threshold: {threshold:.4f}")
    print(f"Test accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test precision: {test_metrics['precision']:.4f}")
    print(f"Test recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-score:  {test_metrics['f1_score']:.4f}")
    print(f"Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print("Test confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, (test_prob >= threshold).astype(int)))
    print("\nTest classification report:")
    print(classification_report(
        y_test,
        (test_prob >= threshold).astype(int),
        labels=[0, 1],
        target_names=["normal", "suspicious"],
        zero_division=0,
    ))

    # ------------------------------------------------------------
    # 2. Unsupervised Isolation Forest
    # Fit only on normal training examples. Tune threshold using
    # normal validation examples, targeting approximately 5% FPR.
    # ------------------------------------------------------------
    normal_train = X_train.loc[y_train == 0]
    normal_val = X_val.loc[y_val == 0]

    anomaly_imputer = SimpleImputer(strategy="median")
    normal_train_imp = anomaly_imputer.fit_transform(normal_train)
    normal_val_imp = anomaly_imputer.transform(normal_val)
    X_test_imp = anomaly_imputer.transform(X_test)

    anomaly_model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=SEED,
        n_jobs=-1,
    )
    anomaly_model.fit(normal_train_imp)

    # Negate score_samples so larger scores mean more anomalous.
    val_anomaly_score = -anomaly_model.score_samples(normal_val_imp)
    anomaly_threshold = float(np.quantile(val_anomaly_score, 0.95))

    test_anomaly_score = -anomaly_model.score_samples(X_test_imp)
    anomaly_pred = (test_anomaly_score >= anomaly_threshold).astype(int)

    anomaly_cm = confusion_dict(y_test, anomaly_pred)
    anomaly_metrics = {
        "model": "IsolationForest",
        "fit_records": int(len(normal_train)),
        "threshold_rule": (
            "95th percentile of anomaly scores on normal validation records "
            "(targeting approximately 5% validation false-positive rate)"
        ),
        "anomaly_score_definition": "-IsolationForest.score_samples; higher is more anomalous",
        "threshold": anomaly_threshold,
        "test_records_flagged": int(anomaly_pred.sum()),
        "test_flagged_fraction": float(anomaly_pred.mean()),
        "precision": float(precision_score(y_test, anomaly_pred, zero_division=0)),
        "recall": float(recall_score(y_test, anomaly_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, anomaly_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_anomaly_score)),
        "pr_auc": float(average_precision_score(y_test, test_anomaly_score)),
        "confusion_matrix": anomaly_cm,
        "normal_test_mean_anomaly_score": float(
            test_anomaly_score[y_test.to_numpy() == 0].mean()
        ),
        "suspicious_test_mean_anomaly_score": float(
            test_anomaly_score[y_test.to_numpy() == 1].mean()
        ),
    }

    print("\n--- ISOLATION FOREST ---")
    print(f"Anomaly threshold: {anomaly_threshold:.6f}")
    print(f"Test flagged: {anomaly_pred.sum()} / {len(y_test)}")
    print(f"Precision: {anomaly_metrics['precision']:.4f}")
    print(f"Recall:    {anomaly_metrics['recall']:.4f}")
    print(f"F1-score:  {anomaly_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {anomaly_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {anomaly_metrics['pr_auc']:.4f}")
    print("Test confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, anomaly_pred))

    # ------------------------------------------------------------
    # Save model bundle, metrics and per-test-record predictions.
    # Keep names distinct from the non-temporal artifacts.
    # ------------------------------------------------------------
    model_path = os.path.join(args.artifact_dir, "temporal_model.joblib")
    metrics_path = os.path.join(args.artifact_dir, "temporal_metrics.json")
    predictions_path = os.path.join(
        args.artifact_dir, "temporal_test_predictions.csv"
    )

    bundle = {
        "model": classifier,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "anomaly_model": anomaly_model,
        "anomaly_imputer": anomaly_imputer,
        "anomaly_threshold": anomaly_threshold,
        "model_type": "temporal",
        "label_mapping": LABEL_MAP,
    }
    joblib.dump(bundle, model_path)

    metrics = {
        "dataset": {
            "train_records": len(train_df),
            "validation_records": len(val_df),
            "test_records": len(test_df),
            "train_employees": len(employee_sets["train"]),
            "validation_employees": len(employee_sets["validation"]),
            "test_employees": len(employee_sets["test"]),
            "employee_disjoint": True,
            "class_counts": {
                "train": train_df["label"].value_counts().to_dict(),
                "validation": val_df["label"].value_counts().to_dict(),
                "test": test_df["label"].value_counts().to_dict(),
            },
            "synthetic_data_warning": (
                "All records are synthetic and classes are artificially balanced. "
                "Metrics do not establish real-world insider-threat performance."
            ),
        },
        "feature_columns": FEATURE_COLUMNS,
        "supervised": {
            "model": "RandomForestClassifier",
            "threshold_selection": "maximum F1 on validation set",
            "validation": val_metrics,
            "test": test_metrics,
        },
        "anomaly_detection": anomaly_metrics,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    predictions = pd.DataFrame({
        "event_id": test_df["event_id"].to_numpy(),
        "username": test_df["username"].to_numpy(),
        "actual_label": test_df["label"].to_numpy(),
        "suspicious_probability": test_prob,
        "supervised_predicted_label": np.where(
            test_prob >= threshold, "suspicious", "normal"
        ),
        "anomaly_score": test_anomaly_score,
        "anomaly_flag": anomaly_pred.astype(bool),
        "threat_scenario": test_df["threat_scenario"].to_numpy(),
    })
    predictions.to_csv(predictions_path, index=False)

    print("\n" + "=" * 68)
    print("TRAINING COMPLETE")
    print("=" * 68)
    print(f"Model:       {model_path}")
    print(f"Metrics:     {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print("\nNote: test metrics were calculated once after threshold selection on validation data.")


if __name__ == "__main__":
    main()
