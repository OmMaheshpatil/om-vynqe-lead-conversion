"""
train.py – Model training pipeline for Lead Conversion Prediction.

Trains three classifiers (Logistic Regression, Random Forest, XGBoost),
evaluates them on a hold-out test set, and saves:
  • model.pkl         – best performing model (pickle)
  • outputs/model_metrics.json – all model metrics
  • outputs/feature_importance.json – feature importances (tree models)

Usage:
    python train.py
"""

from __future__ import annotations

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from utils import load_and_preprocess_data

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_and_evaluate() -> None:
    """End-to-end training pipeline."""

    # Load & preprocess data
    X, y, feature_names, label_encoders = load_and_preprocess_data()

    # Drop zero-variance features (e.g. email_opens)
    zero_var_cols = X.columns[X.std() == 0].tolist()
    if zero_var_cols:
        print(f"\n   Dropping zero-variance features: {zero_var_cols}")
        X = X.drop(columns=zero_var_cols)
        feature_names = [f for f in feature_names if f not in zero_var_cols]

    # Train/test split
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    print("\n1. Splitting data (stratified)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"   Train: {X_train.shape[0]} samples ({y_train.sum()} positives, "
          f"{(y_train.sum() / len(y_train) * 100):.1f}%)")
    print(f"   Test:  {X_test.shape[0]} samples ({y_test.sum()} positives, "
          f"{(y_test.sum() / len(y_test) * 100):.1f}%)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    print("\n2. Training models...\n")

    models = {
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                C=1.0,
                solver="lbfgs",
            ),
            "use_scaled": True,
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "use_scaled": False,
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=(len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
                random_state=RANDOM_STATE,
                verbosity=0,
                eval_metric="logloss",
            ),
            "use_scaled": False,
        },
    }

    # Train and evaluate models
    results = {}
    best_model = None
    best_model_name = ""
    best_f1 = -1.0

    for name, config in models.items():
        model = config["model"]
        use_scaled = config["use_scaled"]

        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        # Train
        model.fit(X_tr, y_train)

        # Predict
        y_pred = model.predict(X_te)
        y_pred_proba = model.predict_proba(X_te)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Cross-validation F1
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            config["model"].__class__(**config["model"].get_params()),
            X_tr, y_train, cv=cv, scoring="f1", n_jobs=-1
        )
        cv_f1_mean = cv_scores.mean()
        cv_f1_std = cv_scores.std()

        results[name] = {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "auc_roc": round(float(auc_roc), 4),
            "cv_f1_mean": round(float(cv_f1_mean), 4),
            "cv_f1_std": round(float(cv_f1_std), 4),
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"   +-- {name} {'-' * (45 - len(name))}+")
        print(f"   |  Accuracy:      {accuracy:.4f}                      |")
        print(f"   |  Precision:     {precision:.4f}                      |")
        print(f"   |  Recall:        {recall:.4f}                      |")
        print(f"   |  F1 Score:      {f1:.4f}                      |")
        print(f"   |  AUC-ROC:       {auc_roc:.4f}                      |")
        print(f"   |  CV F1:         {cv_f1_mean:.4f} +/- {cv_f1_std:.4f}            |")
        print(f"   |  Confusion:  TP={tp} FP={fp} FN={fn} TN={tn}     |")
        print(f"   +{'-' * 49}+\n")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    # Extract feature importances
    print("\n3. Extracting feature importances...")

    importances = {}
    if hasattr(best_model, "feature_importances_"):
        feat_imp = dict(zip(feature_names, best_model.feature_importances_.tolist()))
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
        importances = {k: round(v, 6) for k, v in feat_imp.items()}

        print(f"\n   Top 10 features ({best_model_name}):")
        for i, (feat, imp) in enumerate(list(importances.items())[:10]):
            bar = "#" * int(imp * 200)
            print(f"   {i+1:2d}. {feat:<35s} {imp:.4f} {bar}")
    elif hasattr(best_model, "coef_"):
        coef_imp = dict(zip(feature_names, np.abs(best_model.coef_[0]).tolist()))
        coef_imp = dict(sorted(coef_imp.items(), key=lambda x: x[1], reverse=True))
        importances = {k: round(v, 6) for k, v in coef_imp.items()}

        print(f"\n   Top 10 features ({best_model_name} - absolute coefficients):")
        for i, (feat, imp) in enumerate(list(importances.items())[:10]):
            print(f"   {i+1:2d}. {feat:<35s} {imp:.4f}")

    # Optimize classification threshold for F1-score
    print("\n4. Optimising classification threshold for F1-score...")
    X_te_final = X_test_scaled if models[best_model_name]["use_scaled"] else X_test
    y_pred_proba_final = best_model.predict_proba(X_te_final)[:, 1]

    best_threshold = 0.5
    best_threshold_f1 = -1.0
    for th in np.linspace(0.1, 0.9, 81):
        th_preds = (y_pred_proba_final >= th).astype(int)
        th_f1 = f1_score(y_test, th_preds, zero_division=0)
        if th_f1 > best_threshold_f1:
            best_threshold_f1 = th_f1
            best_threshold = float(th)

    print(f"   [OK] Optimal threshold: {best_threshold:.4f} (F1 = {best_threshold_f1:.4f} vs 0.5 threshold F1 = {best_f1:.4f})")

    # Save artifacts
    print(f"\n5. Saving best model ({best_model_name})...")

    # Save model bundle
    model_path = "model.pkl"
    model_data = {
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "label_encoders": label_encoders,
        "model_name": best_model_name,
        "use_scaled": models[best_model_name]["use_scaled"],
        "optimal_threshold": best_threshold,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"   [OK] Model bundle saved to {model_path}")

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.json")
    output_metrics = {
        "best_model": best_model_name,
        "best_f1_default_threshold": round(float(best_f1), 4),
        "best_f1_optimal_threshold": round(float(best_threshold_f1), 4),
        "optimal_threshold": round(best_threshold, 4),
        "models": results,
    }
    with open(metrics_path, "w") as f:
        json.dump(output_metrics, f, indent=2)
    print(f"   [OK] Metrics saved to {metrics_path}")

    # Save feature importances
    if importances:
        imp_path = os.path.join(OUTPUT_DIR, "feature_importance.json")
        with open(imp_path, "w") as f:
            json.dump(importances, f, indent=2)
        print(f"   [OK] Feature importances saved to {imp_path}")

        # Save feature importance plot
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(10, 6))
            top_n = 15
            plot_data = pd.DataFrame({
                'Feature': list(importances.keys())[:top_n],
                'Importance': list(importances.values())[:top_n]
            })
            sns.barplot(
                x='Importance',
                y='Feature',
                data=plot_data,
                palette='viridis',
                hue='Feature',
                legend=False
            )
            plt.title(f'Top {top_n} Feature Importance ({best_model_name})')
            plt.xlabel('Importance Value')
            plt.ylabel('Features')
            plt.tight_layout()

            plot_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"   [OK] Feature importance plot saved to {plot_path}")
        except Exception as e:
            print(f"   [WARNING] Failed to save feature importance plot: {e}")

    # Detailed classification report
    print(f"\n6. Detailed Classification Report at Optimal Threshold ({best_threshold:.4f}):\n")
    y_pred_final = (y_pred_proba_final >= best_threshold).astype(int)
    print(classification_report(y_test, y_pred_final, target_names=["Not Converted", "Converted"]))

    print("=" * 60)
    print(f"TRAINING COMPLETE - Best model: {best_model_name} (Optimal Threshold F1={best_threshold_f1:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_and_evaluate()
