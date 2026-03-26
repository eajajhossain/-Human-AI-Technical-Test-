"""
TableTalk - Task 2: Narrative Tone Classification
Maps RAVDESS emotions → narrative tones, trains and evaluates a classifier.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import joblib

# RAVDESS emotion → TableTalk narrative tone

NARRATIVE_MAP = {
    "neutral":   "calm_description",
    "calm":      "calm_description",
    "happy":     "character_dialogue",
    "sad":       "dramatic_emphasis",
    "angry":     "urgency",
    "fearful":   "suspense",
    "disgust":   "urgency",
    "surprised": "dramatic_emphasis",
}

# Feature columns used for training

def get_feature_cols(df: pd.DataFrame) -> list:
    """Return numeric feature columns (exclude metadata and label columns)."""
    exclude = {
        "filepath", "filename", "modality", "vocal_channel", "emotion_code",
        "emotion_label", "intensity", "statement", "repetition", "actor",
        "narrative_tone",
    }
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]

# Load & prepare data

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["narrative_tone"] = df["emotion_label"].map(NARRATIVE_MAP)
    df = df.dropna(subset=["narrative_tone"])
    print(f"[Task2] Loaded {len(df)} samples, {df['narrative_tone'].nunique()} narrative tones")
    print("\nNarrative tone distribution:")
    print(df["narrative_tone"].value_counts().to_string())
    return df


# Model definitions

def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, max_depth=15,
                                              random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                                   max_depth=4, random_state=42)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=5.0, gamma="scale",
                           probability=True, random_state=42)),
        ]),
    }

# Training & evaluation

def evaluate_models(df: pd.DataFrame, output_dir: str = "./outputs") -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    feat_cols = get_feature_cols(df)
    X = df[feat_cols].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(df["narrative_tone"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models  = build_models()
    results = {}
    best_model, best_f1 = None, 0.0

    print(f"\n[Task2] Training on {len(X_train)} samples, evaluating on {len(X_test)} ...\n")

    for name, pipe in models.items():
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)

        # Fit & test
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro")

        results[name] = {
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std":  round(cv_scores.std(), 4),
            "test_acc":   round(acc, 4),
            "test_f1":    round(f1, 4),
            "model":      pipe,
            "y_pred":     y_pred,
        }

        print(f"  {name:<25}  CV F1={cv_scores.mean():.4f}±{cv_scores.std():.4f}  "
              f"Test Acc={acc:.4f}  Test F1={f1:.4f}")

        if f1 > best_f1:
            best_f1   = f1
            best_model = name

    print(f"\n[Task2] Best model: {best_model}  (F1={best_f1:.4f})")

    #  Save best model 
    best_pipe = results[best_model]["model"]
    joblib.dump(best_pipe, out / "best_classifier.pkl")
    joblib.dump(le,        out / "label_encoder.pkl")
    np.save(out / "feature_columns.npy", np.array(feat_cols))
    print(f"[Task2] Saved best model → {out / 'best_classifier.pkl'}")

    #  Classification report 
    y_pred_best = results[best_model]["y_pred"]
    report = classification_report(y_test, y_pred_best,
                                    target_names=le.classes_, digits=4)
    print(f"\nClassification Report ({best_model}):\n{report}")
    (out / "classification_report.txt").write_text(
        f"Best Model: {best_model}\n\n{report}"
    )

    #  Confusion matrix plot 
    _plot_confusion_matrix(y_test, y_pred_best, le.classes_, best_model, out)

    #  Feature importance (Random Forest) 
    if "Random Forest" in results:
        _plot_feature_importance(results["Random Forest"]["model"], feat_cols, out)

    #  Summary CSV 
    summary = pd.DataFrame([
        {"Model": k, "CV_F1_mean": v["cv_f1_mean"], "CV_F1_std": v["cv_f1_std"],
         "Test_Acc": v["test_acc"], "Test_F1": v["test_f1"]}
        for k, v in results.items() if k != "model"
    ])
    summary.to_csv(out / "model_comparison.csv", index=False)

    return results, le, feat_cols


# Plotting helpers

def _plot_confusion_matrix(y_true, y_pred, class_names, model_name, out):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f"{title}\n{model_name}", fontsize=12)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Task2] Confusion matrix saved → {out / 'confusion_matrix.png'}")


def _plot_feature_importance(rf_pipe, feat_cols, out, top_n=20):
    rf_clf = rf_pipe.named_steps["clf"]
    importances = rf_clf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feat_cols[i] for i in idx[::-1]], importances[idx[::-1]], color="#4A90D9")
    ax.set_title("Top Feature Importances (Random Forest)", fontsize=13)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Task2] Feature importance plot saved → {out / 'feature_importance.png'}")



# Predict on new file

def predict_tone(filepath: str, model_dir: str = "./outputs") -> str:
    """Predict narrative tone for a single audio file."""
    from src.audio.task1_audio_processing import extract_features

    model_dir = Path(model_dir)
    pipe      = joblib.load(model_dir / "best_classifier.pkl")
    le        = joblib.load(model_dir / "label_encoder.pkl")
    feat_cols = list(np.load(model_dir / "feature_columns.npy", allow_pickle=True))

    feats = extract_features(filepath)
    row   = {k: feats.get(k, 0.0) for k in feat_cols}
    X     = np.array([[row[k] for k in feat_cols]])
    pred  = pipe.predict(X)[0]
    proba = pipe.predict_proba(X)[0]

    label = le.inverse_transform([pred])[0]
    conf  = proba[pred]
    print(f"  Predicted tone: {label}  (confidence={conf:.3f})")
    return label


# Entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TableTalk Narrative Tone Classifier")
    parser.add_argument("--csv",        type=str, default="./outputs/features_dataset.csv")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    df = load_data(args.csv)
    evaluate_models(df, args.output_dir)