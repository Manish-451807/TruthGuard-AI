"""
truthguard/evaluation.py - honest model evaluation helpers.

Everything here is computed on data the model did NOT train on. The numbers
are written to JSON so the dashboard shows real results only.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config
from truthguard.modeling import build_nb_pipeline

MAX_ROC_POINTS = 60


def evaluate_classifier(pipeline, texts, y_true):
    """Return a metrics dict for a fitted pipeline on held-out data."""
    classes = [str(c) for c in pipeline.classes_]
    y_true = np.asarray([str(v) for v in y_true])
    y_pred = pipeline.predict(texts)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(prec.mean()), 4),
        "macro_recall": round(float(rec.mean()), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "labels": classes,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "per_class": {c: {"precision": round(float(prec[i]), 4),
                          "recall": round(float(rec[i]), 4),
                          "f1": round(float(f1[i]), 4),
                          "support": int(support[i])} for i, c in enumerate(classes)},
        "roc_auc_macro_ovr": None,
        "roc_curves": {},
    }

    # ROC-AUC only makes sense with probabilities, and only for classes that
    # appear in the test set with at least one positive AND one negative example.
    if hasattr(pipeline[-1], "predict_proba"):
        proba = pipeline.predict_proba(texts)
        aucs = []
        for i, label in enumerate(classes):
            positive = (y_true == label).astype(int)
            if positive.sum() == 0 or positive.sum() == len(positive):
                continue
            aucs.append(roc_auc_score(positive, proba[:, i]))
            fpr, tpr, _ = roc_curve(positive, proba[:, i])
            if len(fpr) > MAX_ROC_POINTS:                       # keep the JSON small
                keep = np.linspace(0, len(fpr) - 1, MAX_ROC_POINTS).astype(int)
                fpr, tpr = fpr[keep], tpr[keep]
            metrics["roc_curves"][label] = {"fpr": [round(float(x), 4) for x in fpr],
                                            "tpr": [round(float(x), 4) for x in tpr]}
        if aucs:
            metrics["roc_auc_macro_ovr"] = round(float(np.mean(aucs)), 4)
    return metrics


def majority_baseline_accuracy(y_train, y_test):
    """Accuracy of always predicting the most common training class."""
    majority = y_train.value_counts().idxmax()
    return round(float((np.asarray(y_test) == majority).mean()), 4)


def cross_validate_macro_f1(alpha, texts, labels):
    """Stratified CV on the TRAINING data only. Returns None if too little data."""
    smallest_class = int(labels.value_counts().min())
    folds = min(5, smallest_class)
    if folds < 2:
        return None
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_SEED)
    scores = cross_val_score(build_nb_pipeline(alpha), texts, labels,
                             cv=cv, scoring="f1_macro")
    return {"folds": folds, "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4)}


def save_confusion_matrix_png(path, matrix, labels, title):
    """Save a seaborn heatmap of the confusion matrix (no display window needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    path.parent.mkdir(parents=True, exist_ok=True)
    size = max(4, 0.9 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
