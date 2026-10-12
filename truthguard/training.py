"""
truthguard/training.py - the ONE training routine used by both training scripts.

Steps (nothing here ever touches the test set until the very end):
  1. choose the Naive Bayes smoothing value on the validation set (if any)
  2. fit Naive Bayes (served model) and Logistic Regression (comparison)
     on the training data only - the TF-IDF vectorizer is inside the pipeline,
     so it is also fitted on training data only
  3. evaluate both on the held-out test set
  4. cross-validate on the training data
  5. save the Naive Bayes model, a JSON metrics report and a confusion matrix PNG
"""
import json

import numpy as np
from sklearn.metrics import f1_score

import config
from truthguard.evaluation import (cross_validate_macro_f1, evaluate_classifier,
                                   majority_baseline_accuracy,
                                   save_confusion_matrix_png)
from truthguard.modeling import (build_logreg_pipeline, build_nb_pipeline,
                                 save_bundle)

ALPHA_GRID = [0.1, 0.3, 1.0]


def _fit(pipeline, df, text_col, label_col):
    return pipeline.fit(df[text_col].astype(str), df[label_col].astype(str))


def train_and_evaluate(task, train_df, test_df, valid_df, text_col, label_col,
                       model_filename, is_demo, dataset_info=None, extra_meta=None):
    if train_df[label_col].nunique() < 2:
        raise ValueError("Training data needs at least two different classes.")
    if test_df.empty:
        raise ValueError("The test set is empty.")

    # 1. Choose the smoothing value without looking at the test set.
    alpha, alpha_note = 1.0, "default (no validation set available)"
    if valid_df is not None and len(valid_df):
        best = -1.0
        for candidate in ALPHA_GRID:
            model = _fit(build_nb_pipeline(candidate), train_df, text_col, label_col)
            score = f1_score(valid_df[label_col].astype(str),
                             model.predict(valid_df[text_col].astype(str)),
                             average="macro", zero_division=0)
            if score > best:
                best, alpha = score, candidate
        alpha_note = f"chosen on the validation set (macro-F1 {best:.3f})"

    # 2. Fit both models on the training data only.
    nb = _fit(build_nb_pipeline(alpha), train_df, text_col, label_col)
    lr = _fit(build_logreg_pipeline(), train_df, text_col, label_col)

    # 3. Evaluate on the held-out test set.
    X_test = test_df[text_col].astype(str)
    y_test = test_df[label_col].astype(str)
    metrics = evaluate_classifier(nb, X_test, y_test)
    lr_metrics = evaluate_classifier(lr, X_test, y_test)

    # 4. Cross-validation on training data only.
    X_train = train_df[text_col].astype(str)
    y_train = train_df[label_col].astype(str)
    cross_val = cross_validate_macro_f1(alpha, X_train, y_train)

    metrics.update({
        "task": task,
        "model_name": "MultinomialNB",
        "is_demo": bool(is_demo),
        "evaluated_on": "held-out test set",
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)) if valid_df is not None else 0,
        "n_test": int(len(test_df)),
        "nb_alpha": alpha,
        "nb_alpha_note": alpha_note,
        "cross_val": cross_val,
        "baseline_majority_accuracy": majority_baseline_accuracy(y_train, y_test),
        "comparison": {
            "MultinomialNB": {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                              "weighted_f1": metrics["weighted_f1"]},
            "LogisticRegression": {"accuracy": lr_metrics["accuracy"], "macro_f1": lr_metrics["macro_f1"],
                                   "weighted_f1": lr_metrics["weighted_f1"]},
        },
        "random_seed": config.RANDOM_SEED,
        "dataset": dataset_info or {},
    })
    if extra_meta:
        metrics.update(extra_meta)

    # 5. Save everything.
    save_bundle(config.MODELS_DIR / model_filename, nb, task=task, is_demo=is_demo,
                extra={"nb_alpha": alpha})
    metrics_path = config.MODELS_DIR / f"{task}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    save_confusion_matrix_png(config.REPORTS_DIR / f"{task}_confusion_matrix.png",
                              np.array(metrics["confusion_matrix"]), metrics["labels"],
                              f"{task.title()} - confusion matrix (test set)")
    return metrics


def print_summary(metrics):
    """Plain-text summary that clearly says these are TEST-set numbers."""
    print("\n=== Results on the held-out TEST set (not training data) ===")
    print(f"  Rows: train {metrics['n_train']}, validation {metrics['n_valid']}, test {metrics['n_test']}")
    print(f"  Majority-class baseline accuracy : {metrics['baseline_majority_accuracy']}")
    print(f"  Naive Bayes accuracy             : {metrics['accuracy']}")
    print(f"  Naive Bayes macro-F1 / weighted-F1: {metrics['macro_f1']} / {metrics['weighted_f1']}")
    for name, c in metrics["comparison"].items():
        print(f"  [{name}] accuracy {c['accuracy']}, macro-F1 {c['macro_f1']}")
    if metrics["cross_val"]:
        cv = metrics["cross_val"]
        print(f"  {cv['folds']}-fold CV macro-F1 (train only): {cv['mean']} +/- {cv['std']}")
    print(f"  ROC-AUC (macro one-vs-rest): {metrics['roc_auc_macro_ovr']}")
    print(f"  Naive Bayes alpha: {metrics['nb_alpha']} ({metrics['nb_alpha_note']})")
    if metrics["is_demo"]:
        print("\n  NOTE: DEMO data. These numbers only show the pipeline works; "
              "they are not real results.")
