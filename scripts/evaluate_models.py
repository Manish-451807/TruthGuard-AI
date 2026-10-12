"""
scripts/evaluate_models.py

Re-tests the SAVED models on their held-out test files (without re-training)
and checks the numbers against the JSON reports the dashboard shows.

Run from the project root:   python scripts/evaluate_models.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from truthguard.evaluation import evaluate_classifier  # noqa: E402
from truthguard.modeling import ModelUnavailableError, load_bundle  # noqa: E402

TASKS = {
    # task: (model file, test csv, text column)
    "credibility": ("credibility_model.joblib", "test.csv", "statement"),
    "sentiment": ("sentiment_model.joblib", "sentiment_test.csv", "text"),
}


def evaluate_task(task):
    model_file, test_file, text_col = TASKS[task]
    model_path = config.MODELS_DIR / model_file
    test_path = config.PROCESSED_DATA_DIR / test_file
    if not model_path.exists():
        print(f"[{task}] no trained model found ({model_file}); skipping.")
        return
    if not test_path.exists():
        print(f"[{task}] test file {test_path.name} not found; skipping.")
        return
    try:
        bundle = load_bundle(model_path)
    except ModelUnavailableError as err:
        print(f"[{task}] ERROR: {err}")
        return

    test = pd.read_csv(test_path).dropna()
    metrics = evaluate_classifier(bundle["pipeline"], test[text_col].astype(str), test["label"])
    print(f"\n[{task}] model: {bundle['model_name']} | demo: {bundle['is_demo']} | test rows: {len(test)}")
    print(f"  accuracy {metrics['accuracy']} | macro-F1 {metrics['macro_f1']} | "
          f"weighted-F1 {metrics['weighted_f1']} | ROC-AUC {metrics['roc_auc_macro_ovr']}")
    for label, m in metrics["per_class"].items():
        print(f"    {label:<22} precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
              f"F1 {m['f1']:.3f}  (n={m['support']})")

    report_path = config.MODELS_DIR / f"{task}_metrics.json"
    if report_path.exists():
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        same = abs(saved.get("accuracy", -1) - metrics["accuracy"]) < 1e-3
        print("  matches the saved dashboard report." if same else
              "  WARNING: differs from the saved report. Re-train to refresh the dashboard.")
    if bundle["is_demo"]:
        print("  NOTE: demo model, not meaningful results.")


def main():
    for task in TASKS:
        evaluate_task(task)


if __name__ == "__main__":
    main()
