"""
scripts/train_credibility_model.py

Trains the credibility classifier (TF-IDF + Multinomial Naive Bayes) on the
files made by prepare_dataset.py, compares it with Logistic Regression, and
saves the model + evaluation report.

Run from the project root:
    python scripts/prepare_dataset.py          (or --use-sample first)
    python scripts/train_credibility_model.py

Outputs: models/credibility_model.joblib, models/credibility_metrics.json,
         reports/credibility_confusion_matrix.png
Restart the Flask app afterwards so it loads the new model.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from truthguard.training import print_summary, train_and_evaluate  # noqa: E402


def read_split(name, required=True):
    path = config.PROCESSED_DATA_DIR / f"{name}.csv"
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"{path} not found. Run: python scripts/prepare_dataset.py "
                "(or add --use-sample for the demo data).")
        return None
    if path.stat().st_size == 0:
        raise ValueError(f"{path} is empty.")
    df = pd.read_csv(path)
    missing = {"statement", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["statement", "label"])
    if df.empty:
        raise ValueError(f"{path.name} has no usable rows.")
    return df


def main():
    try:
        train, test = read_split("train"), read_split("test")
        valid = read_split("valid", required=False)
        info_path = config.PROCESSED_DATA_DIR / "dataset_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"{info_path} not found. Run prepare_dataset.py first.")
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}")
        sys.exit(1)

    if info.get("label_mode") != config.LABEL_MODE:
        print(f"ERROR: the prepared data uses label mode '{info.get('label_mode')}' but "
              f".env says '{config.LABEL_MODE}'. Re-run prepare_dataset.py.")
        sys.exit(1)

    print(f"Training credibility model ({info['source']}, label mode: {config.LABEL_MODE})")
    metrics = train_and_evaluate(
        task="credibility", train_df=train, test_df=test, valid_df=valid,
        text_col="statement", label_col="label",
        model_filename="credibility_model.joblib",
        is_demo=info.get("is_demo", False), dataset_info=info,
        extra_meta={"label_mode": config.LABEL_MODE})
    print_summary(metrics)
    print(f"\nSaved: {config.MODELS_DIR / 'credibility_model.joblib'}")
    print(f"Saved: {config.MODELS_DIR / 'credibility_metrics.json'}")
    print(f"Saved: {config.REPORTS_DIR / 'credibility_confusion_matrix.png'}")
    print("Remember: predictions show learned wording patterns, not verified facts.")


if __name__ == "__main__":
    main()
