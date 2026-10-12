"""
scripts/train_sentiment_model.py

Trains the sentiment classifier (positive / neutral / negative).

Dataset: TweetEval "sentiment" task (social media posts, 3 classes).
See scripts/download_instructions.md. Expected files in --data-dir:
    train_text.txt  train_labels.txt
    val_text.txt    val_labels.txt
    test_text.txt   test_labels.txt
(one text / one numeric label per line; 0=negative, 1=neutral, 2=positive)

Usage (project root):
    python scripts/train_sentiment_model.py                 # real TweetEval files
    python scripts/train_sentiment_model.py --use-sample    # tiny synthetic demo
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from truthguard.training import print_summary, train_and_evaluate  # noqa: E402

DEFAULT_DIR = config.DATA_DIR / "raw" / "tweeteval" / "sentiment"
LABEL_NAMES = {"0": "negative", "1": "neutral", "2": "positive"}


def read_tweeteval_split(directory, name):
    """Read <name>_text.txt and <name>_labels.txt into a DataFrame."""
    text_path = directory / f"{name}_text.txt"
    label_path = directory / f"{name}_labels.txt"
    for path in (text_path, label_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. See scripts/download_instructions.md.")
        if path.stat().st_size == 0:
            raise ValueError(f"{path} is empty.")
    # Split on "\n" only (str.splitlines would also split on odd Unicode separators).
    with open(text_path, encoding="utf-8") as fh:
        texts = fh.read().split("\n")
    if texts and texts[-1] == "":
        texts.pop()
    labels = label_path.read_text(encoding="utf-8").split()
    if len(texts) != len(labels):
        raise ValueError(f"{name}: {len(texts)} texts but {len(labels)} labels. Files do not match.")
    unknown = sorted(set(labels) - set(LABEL_NAMES))
    if unknown:
        raise ValueError(f"{name}: unexpected labels {unknown}; expected 0, 1 or 2.")
    df = pd.DataFrame({"text": texts, "label": [LABEL_NAMES[x] for x in labels]})
    df["text"] = df["text"].str.strip()
    return df[df["text"] != ""].reset_index(drop=True)


def load_real(directory):
    splits = {n: read_tweeteval_split(directory, n) for n in ("train", "val", "test")}
    # Leakage guard: drop training texts that also occur in validation/test data.
    held_out = set(splits["val"]["text"]) | set(splits["test"]["text"])
    leaked = splits["train"]["text"].isin(held_out)
    if leaked.any():
        print(f"Removed {int(leaked.sum())} training rows that also appear in validation/test data.")
    train = splits["train"][~leaked].drop_duplicates("text").reset_index(drop=True)
    info = {"source": "TweetEval sentiment (Barbieri et al., 2020)", "is_demo": False}
    return train, splits["val"], splits["test"], info


def load_sample():
    path = config.SAMPLE_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Sample file missing: {path}")
    df = pd.read_csv(path)
    if not {"statement", "sentiment"} <= set(df.columns):
        raise ValueError("sample_news.csv needs 'statement' and 'sentiment' columns.")
    # Pick the two columns FIRST: the file also has a credibility 'label' column.
    df = df[["statement", "sentiment"]].rename(columns={"statement": "text", "sentiment": "label"})
    train, test = train_test_split(df, test_size=config.TEST_SIZE,
                                   random_state=config.RANDOM_SEED, stratify=df["label"])
    info = {"source": "Synthetic demonstration data (NOT real research data)", "is_demo": True}
    return train.reset_index(drop=True), None, test.reset_index(drop=True), info


def main():
    parser = argparse.ArgumentParser(description="Train the sentiment model.")
    parser.add_argument("--use-sample", action="store_true", help="use the synthetic demo data")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIR, help="folder with TweetEval files")
    args = parser.parse_args()

    try:
        train, valid, test, info = load_sample() if args.use_sample else load_real(args.data_dir)
    except (FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}")
        sys.exit(1)

    # Save the splits so evaluate_models.py can re-test the saved model later.
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.PROCESSED_DATA_DIR / "sentiment_train.csv", index=False)
    test.to_csv(config.PROCESSED_DATA_DIR / "sentiment_test.csv", index=False)
    if valid is not None:
        valid.to_csv(config.PROCESSED_DATA_DIR / "sentiment_valid.csv", index=False)

    print(f"Training sentiment model ({info['source']})")
    metrics = train_and_evaluate(
        task="sentiment", train_df=train, test_df=test, valid_df=valid,
        text_col="text", label_col="label", model_filename="sentiment_model.joblib",
        is_demo=info["is_demo"], dataset_info=info)
    print_summary(metrics)
    print(f"\nSaved: {config.MODELS_DIR / 'sentiment_model.joblib'}")
    print("Remember: sentiment models can miss sarcasm and context.")


if __name__ == "__main__":
    main()
