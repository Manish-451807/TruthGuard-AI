"""
truthguard/modeling.py - everything the credibility and sentiment models share.

* clean_text()          one preprocessing function used in training AND serving
* pipeline builders     TF-IDF + Naive Bayes / Logistic Regression
* save_bundle / load_bundle   store a trained pipeline with its metadata
* ClassifierService     loads a model ONCE and answers prediction requests

IMPORTANT: the training scripts (Part 4) must save models with save_bundle()
so that this file can load them.
"""
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import config

_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_LETTERS = re.compile(r"[^a-z\s']")


class ModelUnavailableError(RuntimeError):
    """Raised when no usable model (trained or demo) can be loaded."""


def clean_text(text: str) -> str:
    """Lowercase, remove URLs and non-letters, collapse whitespace."""
    text = _URL.sub(" ", str(text).lower())
    text = _NON_LETTERS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _vectorizer() -> TfidfVectorizer:
    # The vectorizer is part of the pipeline, so it is fitted ONLY on the
    # training data when pipeline.fit() is called.
    return TfidfVectorizer(preprocessor=clean_text, stop_words="english",
                           ngram_range=(1, 2), sublinear_tf=True)


def build_nb_pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline([("tfidf", _vectorizer()), ("clf", MultinomialNB(alpha=alpha))])


def build_logreg_pipeline() -> Pipeline:
    return Pipeline([("tfidf", _vectorizer()),
                     ("clf", LogisticRegression(max_iter=1000,
                                                class_weight="balanced",
                                                random_state=config.RANDOM_SEED))])


# ------------------------------------------------------------------ saving / loading
def save_bundle(path, pipeline, task, is_demo, extra=None):
    """Save pipeline + metadata in one joblib file."""
    bundle = {
        "pipeline": pipeline,
        "task": task,
        "labels": [str(c) for c in pipeline.classes_],
        "is_demo": bool(is_demo),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": type(pipeline[-1]).__name__,
        "label_mode": config.LABEL_MODE,
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path):
    """Load a bundle. Only files inside models/ are accepted (they are pickles)."""
    path = Path(path).resolve()
    if config.MODELS_DIR.resolve() not in path.parents:
        raise ModelUnavailableError("Model files must be inside the models/ folder.")
    try:
        bundle = joblib.load(path)
    except FileNotFoundError:
        raise
    except Exception as exc:  # corrupted or incompatible file
        raise ModelUnavailableError(
            f"{path.name} could not be loaded ({type(exc).__name__}). "
            "Re-train the model with the scripts in scripts/.") from exc
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ModelUnavailableError(f"{path.name} has an unexpected format.")
    return bundle


def train_demo_bundle(target_column):
    """Fit a tiny model on the SYNTHETIC sample file so the app can start."""
    path = config.SAMPLE_DATA_PATH
    if not path.exists():
        raise ModelUnavailableError(
            "No trained model found and the demo sample file is missing. "
            "Run the training scripts (see README).")
    df = pd.read_csv(path)
    if "statement" not in df.columns or target_column not in df.columns:
        raise ModelUnavailableError("The demo sample file has missing columns.")
    y = df[target_column].astype(str)
    if target_column == "label" and config.LABEL_MODE == "grouped_two_class":
        y = y.map(config.GROUPED_TWO_CLASS)
    pipeline = build_nb_pipeline().fit(df["statement"], y)
    return {"pipeline": pipeline, "labels": [str(c) for c in pipeline.classes_],
            "is_demo": True, "model_name": "MultinomialNB (demo)", "task": target_column}


# ------------------------------------------------------------------ serving
class ClassifierService:
    """Loads one model lazily, keeps it in memory, and predicts."""

    def __init__(self, task, model_filename, demo_column):
        self.task = task
        self.model_path = config.MODELS_DIR / model_filename
        self.demo_column = demo_column
        self._bundle = None

    def reload(self):
        """Forget the cached model (used after re-training and in tests)."""
        self._bundle = None

    def _get_bundle(self):
        if self._bundle is None:
            try:
                self._bundle = load_bundle(self.model_path)
            except FileNotFoundError:
                # No trained model yet -> clearly-labelled demo model.
                self._bundle = train_demo_bundle(self.demo_column)
        return self._bundle

    def status(self):
        """Small dict for the UI: is the model usable, and in which mode?"""
        try:
            bundle = self._get_bundle()
        except ModelUnavailableError as exc:
            return {"available": False, "mode": "unavailable", "message": str(exc)}
        mode = "demo" if bundle.get("is_demo") else "trained"
        msg = ("Demonstration mode: the model was fitted on a tiny synthetic sample, "
               "so results are not meaningful." if mode == "demo"
               else "Using the trained model.")
        return {"available": True, "mode": mode, "message": msg,
                "model_name": bundle.get("model_name", "")}

    def predict(self, text):
        bundle = self._get_bundle()
        pipeline = bundle["pipeline"]
        features = pipeline[:-1].transform([text])
        classifier = pipeline[-1]
        known_words = features.nnz > 0       # False if no word was seen in training
        if hasattr(classifier, "predict_proba"):
            proba = classifier.predict_proba(features)[0]
            order = proba.argsort()[::-1]
            probabilities = [{"label": str(classifier.classes_[i]),
                              "probability": round(float(proba[i]), 4)} for i in order]
            label = probabilities[0]["label"]
            confidence = probabilities[0]["probability"]
        else:
            label = str(classifier.predict(features)[0])
            probabilities, confidence = [], None
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
            "known_words": known_words,
            "mode": "demo" if bundle.get("is_demo") else "trained",
            "model_name": bundle.get("model_name", ""),
        }
