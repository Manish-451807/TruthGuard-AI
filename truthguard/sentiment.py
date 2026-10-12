"""
truthguard/sentiment.py - positive / negative / neutral sentiment.
Completely separate from credibility: sentiment is about tone, not truth.
"""
from truthguard.modeling import ClassifierService, ModelUnavailableError  # noqa: F401

service = ClassifierService(task="sentiment",
                            model_filename="sentiment_model.joblib",
                            demo_column="sentiment")

NOTICE = ("Sentiment models read word patterns. They can miss sarcasm, irony, "
          "negation and context, and tone says nothing about whether a text is true.")


def status():
    return service.status()


def analyze(text):
    result = service.predict(text)
    result["notice"] = NOTICE
    if not result["known_words"]:
        result["warning"] = "None of these words appeared in the training data."
    if result["mode"] == "demo":
        result["demo_notice"] = ("Demonstration mode: trained on 24 synthetic examples, "
                                 "so results are not reliable.")
    return result
