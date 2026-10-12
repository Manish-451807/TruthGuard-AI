"""
truthguard/credibility.py - news credibility CLASSIFICATION.

The model predicts which fact-check rating a statement's wording most resembles
in its TRAINING DATA. It does not check facts and never proves anything.
"""
from truthguard.modeling import ClassifierService, ModelUnavailableError  # noqa: F401

service = ClassifierService(task="credibility",
                            model_filename="credibility_model.joblib",
                            demo_column="label")

NOTICE = ("This is a statistical prediction based on word patterns in the training "
          "data. It is NOT fact verification. Check claims against reliable, "
          "independent sources.")

LABEL_DESCRIPTIONS = {
    "pants-fire": "Wording resembles statements rated 'Pants on Fire' in the training data.",
    "false": "Wording resembles statements rated 'False' in the training data.",
    "barely-true": "Wording resembles statements rated 'Barely True' in the training data.",
    "half-true": "Wording resembles statements rated 'Half True' in the training data.",
    "mostly-true": "Wording resembles statements rated 'Mostly True' in the training data.",
    "true": "Wording resembles statements rated 'True' in the training data.",
    "less_credible_group": "Wording resembles the group of statements rated pants-fire, "
                           "false or barely-true (our own grouping).",
    "more_credible_group": "Wording resembles the group of statements rated half-true, "
                           "mostly-true or true (our own grouping).",
}


def status():
    return service.status()


def analyze(text):
    """Return a prediction dict for already-validated text."""
    result = service.predict(text)
    result["label_description"] = LABEL_DESCRIPTIONS.get(result["label"], "")
    result["notice"] = NOTICE
    if not result["known_words"]:
        result["warning"] = ("None of the words in this text appeared in the training "
                             "data, so this result reflects only class frequencies.")
    elif result["confidence"] is not None and result["confidence"] < 0.5:
        result["warning"] = "Low confidence: treat this prediction with extra caution."
    if result["mode"] == "demo":
        result["demo_notice"] = ("Demonstration mode: trained on a tiny synthetic "
                                 "sample. Train on the real dataset for meaningful results.")
    return result
