"""Tests for the credibility/sentiment model layer (truthguard/modeling.py etc.)."""
import pytest

import config
from truthguard import credibility, sentiment
from truthguard.modeling import (ClassifierService, ModelUnavailableError,
                                 build_nb_pipeline, clean_text, load_bundle,
                                 save_bundle)

HEADLINE = "Riverton added a few new bike lanes but most roads still have none."


def test_clean_text_lowercases_and_strips_noise():
    assert clean_text("Visit HTTP://x.com NOW!!! It's 100% true...") == "visit now it's true"


def test_prediction_has_the_expected_shape():
    result = credibility.analyze(HEADLINE)
    assert result["label"]                      # a non-empty class name
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["mode"] in ("demo", "trained")
    assert abs(sum(p["probability"] for p in result["probabilities"]) - 1.0) < 0.01
    assert result["probabilities"][0]["label"] == result["label"]


def test_every_credibility_result_carries_the_not_fact_checking_notice():
    notice = credibility.analyze(HEADLINE)["notice"]
    assert "NOT fact verification" in notice


def test_unseen_words_produce_a_warning_not_a_crash():
    result = credibility.analyze("zzzzqq xxxyyy vvvwww")
    assert result["known_words"] is False
    assert "warning" in result


def test_sentiment_label_is_valid():
    result = sentiment.analyze("The new park is wonderful and everyone loves it.")
    assert result["label"] in ("positive", "negative", "neutral")
    assert "sarcasm" in result["notice"]


def test_status_reports_availability_and_mode():
    info = credibility.status()
    assert info["available"] is True
    assert info["mode"] in ("demo", "trained")


def test_missing_model_and_missing_sample_raises_model_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "SAMPLE_DATA_PATH", tmp_path / "nope.csv")
    service = ClassifierService("credibility", "credibility_model.joblib", "label")
    assert service.status()["available"] is False
    with pytest.raises(ModelUnavailableError):
        service.predict("some text to classify")


def test_corrupted_model_file_is_an_error_not_a_silent_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    (tmp_path / "credibility_model.joblib").write_bytes(b"this is not a joblib file")
    service = ClassifierService("credibility", "credibility_model.joblib", "label")
    status = service.status()
    assert status["available"] is False
    assert "could not be loaded" in status["message"]


def test_model_files_outside_models_folder_are_refused(tmp_path):
    outside = tmp_path / "evil.joblib"
    outside.write_bytes(b"x")
    with pytest.raises(ModelUnavailableError, match="models/"):
        load_bundle(outside)


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    pipeline = build_nb_pipeline().fit(
        ["good great happy", "bad awful sad", "good happy fine", "bad sad awful"],
        ["positive", "negative", "positive", "negative"])
    save_bundle(tmp_path / "m.joblib", pipeline, task="sentiment", is_demo=True)
    bundle = load_bundle(tmp_path / "m.joblib")
    assert bundle["is_demo"] is True and bundle["labels"] == ["negative", "positive"]
    assert bundle["pipeline"].predict(["happy good"])[0] == "positive"


def test_a_trained_model_file_is_used_and_not_labelled_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    pipeline = build_nb_pipeline().fit(["alpha beta", "gamma delta", "alpha beta beta", "gamma gamma delta"],
                                       ["x", "y", "x", "y"])
    save_bundle(tmp_path / "credibility_model.joblib", pipeline, task="credibility", is_demo=False)
    service = ClassifierService("credibility", "credibility_model.joblib", "label")
    assert service.status()["mode"] == "trained"
    assert service.predict("alpha beta")["label"] == "x"
