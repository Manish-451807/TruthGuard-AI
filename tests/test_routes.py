"""Tests for the Flask routes in app.py (using a temporary database)."""
import pytest

import config
from app import create_app
from truthguard import database

HEADLINE = "Riverton added a few new bike lanes but most roads still have none."


# ------------------------------------------------------------------ pages
@pytest.mark.parametrize("path", ["/", "/analyzer", "/sentiment", "/clustering",
                                  "/summarizer", "/dashboard", "/history"])
def test_main_pages_load(client, path):
    assert client.get(path).status_code == 200


def test_unknown_page_gives_friendly_404(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert b"Page not found" in response.data
    assert b"Traceback" not in response.data


def test_home_page_states_the_limitations(client):
    assert b"not</strong> proof" in client.get("/").data


def test_navigation_links_are_present(client):
    html = client.get("/").data.decode()
    for target in ("/analyzer", "/sentiment", "/clustering", "/summarizer", "/dashboard", "/history"):
        assert f'href="{target}"' in html


# ------------------------------------------------------------------ credibility
def test_analyzer_returns_a_neutral_prediction_with_disclaimer(client):
    response = client.post("/analyzer", data={"text": HEADLINE})
    html = response.data.decode()
    assert response.status_code == 200
    assert "Model prediction" in html
    assert "NOT fact verification" in html
    assert "not the chance the claim is true" in html      # confidence is explained, no verdict given


@pytest.mark.parametrize("text, fragment", [("", "empty"), ("   ", "empty"), ("hi", "too short")])
def test_analyzer_rejects_bad_input(client, text, fragment):
    response = client.post("/analyzer", data={"text": text})
    assert response.status_code == 400
    assert fragment in response.data.decode()


def test_analyzer_rejects_overlong_input(client):
    response = client.post("/analyzer", data={"text": "x" * (config.MAX_TEXT_LENGTH + 1)})
    assert response.status_code == 400
    assert b"too long" in response.data


def test_analyzer_handles_missing_field(client):
    assert client.post("/analyzer", data={}).status_code == 400


def test_analyzer_reports_model_unavailable(client, monkeypatch):
    from truthguard import credibility
    from truthguard.modeling import ModelUnavailableError

    def broken(_text):
        raise ModelUnavailableError("The model could not be loaded.")
    monkeypatch.setattr(credibility, "analyze", broken)
    response = client.post("/analyzer", data={"text": HEADLINE})
    assert response.status_code == 503
    assert b"could not be loaded" in response.data


# ------------------------------------------------------------------ history
def test_history_saved_only_when_requested(client, app):
    client.post("/analyzer", data={"text": HEADLINE})                          # box not ticked
    assert database.get_recent(db_path=app.config["DATABASE_PATH"]) == []
    client.post("/analyzer", data={"text": HEADLINE, "save_history": "on"})    # box ticked
    rows = database.get_recent(db_path=app.config["DATABASE_PATH"])
    assert len(rows) == 1 and rows[0]["analysis_type"] == "credibility"


def test_history_page_escapes_html(client):
    client.post("/analyzer", data={"text": "<script>alert(1)</script> some news words", "save_history": "on"})
    html = client.get("/history").data.decode()
    assert "<script>alert(1)" not in html
    assert "&lt;script&gt;alert(1)" in html


def test_delete_single_entry_and_clear_all(client, app):
    path = app.config["DATABASE_PATH"]
    for _ in range(2):
        client.post("/sentiment", data={"text": "The park is wonderful today.", "save_history": "on"})
    first = database.get_recent(db_path=path)[0]["id"]
    assert client.post(f"/history/{first}/delete").status_code == 302
    assert len(database.get_recent(db_path=path)) == 1
    assert client.post("/history/clear").status_code == 302
    assert database.get_recent(db_path=path) == []


def test_deleting_a_missing_entry_does_not_crash(client):
    assert client.post("/history/9999/delete").status_code == 302


def test_delete_requires_post(client):
    assert client.get("/history/1/delete").status_code == 405


# ------------------------------------------------------------------ sentiment
def test_sentiment_page_shows_distribution(client):
    response = client.post("/sentiment", data={"text": "The new park is wonderful and I love it."})
    assert response.status_code == 200
    assert b"Sentiment distribution" in response.data and b"bar-fill" in response.data


# ------------------------------------------------------------------ clustering
TEXTS = "\n".join(["park playground opened", "library books lending", "park walking track opened",
                   "library extended hours books", "bus service schedule changed"])


def test_clustering_works(client):
    response = client.post("/clustering", data={"k": "2", "texts": TEXTS})
    assert response.status_code == 200
    assert b"Cluster 1" in response.data and b"Cluster 2" in response.data


@pytest.mark.parametrize("k", ["abc", "1", "99", ""])
def test_clustering_rejects_bad_k(client, k):
    assert client.post("/clustering", data={"k": k, "texts": TEXTS}).status_code == 400


def test_clustering_rejects_too_few_texts(client):
    assert client.post("/clustering", data={"k": "3", "texts": "just one line"}).status_code == 400


def test_clustering_rejects_identical_texts(client):
    same = "\n".join(["same words here"] * 6)
    assert client.post("/clustering", data={"k": "3", "texts": same}).status_code == 400


# ------------------------------------------------------------------ summarizer
def test_summarizer_reports_unavailable_without_api_key(client, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    assert b"AI summaries are unavailable" in client.get("/summarizer").data
    response = client.post("/summarizer", data={"text": "A long article. " * 20})
    assert response.status_code == 503
    assert b"Summary <span" not in response.data           # no fake summary was produced


def test_summarizer_validates_input(client):
    assert client.post("/summarizer", data={"text": "too short"}).status_code == 400


# ------------------------------------------------------------------ dashboard / API
def test_dashboard_api_returns_real_structure(client):
    data = client.get("/api/dashboard-data").get_json()
    assert set(data) == {"live", "offline", "model_status"}
    assert data["live"]["total"] == 0                       # fresh database: nothing invented


def test_dashboard_counts_follow_the_database(client):
    client.post("/sentiment", data={"text": "The park is wonderful today.", "save_history": "on"})
    live = client.get("/api/dashboard-data").get_json()["live"]
    assert live["total"] == 1 and live["counts"] == {"sentiment": 1}


# ------------------------------------------------------------------ security
def test_security_headers_are_set(client):
    headers = client.get("/").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"


def test_post_without_csrf_token_is_rejected(tmp_path):
    strict_app = create_app({"TESTING": True, "DATABASE_PATH": str(tmp_path / "csrf.db")})
    response = strict_app.test_client().post("/analyzer", data={"text": HEADLINE})
    assert response.status_code == 400
    assert b"expired" in response.data


def test_oversized_request_is_rejected(client):
    response = client.post("/analyzer", data={"text": "x" * 1_200_000})
    assert response.status_code == 413
