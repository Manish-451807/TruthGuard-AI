"""Shared pytest fixtures: every route test gets its OWN temporary database."""
import pytest

from app import create_app


@pytest.fixture
def app(tmp_path):
    flask_app = create_app({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,                 # forms are tested without tokens...
        "DATABASE_PATH": str(tmp_path / "test.db"),
    })
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()
