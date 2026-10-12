"""Tests for truthguard/database.py (SQLite history)."""
import pytest

import config
from truthguard import database


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "history.db"
    database.init_db(path)
    return path


def test_init_db_creates_a_usable_database_and_is_repeatable(tmp_path):
    path = tmp_path / "fresh.db"
    database.init_db(path)
    database.init_db(path)                      # calling twice must not fail or wipe data
    assert path.exists()
    assert database.get_recent(db_path=path) == []


def test_add_and_read_back_an_entry(db):
    entry_id = database.add_analysis("credibility", "A test statement", "half-true", 0.42,
                                     {"mode": "demo"}, db_path=db)
    rows = database.get_recent(db_path=db)
    assert len(rows) == 1 and rows[0]["id"] == entry_id
    assert rows[0]["result_label"] == "half-true"
    assert rows[0]["confidence"] == 0.42
    assert rows[0]["metadata"] == {"mode": "demo"}
    assert rows[0]["created_at"]


def test_only_a_short_excerpt_is_stored(db):
    long_text = "word " * 500
    database.add_analysis("summary", long_text, "summary", db_path=db)
    row = database.get_recent(db_path=db)[0]
    assert len(row["excerpt"]) <= config.SAVED_EXCERPT_LENGTH + 3      # + the "..." marker
    assert row["excerpt"].endswith("...")
    assert row["input_length"] == len(long_text)


def test_unknown_analysis_type_is_rejected(db):
    with pytest.raises(ValueError):
        database.add_analysis("hacking", "text", "label", db_path=db)


def test_newest_entries_come_first_and_limit_works(db):
    for i in range(5):
        database.add_analysis("sentiment", f"text {i}", "positive", db_path=db)
    rows = database.get_recent(limit=3, db_path=db)
    assert [r["excerpt"] for r in rows] == ["text 4", "text 3", "text 2"]


def test_filter_by_type(db):
    database.add_analysis("sentiment", "a", "positive", db_path=db)
    database.add_analysis("credibility", "b", "true", db_path=db)
    rows = database.get_recent(analysis_type="credibility", db_path=db)
    assert [r["analysis_type"] for r in rows] == ["credibility"]


def test_delete_one_entry(db):
    first = database.add_analysis("sentiment", "a", "positive", db_path=db)
    database.add_analysis("sentiment", "b", "negative", db_path=db)
    assert database.delete_analysis(first, db_path=db) is True
    assert database.delete_analysis(first, db_path=db) is False          # already gone
    assert [r["excerpt"] for r in database.get_recent(db_path=db)] == ["b"]


def test_clear_history_returns_number_removed(db):
    for i in range(3):
        database.add_analysis("sentiment", f"t{i}", "neutral", db_path=db)
    assert database.clear_history(db_path=db) == 3
    assert database.get_recent(db_path=db) == []


def test_counts_and_label_distribution(db):
    database.add_analysis("sentiment", "a", "positive", db_path=db)
    database.add_analysis("sentiment", "b", "positive", db_path=db)
    database.add_analysis("sentiment", "c", "negative", db_path=db)
    database.add_analysis("credibility", "d", "true", db_path=db)
    assert database.count_by_type(db_path=db) == {"sentiment": 3, "credibility": 1}
    assert database.label_distribution("sentiment", db_path=db) == {"positive": 2, "negative": 1}


def test_latest_metadata(db):
    assert database.latest_metadata("clustering", db_path=db) is None
    database.add_analysis("clustering", "x", "2 clusters", metadata={"sizes": [3, 4]}, db_path=db)
    assert database.latest_metadata("clustering", db_path=db) == {"sizes": [3, 4]}


def test_sql_injection_text_is_stored_as_plain_text(db):
    nasty = "'); DROP TABLE analyses; --"
    database.add_analysis("credibility", nasty, "false", db_path=db)
    rows = database.get_recent(db_path=db)               # the table must still exist
    assert rows[0]["excerpt"] == nasty
