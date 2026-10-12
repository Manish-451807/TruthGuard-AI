"""Tests for truthguard/validators.py (input validation)."""
import pytest

import config
from truthguard.validators import (ValidationError, parse_cluster_texts,
                                   validate_cluster_count, validate_text)


def test_valid_text_is_returned_stripped():
    assert validate_text("   A perfectly normal headline.  ") == "A perfectly normal headline."


@pytest.mark.parametrize("bad", [None, 123, ["list"]])
def test_non_string_is_rejected(bad):
    with pytest.raises(ValidationError):
        validate_text(bad)


@pytest.mark.parametrize("empty", ["", "   ", "\n\t  \n"])
def test_empty_or_blank_text_is_rejected(empty):
    with pytest.raises(ValidationError, match="empty"):
        validate_text(empty)


def test_too_short_text_is_rejected():
    with pytest.raises(ValidationError, match="too short"):
        validate_text("hi")


def test_text_at_the_limits_is_accepted():
    assert len(validate_text("x" * config.MIN_TEXT_LENGTH)) == config.MIN_TEXT_LENGTH
    assert len(validate_text("x" * config.MAX_TEXT_LENGTH)) == config.MAX_TEXT_LENGTH


def test_too_long_text_is_rejected():
    with pytest.raises(ValidationError, match="too long"):
        validate_text("x" * (config.MAX_TEXT_LENGTH + 1))


def test_control_characters_and_nul_bytes_are_removed():
    assert validate_text("Hello\x00 wor\x07ld again") == "Hello world again"


def test_newlines_are_kept():
    assert "\n" in validate_text("first line here\nsecond line here")


def test_custom_limits_and_field_name():
    with pytest.raises(ValidationError, match="Article"):
        validate_text("short", min_length=50, field="Article")


@pytest.mark.parametrize("value, expected", [("3", 3), (5, 5), (str(config.MIN_CLUSTERS), config.MIN_CLUSTERS)])
def test_valid_cluster_counts(value, expected):
    assert validate_cluster_count(value) == expected


@pytest.mark.parametrize("bad", ["abc", None, "", "2.5", "0", "1", str(config.MAX_CLUSTERS + 1), "-3"])
def test_invalid_cluster_counts(bad):
    with pytest.raises(ValidationError):
        validate_cluster_count(bad)


def test_parse_cluster_texts_splits_lines_and_ignores_blanks():
    assert parse_cluster_texts("one\n\n  two  \nthree\n", 2) == ["one", "two", "three"]


def test_parse_cluster_texts_rejects_empty_input():
    with pytest.raises(ValidationError):
        parse_cluster_texts("   \n  ", 2)


def test_parse_cluster_texts_needs_at_least_k_texts():
    with pytest.raises(ValidationError, match="at least"):
        parse_cluster_texts("only one line", 3)


def test_parse_cluster_texts_rejects_too_many_lines():
    many = "\n".join(f"line {i}" for i in range(config.MAX_CLUSTER_TEXTS + 1))
    with pytest.raises(ValidationError, match="Too many"):
        parse_cluster_texts(many, 2)


def test_parse_cluster_texts_rejects_a_too_long_line():
    with pytest.raises(ValidationError, match="longer"):
        parse_cluster_texts("ok\n" + "x" * (config.MAX_TEXT_LENGTH + 1), 2)
