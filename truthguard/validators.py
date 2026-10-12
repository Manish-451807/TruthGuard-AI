"""
truthguard/validators.py - checks every piece of user input BEFORE it reaches
a model or the database. Each function either returns clean data or raises
ValidationError with a message that is safe to show to the user.
"""
import config


class ValidationError(ValueError):
    """Raised when user input is missing, too short, too long or malformed."""


def validate_text(text, min_length=config.MIN_TEXT_LENGTH,
                  max_length=config.MAX_TEXT_LENGTH, field="Text"):
    """Return the cleaned text, or raise ValidationError."""
    if text is None or not isinstance(text, str):
        raise ValidationError(f"{field} is required.")
    # Drop control characters (including NUL bytes) but keep newlines and tabs.
    cleaned = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t").strip()
    if not cleaned:
        raise ValidationError(f"{field} cannot be empty.")
    if len(cleaned) < min_length:
        raise ValidationError(
            f"{field} is too short. Please enter at least {min_length} characters.")
    if len(cleaned) > max_length:
        raise ValidationError(
            f"{field} is too long ({len(cleaned)} characters). "
            f"The limit is {max_length} characters.")
    return cleaned


def validate_cluster_count(value):
    """Convert the form value for k into an int inside the allowed range."""
    try:
        k = int(value)
    except (TypeError, ValueError):
        raise ValidationError("The number of clusters must be a whole number.")
    if not config.MIN_CLUSTERS <= k <= config.MAX_CLUSTERS:
        raise ValidationError(
            f"Choose between {config.MIN_CLUSTERS} and {config.MAX_CLUSTERS} clusters.")
    return k


def parse_cluster_texts(raw, k):
    """Split a textarea into one text per non-empty line and validate the list."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValidationError("Please paste some texts, one per line.")
    lines = []
    for line in raw.splitlines():
        line = "".join(ch for ch in line if ch.isprintable()).strip()
        if line:
            if len(line) > config.MAX_TEXT_LENGTH:
                raise ValidationError(
                    f"One line is longer than {config.MAX_TEXT_LENGTH} characters.")
            lines.append(line)
    if len(lines) > config.MAX_CLUSTER_TEXTS:
        raise ValidationError(
            f"Too many texts ({len(lines)}). The limit is {config.MAX_CLUSTER_TEXTS}.")
    if len(lines) < k:
        raise ValidationError(
            f"You need at least as many texts as clusters ({k}); you gave {len(lines)}.")
    return lines
