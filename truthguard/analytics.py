"""
truthguard/analytics.py - builds the dashboard data.

LIVE numbers come from the SQLite history (what users actually ran).
OFFLINE numbers come from the JSON reports written by the training scripts.
They are kept in separate sections and never mixed or invented.
"""
import json

import config
from truthguard import credibility, database, sentiment


def load_metrics(filename):
    """Read a metrics report from models/, or return None if absent/invalid."""
    path = config.MODELS_DIR / filename
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, ValueError, OSError):
        return None
    return data if isinstance(data, dict) else None


def get_dashboard_data(db_path=None):
    counts = database.count_by_type(db_path)
    return {
        "live": {
            "total": sum(counts.values()),
            "counts": counts,
            "credibility_labels": database.label_distribution("credibility", db_path),
            "sentiment_labels": database.label_distribution("sentiment", db_path),
            "latest_clustering": database.latest_metadata("clustering", db_path),
            "recent": database.get_recent(limit=8, db_path=db_path),
        },
        "offline": {
            "credibility": load_metrics("credibility_metrics.json"),
            "sentiment": load_metrics("sentiment_metrics.json"),
        },
        "model_status": {
            "credibility": credibility.status(),
            "sentiment": sentiment.status(),
        },
    }
