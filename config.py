"""
config.py - ONE place for every setting in TruthGuard AI.

Other files do `import config` and read values from here, so you never have to
edit paths or limits in several files.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Folder that contains this file = project root.
BASE_DIR = Path(__file__).resolve().parent

# Read the .env file (if it exists) into environment variables.
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------- Paths
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_PATH = DATA_DIR / "sample_news.csv"          # demo-only, synthetic
RAW_DATA_DIR = Path(os.getenv("LIAR_DIR", DATA_DIR / "raw" / "liar"))
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_PATH = BASE_DIR / "truthguard.db"

# ---------------------------------------------------------------- Flask
# A missing SECRET_KEY gets a random one per start-up: fine for local use,
# but sessions reset on every restart. Set a fixed value in .env.
SECRET_KEY = os.getenv("SECRET_KEY") or os.urandom(32).hex()
DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

# ---------------------------------------------------------------- Input limits
MIN_TEXT_LENGTH = 10          # characters
MAX_TEXT_LENGTH = 5000        # characters for analyzer / sentiment
MAX_ARTICLE_LENGTH = 20000    # characters for the summarizer
MAX_CLUSTER_TEXTS = 500       # maximum number of texts for clustering
MIN_CLUSTERS, MAX_CLUSTERS = 2, 10
SAVED_EXCERPT_LENGTH = 200    # history stores only this many characters

# ---------------------------------------------------------------- ML
RANDOM_SEED = 42
TEST_SIZE = 0.25              # used only for the small sample dataset

# LIAR dataset labels, ordered from least to most truthful (as rated by
# PolitiFact fact-checkers).
LIAR_LABELS = ["pants-fire", "false", "barely-true",
               "half-true", "mostly-true", "true"]

# Optional coarse grouping. This is a MODELLING CHOICE made by us, not an
# official LIAR category. It is only used when LABEL_MODE=grouped_two_class.
GROUPED_TWO_CLASS = {
    "pants-fire": "less_credible_group",
    "false": "less_credible_group",
    "barely-true": "less_credible_group",
    "half-true": "more_credible_group",
    "mostly-true": "more_credible_group",
    "true": "more_credible_group",
}
LABEL_MODE = os.getenv("LABEL_MODE", "six_class")
if LABEL_MODE not in ("six_class", "grouped_two_class"):
    raise ValueError("LABEL_MODE must be 'six_class' or 'grouped_two_class'")

# ---------------------------------------------------------------- Gemini (optional)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
