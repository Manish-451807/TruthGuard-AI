# TruthGuard AI

**AI-Powered News Credibility & Sentiment Analysis Platform**

TruthGuard AI is a student machine-learning project that demonstrates text classification, sentiment analysis, news-text clustering, model evaluation, a Flask web application, and optional LLM-based summarization.

> **Important limitation:** A text classifier learns patterns from its labeled training data. It does not independently verify facts and must not be treated as a definitive truth detector. Synthetic demo data is only for checking that the app runs; its predictions and metrics are not meaningful research results.

## Features

- News statement classification using TF-IDF and a trained Naive Bayes model.
- Sentiment classification using a separate model.
- K-Means clustering for groups of similar text.
- Optional Google Gemini summarization (the app still starts without an API key).
- SQLite analysis history with an opt-in checkbox on analysis forms.
- Dashboard and evaluation reports.
- Tests for validators, database operations, model handling, and Flask routes.

## Requirements

- Python 3.10, 3.11, or 3.12
- VS Code (recommended)
- Internet access to install Python packages; dataset and Gemini API access are optional for demo mode

## Windows PowerShell setup

Open PowerShell in the `TruthGuard-AI` project folder.

### 1. Check Python

```powershell
py --version
```

Use Python 3.10–3.12. If `py` is unavailable, try `python --version`.

### 2. Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

If you have Python 3.10 or 3.12 instead, change `-3.11` to your installed version.

### 3. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the current terminal session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Or use Command Prompt and run `.venv\Scripts\activate.bat`.

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Create your environment file

```powershell
Copy-Item .env.example .env
```

The app runs without a Gemini API key. To enable the optional summarizer, add your own `GEMINI_API_KEY` to `.env`. Never publish `.env` or commit it to GitHub.

### 6. Prepare demonstration data

The app can start in demonstration mode with the included synthetic sample. To prepare the sample data and train demo models, run:

```powershell
python scripts/prepare_dataset.py --use-sample
python scripts/train_credibility_model.py
python scripts/train_sentiment_model.py --use-sample
python scripts/evaluate_models.py
```

Demo data is synthetic and small. Do not report its model scores as meaningful performance or use them as evidence that the app can detect real misinformation.

### 7. Run the application

```powershell
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Stop the server with `Ctrl+C` in the terminal.

### 8. Run tests

```powershell
python -m pytest -q
```

If the tests fail, save the full terminal output. Do not ignore failures or assume that a test passed just because the server starts.

## macOS / Linux setup

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python scripts/prepare_dataset.py --use-sample
python scripts/train_credibility_model.py
python scripts/train_sentiment_model.py --use-sample
python scripts/evaluate_models.py
python app.py
```

Then visit http://127.0.0.1:5000. Run tests with `python -m pytest -q`.

## Use real datasets

### Credibility classification: LIAR

This project does not include the LIAR dataset. Read [`scripts/download_instructions.md`](scripts/download_instructions.md) for the source, file layout, labels, and placement instructions. Put the original `train.tsv`, `valid.tsv`, and `test.tsv` files in `data/raw/liar/`, then run:

```bash
python scripts/prepare_dataset.py
python scripts/train_credibility_model.py
python scripts/evaluate_models.py
```

The default setup retains the six original LIAR labels. A two-group mapping can be selected with `LABEL_MODE=grouped_two_class` in `.env`; that grouping is a project modeling choice, not an official LIAR label scheme.

### Sentiment classification: TweetEval

The LIAR data does not provide sentiment labels, so sentiment uses a separate dataset. Follow the TweetEval instructions in [`scripts/download_instructions.md`](scripts/download_instructions.md), place its six sentiment text/label files in `data/raw/tweeteval/sentiment/`, then run:

```bash
python scripts/train_sentiment_model.py
```

Check the dataset's current license and terms before using or redistributing it. Keep downloaded datasets out of public Git repositories unless their terms explicitly allow it.

## Demo mode vs trained mode

- **Demo mode:** Uses the small synthetic CSV included in this project. It allows you to explore routes and functionality, but predictions and metrics are not meaningful.
- **Trained mode:** Uses models trained on the real labeled datasets after you download and prepare them. Results still reflect dataset labels and model limitations; they are not proof that a new claim is true or false.
- **Summarizer:** Requires a valid Gemini API key and internet access. If unavailable, the UI should explain that the feature is not configured rather than pretending a summary was generated.

## Project structure

```text
TruthGuard-AI/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .env.example
├── data/
├── models/
├── scripts/
├── truthguard/
├── templates/
├── static/
└── tests/
```

## Troubleshooting

- **`ModuleNotFoundError`:** Activate `.venv`, then run `python -m pip install -r requirements.txt` again.
- **`python` / `py` not found:** Install Python 3.10–3.12 and enable the PATH option, or use the Python launcher available on your system.
- **PowerShell activation blocked:** Use the process-scoped execution policy command above, or activate from Command Prompt.
- **Port 5000 already in use:** Stop the other local server using that port before starting this app.
- **Model unavailable:** Run the appropriate preparation and training scripts; check that generated model files exist under `models/`.
- **Missing LIAR files:** Verify the three TSV files are in `data/raw/liar/`, or set `LIAR_DIR` in `.env` to the correct folder.
- **Missing TweetEval files:** Confirm the six expected text and label files are in `data/raw/tweeteval/sentiment/`.
- **Gemini summarizer unavailable:** Check that `GEMINI_API_KEY` is set correctly in `.env`, the optional SDK is installed, and network access is available. Restart Flask after changing `.env`.
- **CSRF/session error:** Reload the form page and submit again. For local work, check that cookies are enabled and that `.env` contains a stable `SECRET_KEY`.
- **Unexpected template or endpoint error:** Start the server from the project root with the virtual environment activated, then read the terminal traceback. Do not expose debug mode on a public server.
- **Tests fail:** Run `python -m pytest -q` inside the active virtual environment and use the first traceback to diagnose the issue.

## Privacy, security, and deployment notes

- The application is intended for local learning and demonstration. It has no user login or multi-user access control, so do not expose it publicly with sensitive data.
- The analysis form has an opt-in save-history checkbox. History stores an excerpt and associated result metadata, not an unlimited archive of every full input.
- Keep API keys in `.env`; `.env` is listed in `.gitignore`.
- Flask's development server is for local development only. Use a production WSGI server and review deployment security before public hosting.
- Treat external news text as untrusted input. Summaries can contain errors, omit context, or generate unsupported claims; verify important information with reliable primary sources.

## Academic / resume note

Describe only features you have personally run and verified. Record the real dataset version, train/test split, model parameters, and measured test metrics in your report. Never claim a performance number until it is produced by the evaluation script on the appropriate held-out data.
