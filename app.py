"""
app.py - the Flask web application (routes only; the logic lives in truthguard/).

Run with:  python app.py   then open http://127.0.0.1:5000
"""
from flask import (Flask, current_app, flash, jsonify, redirect,
                   render_template, request, url_for)
from flask_wtf.csrf import CSRFError, CSRFProtect

import config
from truthguard import analytics, clustering, credibility, database, sentiment, summarizer
from truthguard.modeling import ModelUnavailableError
from truthguard.validators import (ValidationError, parse_cluster_texts,
                                   validate_cluster_count, validate_text)


def create_app(test_config=None):
    """Application factory: lets tests build an app with a temporary database."""
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=config.SECRET_KEY,
        DATABASE_PATH=str(config.DATABASE_PATH),
        MAX_CONTENT_LENGTH=1_000_000,          # reject request bodies over ~1 MB
    )
    if test_config:
        app.config.update(test_config)

    CSRFProtect(app)                           # every POST form needs a csrf_token
    database.init_db(app.config["DATABASE_PATH"])

    # ------------------------------------------------------------ helpers
    def db_path():
        return current_app.config["DATABASE_PATH"]

    def status():
        return {"credibility": credibility.status(),
                "sentiment": sentiment.status(),
                "summarizer": summarizer.status()}

    def wants_to_save():
        # The form checkbox "save_history". If unticked, NOTHING is stored.
        return request.form.get("save_history") == "on"

    def error_page(code, title, message):
        return render_template("404.html", code=code, title=title,
                               message=message), code

    # ------------------------------------------------------------ pages
    @app.route("/")
    def index():
        return render_template("index.html", status=status())

    def text_tool(template, analysis_type, analyze_fn):
        """Shared logic for the credibility and sentiment pages."""
        ctx = {"form_text": "", "result": None, "error": None,
               "max_length": config.MAX_TEXT_LENGTH, "status": status()}
        code = 200
        if request.method == "POST":
            raw = request.form.get("text", "")
            ctx["form_text"] = raw[:config.MAX_TEXT_LENGTH + 1]
            try:
                text = validate_text(raw)
                result = analyze_fn(text)
                result["saved"] = False
                if wants_to_save():
                    database.add_analysis(
                        analysis_type, text, result["label"], result["confidence"],
                        {"mode": result["mode"], "model": result["model_name"]},
                        db_path())
                    result["saved"] = True
                ctx["result"] = result
            except ValidationError as exc:
                ctx["error"], code = str(exc), 400
            except ModelUnavailableError as exc:
                ctx["error"], code = str(exc), 503
        return render_template(template, **ctx), code

    @app.route("/analyzer", methods=["GET", "POST"])
    def analyzer():
        return text_tool("analyzer.html", "credibility", credibility.analyze)

    @app.route("/sentiment", methods=["GET", "POST"])
    def sentiment_page():
        return text_tool("sentiment.html", "sentiment", sentiment.analyze)

    @app.route("/clustering", methods=["GET", "POST"])
    def clustering_page():
        ctx = {"form_text": "", "k": 3, "result": None, "error": None,
               "min_k": config.MIN_CLUSTERS, "max_k": config.MAX_CLUSTERS,
               "max_texts": config.MAX_CLUSTER_TEXTS}
        code = 200
        if request.method == "POST":
            raw = request.form.get("texts", "")
            ctx["form_text"] = raw[:200000]
            try:
                k = validate_cluster_count(request.form.get("k"))
                ctx["k"] = k
                texts = parse_cluster_texts(raw, k)
                result = clustering.cluster_texts(texts, k)
                result["saved"] = False
                if wants_to_save():
                    database.add_analysis(
                        "clustering", texts[0], f"{k} clusters", None,
                        {"k": k, "n_texts": len(texts),
                         "sizes": [c["size"] for c in result["clusters"]]}, db_path())
                    result["saved"] = True
                ctx["result"] = result
            except (ValidationError, clustering.ClusteringError) as exc:
                ctx["error"], code = str(exc), 400
        return render_template("clustering.html", **ctx), code

    @app.route("/summarizer", methods=["GET", "POST"])
    def summarizer_page():
        info = summarizer.status()
        ctx = {"form_text": "", "result": None, "error": None,
               "available": info["available"], "unavailable_message": info["message"],
               "max_length": config.MAX_ARTICLE_LENGTH}
        code = 200
        if request.method == "POST":
            raw = request.form.get("text", "")
            ctx["form_text"] = raw[:config.MAX_ARTICLE_LENGTH + 1]
            try:
                article = validate_text(raw, min_length=50,
                                        max_length=config.MAX_ARTICLE_LENGTH,
                                        field="Article")
                result = summarizer.summarize(article)
                result["saved"] = False
                if wants_to_save():
                    database.add_analysis(
                        "summary", article, "summary", None,
                        {"article_chars": result["article_chars"],
                         "summary_chars": result["summary_chars"],
                         "model": result["model"]}, db_path())
                    result["saved"] = True
                ctx["result"] = result
            except ValidationError as exc:
                ctx["error"], code = str(exc), 400
            except summarizer.SummarizerUnavailable as exc:
                ctx["error"], code = str(exc), 503
            except summarizer.SummarizerError as exc:
                ctx["error"], code = str(exc), 502
        return render_template("summarizer.html", **ctx), code

    @app.route("/dashboard")
    def dashboard():
        return render_template("dashboard.html", data=analytics.get_dashboard_data(db_path()))

    @app.route("/api/dashboard-data")
    def dashboard_data():
        """JSON used by charts.js so charts always show real numbers."""
        return jsonify(analytics.get_dashboard_data(db_path()))

    # ------------------------------------------------------------ history
    @app.route("/history")
    def history():
        entries = database.get_recent(limit=100, db_path=db_path())
        return render_template("history.html", entries=entries, total=len(entries))

    @app.route("/history/<int:entry_id>/delete", methods=["POST"])
    def delete_entry(entry_id):
        removed = database.delete_analysis(entry_id, db_path())
        flash("Entry deleted." if removed else "That entry no longer exists.",
              "success" if removed else "error")
        return redirect(url_for("history"))

    @app.route("/history/clear", methods=["POST"])
    def clear_entries():
        count = database.clear_history(db_path())
        flash(f"History cleared ({count} entries removed).", "success")
        return redirect(url_for("history"))

    # ------------------------------------------------------------ errors
    @app.errorhandler(404)
    def not_found(_):
        return error_page(404, "Page not found", "That page does not exist.")

    @app.errorhandler(413)
    def too_large(_):
        return error_page(413, "Request too large", "The submitted data is too large.")

    @app.errorhandler(CSRFError)
    def csrf_failed(_):
        return error_page(400, "Session expired",
                          "Your form session expired. Go back, reload the page and try again.")

    @app.errorhandler(500)
    def server_error(_):
        # The real traceback goes to the server log, never to the visitor.
        return error_page(500, "Something went wrong", "An unexpected error occurred.")

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    return app


app = create_app()

if __name__ == "__main__":
    # debug=True only if FLASK_DEBUG=1 in .env (never use that in production).
    app.run(host="127.0.0.1", port=5000, debug=config.DEBUG)
