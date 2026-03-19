from flask import Flask, render_template, request, redirect, url_for, session, abort
import os
import sys
import uuid
import secrets
import dotenv
dotenv.load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common.utils as utils

model = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

_secret_key = os.environ.get("SECRET_KEY")
if not _secret_key:
    raise RuntimeError("SECRET_KEY environment variable is not set")

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_DIR, "Deployment", "templates"),
    static_folder=os.path.join(PROJECT_DIR, "Deployment", "static"),
)
app.secret_key = _secret_key
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

_result_cache = {}
_RESULT_CACHE_MAX = 500
_TEXT_MAX_LENGTH = 2000


@app.after_request
def set_security_headers(response):
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self' https://cdn.jsdelivr.net;"
    )
    return response


def _generate_csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(32)
    return session["csrf_token"]


def _validate_csrf():
    token = session.get("csrf_token")
    form_token = request.form.get("csrf_token", "")
    if not token or not secrets.compare_digest(token, form_token):
        abort(403)


@app.route("/")
def index():
    result = None
    result_id = session.pop("result_id", None)
    if result_id:
        result = _result_cache.pop(result_id, None)
    csrf_token = _generate_csrf_token()
    return render_template(
        "index.html",
        result=result,
        csrf_token=csrf_token,
        error_message=None,
        input_text="",
        text_max_length=_TEXT_MAX_LENGTH,
    )


@app.route("/analyse", methods=["POST"])
def analyse():
    _validate_csrf()

    raw_text = request.form.get("text", "")
    text = raw_text.strip()
    if not text:
        csrf_token = _generate_csrf_token()
        return render_template(
            "index.html",
            result=None,
            csrf_token=csrf_token,
            error_message="Please enter some text before submitting.",
            input_text=raw_text,
            text_max_length=_TEXT_MAX_LENGTH,
        ), 400

    if len(raw_text) > _TEXT_MAX_LENGTH:
        csrf_token = _generate_csrf_token()
        return render_template(
            "index.html",
            result=None,
            csrf_token=csrf_token,
            error_message=(
                f"Input is too long ({len(raw_text)} characters). "
                f"Maximum allowed is {_TEXT_MAX_LENGTH}."
            ),
            input_text=raw_text[:_TEXT_MAX_LENGTH],
            text_max_length=_TEXT_MAX_LENGTH,
        ), 400

    result = model.predict(text, enable_plt=False)
    result["text"] = text
    result["confidence"] = round(float(result["confidence"]), 2)

    if len(_result_cache) >= _RESULT_CACHE_MAX:
        _result_cache.pop(next(iter(_result_cache)))

    result_id = str(uuid.uuid4())
    _result_cache[result_id] = result
    session["result_id"] = result_id

    return redirect(url_for("index"))


if __name__ == "__main__":
    model = utils.SentimentAnalyser("Model_Development", sys.argv)
    app.run()