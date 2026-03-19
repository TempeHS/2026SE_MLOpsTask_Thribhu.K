from flask import Flask, render_template, request, redirect, url_for, session, abort
import os
import sys
import uuid
import secrets
import dotenv
import re as _re
from werkzeug.security import generate_password_hash, check_password_hash
from db import UserDatabase, CacheDatabase
# required to make uv happy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common.utils as utils
import logger

dotenv.load_dotenv()

_secret_key = os.environ.get("SECRET_KEY")
if not _secret_key:
    raise RuntimeError("SECRET_KEY environment variable is not set")

_TEXT_MAX_LENGTH = int(os.environ.get("TEXT_MAX_LENGTH", 2000))
_ANON_FREE_TRIES = int(os.environ.get("ANON_FREE_TRIES", 3))
_RESULT_CACHE_MAX = int(os.environ.get("RESULT_CACHE_MAX", 500))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

_DB_PATH = os.path.join(BASE_DIR, "app.db")

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_DIR, "Deployment", "templates"),
    static_folder=os.path.join(PROJECT_DIR, "Deployment", "static"),
)
app.secret_key = _secret_key
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

model = None

user_db = UserDatabase(_DB_PATH)
cache_db = CacheDatabase(_DB_PATH)

# this cache used as a form of handoff (temp storage for that request)
_result_cache = {}

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

def _generate_csrf_token() -> str:
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(32)
    return session["csrf_token"]


def _validate_csrf() -> None:
    token = session.get("csrf_token")
    form_token = request.form.get("csrf_token", "")
    if not token or not secrets.compare_digest(token, form_token):
        abort(403)

def _is_authenticated() -> bool:
    """Return True when a logged-in user_id is stored in the session."""
    return bool(session.get("user_id"))


def _anon_tries_used() -> int:
    return int(session.get("anon_tries_used", 0))


def _anon_tries_remaining() -> int:
    return max(0, _ANON_FREE_TRIES - _anon_tries_used())

def _render_index(result=None, error_message=None, input_text="", status=200,
                    signin_error=None, signup_error=None, open_overlay_tab=None):
    csrf_token = _generate_csrf_token()
    return render_template(
        "index.html",
        result=result,
        csrf_token=csrf_token,
        error_message=error_message,
        input_text=input_text,
        text_max_length=_TEXT_MAX_LENGTH,
        free_tries_remaining=_anon_tries_remaining(),
        free_tries_limit=_ANON_FREE_TRIES,
        is_authenticated=_is_authenticated(),
        username=session.get("username", ""),
        signin_error=signin_error,
        signup_error=signup_error,
        open_overlay_tab=open_overlay_tab,
        console_logs=logger.flush(),
    ), status

@app.route("/")
def index():
    result = None
    result_id = session.pop("result_id", None)
    if result_id:
        result = _result_cache.pop(result_id, None)
    return _render_index(result=result)


@app.route("/analyse", methods=["POST"])
def analyse():
    _validate_csrf()

    raw_text = request.form.get("text", "")
    text = raw_text.strip()

    if not text:
        return _render_index(
            error_message="Please enter some text before submitting.",
            input_text=raw_text,
            status=400,
        )

    if len(raw_text) > _TEXT_MAX_LENGTH:
        return _render_index(
            error_message=(
                f"Input is too long ({len(raw_text)} characters). "
                f"Maximum allowed is {_TEXT_MAX_LENGTH}."
            ),
            input_text=raw_text[:_TEXT_MAX_LENGTH],
            status=400,
        )

    if not _is_authenticated() and _anon_tries_remaining() <= 0:
        return _render_index(
            error_message=(
                f"You have used all {_ANON_FREE_TRIES} free analyses. "
                "Please create an account to continue."
            ),
            input_text=raw_text,
            status=403,
        )

    request_key = {"text": text}
    cached_result, cache_hit, request_hash = cache_db.get_or_push(request_key)

    if cache_hit:
        result = cached_result
        logger.log("cache hit", request_hash[:8])
    else:
        logger.log("not found in cache, adding")
        result = model.predict(text, enable_plt=False)
        result["text"] = text
        result["confidence"] = round(float(result["confidence"]), 2)
        result["request_hash"] = request_hash
        cache_db.get_or_push(request_key, result)

    if not _is_authenticated():
        session["anon_tries_used"] = _anon_tries_used() + 1

    if len(_result_cache) >= _RESULT_CACHE_MAX:
        _result_cache.pop(next(iter(_result_cache)))

    result_id = str(uuid.uuid4())
    _result_cache[result_id] = result
    session["result_id"] = result_id

    return redirect(url_for("index"))

@app.route("/feedback", methods=["POST"])
def feedback():
    _validate_csrf()
    if not _is_authenticated():
        return {"error": "Login required to submit feedback."}, 401

    request_hash = request.form.get("request_hash", "").strip()
    is_correct_str = request.form.get("is_correct", "")

    if not request_hash or is_correct_str not in ("true", "false"):
        return {"error": "Invalid input."}, 400

    cache_db.upsert_feedback(request_hash, session["user_id"], is_correct_str == "true")
    return {"ok": True}, 200

# authentication stuff after here =====================================================================

@app.route("/login", methods=["POST"])
def login():
    _validate_csrf()
    identity = request.form.get("identity", "").strip()
    password = request.form.get("password", "")

    if not identity or not password:
        return _render_index(
            signin_error="Please fill in all fields.",
            open_overlay_tab="signin", status=400,
        )

    user = user_db.get_user_by_identity(identity)
    if user is None or not check_password_hash(user["password_hash"], password):
        return _render_index(
            signin_error="Invalid username/email or password.",
            open_overlay_tab="signin", status=401,
        )

    if not int(user.get("is_active", 1)):
        return _render_index(
            signin_error="This account has been deactivated.",
            open_overlay_tab="signin", status=403,
        )

    session["user_id"] = user["user_id"]
    session["username"] = user["username"]
    session.pop("anon_tries_used", None)
    return redirect(url_for("index"))


@app.route("/register", methods=["POST"])
def register():
    _validate_csrf()
    username = request.form.get("username", "").strip()
    email    = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    if not username or not email or not password:
        return _render_index(
            signup_error="Please fill in all fields.",
            open_overlay_tab="signup", status=400,
        )
    if not _re.fullmatch(r"[A-Za-z0-9_\-]{3,32}", username):
        return _render_index(
            signup_error="Username must be 3–32 characters: letters, numbers, _ or -.",
            open_overlay_tab="signup", status=400,
        )
    if len(password) < 8:
        return _render_index(
            signup_error="Password must be at least 8 characters.",
            open_overlay_tab="signup", status=400,
        )
    if user_db.get_user_by_username(username):
        return _render_index(
            signup_error="That username is already taken.",
            open_overlay_tab="signup", status=409,
        )
    if user_db.get_user_by_email(email):
        return _render_index(
            signup_error="An account with that email already exists.",
            open_overlay_tab="signup", status=409,
        )

    password_hash = generate_password_hash(password)
    user = user_db.create_user(username, email, password_hash)
    session["user_id"] = user["user_id"]
    session["username"] = user["username"]
    session.pop("anon_tries_used", None)
    return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    model = utils.SentimentAnalyser("Model_Development", sys.argv)
    app.run()