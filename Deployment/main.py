from flask import Flask, render_template, request, redirect, url_for, session
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common.utils as utils

model = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_DIR, "Deployment", "templates"),
    static_folder=os.path.join(PROJECT_DIR, "Deployment", "static"),
)
app.secret_key = "sentiment_secret_key"

# server-side store — no size limit
_result_cache = {}


@app.route("/")
def index():
    result = None
    result_id = session.pop("result_id", None)
    if result_id:
        result = _result_cache.pop(result_id, None)
    return render_template("index.html", result=result)


@app.route("/analyse", methods=["POST"])
def analyse():
    text = request.form["text"]
    result = model.predict(text, enable_plt=False)
    result["text"] = text
    result["confidence"] = round(float(result["confidence"]), 2)

    # store result server-side, only put the key in the cookie
    result_id = str(uuid.uuid4())
    _result_cache[result_id] = result
    session["result_id"] = result_id

    return redirect(url_for("index"))


if __name__ == "__main__":
    model = utils.SentimentAnalyser("Model_Development", sys.argv)
    app.run()
