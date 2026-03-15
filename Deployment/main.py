from flask import Flask, render_template, request, redirect, url_for
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ensure this is before importing utils
import common.utils as utils

model = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Deployment/
PROJECT_DIR = os.path.dirname(BASE_DIR)                # project root

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_DIR, "Deployment", "templates"),
    static_folder=os.path.join(PROJECT_DIR, "Deployment", "static"),
)

@app.route("/")
def index():
    result = None
    if "sentiment" in request.args:
        result = {
            "sentiment": request.args["sentiment"],
            "confidence": float(request.args["confidence"]),
            "text": request.args.get("text", "")
        }
    return render_template("index.html", result=result)

@app.route("/analyse", methods=["POST"])
def analyse():
    text = request.form["text"]
    result = model.predict(text, enable_plt=False)
    return redirect(url_for("index",
        sentiment=result["sentiment"],
        confidence=round(float(result["confidence"]), 2),
        text=text
    ))
if __name__ == "__main__":
    model = utils.SentimentAnalyser(
        "Model_Development",
        sys.argv
    )
    app.run()