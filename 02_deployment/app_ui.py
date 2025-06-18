"""
Simple user interface built with Flask,
running on a development server.
"""

from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Renders the main user interface for the Fake News Detection Web Service.

    Handles both GET and POST requests:
    - On GET: displays the input form.
    - On POST: receives the title and text of a news article, sends a request to the
      prediction API endpoint, and displays the predicted label along with the associated
      probabilities.

    Returns:
        Rendered HTML template with prediction results or an error message.
    """
    prediction = None
    probability_fake = None
    probability_real = None
    error = None

    if request.method == "POST":
        title = request.form.get("title")
        text = request.form.get("text")

        # Formatieren für predict_endpoint
        payload = {
            "title": title,
            "text": text
        }

        try:
            res = requests.post("http://fake-news-env.eba-cryzmisk.eu-west-1.elasticbeanstalk.com/predict", json=payload, timeout=10)
            if res.status_code == 200:
                result = res.json()
                prediction = result["label"]
                probability_fake = result["probability being fake"]
                probability_real = result["probability being real"]
            else:
                error = res.json().get("error", "Incorrect request.")
        except Exception as e:
            error = str(e)

    return render_template("index.html",
                           prediction=prediction,
                           probability_fake=probability_fake,
                           probability_real=probability_real,
                           error=error)


if __name__ == "__main__":
    app.run(port=5001, debug=True)
