from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
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
            res = requests.post("http://localhost:9696/predict", json=payload)
            if res.status_code == 200:
                result = res.json()
                prediction = result["label"]
                probability_fake = result["probability being fake"]
                probability_real = result["probability being real"]
            else:
                error = res.json().get("error", "Fehlerhafte Anfrage.")
        except Exception as e:
            error = str(e)

    return render_template("index.html",
                           prediction=prediction,
                           probability_fake=probability_fake,
                           probability_real=probability_real,
                           error=error)


if __name__ == "__main__":
    app.run(port=5000, debug=True)