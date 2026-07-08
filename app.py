"""
OptiCrop - Flask Backend
Loads the trained model and scaler, and serves the web app.
"""

import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/findyourcrop", methods=["GET", "POST"])
def find_your_crop():
    prediction = None

    # Store input values so they remain after prediction
    values = {
        "nitrogen": "",
        "phosphorous": "",
        "potassium": "",
        "temperature": "",
        "humidity": "",
        "ph": "",
        "rainfall": ""
    }

    if request.method == "POST":
        try:
            values["nitrogen"] = request.form["nitrogen"]
            values["phosphorous"] = request.form["phosphorous"]
            values["potassium"] = request.form["potassium"]
            values["temperature"] = request.form["temperature"]
            values["humidity"] = request.form["humidity"]
            values["ph"] = request.form["ph"]
            values["rainfall"] = request.form["rainfall"]

            N = float(values["nitrogen"])
            P = float(values["phosphorous"])
            K = float(values["potassium"])
            temperature = float(values["temperature"])
            humidity = float(values["humidity"])
            ph = float(values["ph"])
            rainfall = float(values["rainfall"])

            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            scaled_data = scaler.transform(input_data)

            result = model.predict(scaled_data)[0]
            prediction = result.capitalize()

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template(
        "find_your_crop.html",
        prediction=prediction,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)