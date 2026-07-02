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
    if request.method == "POST":
        try:
            N = float(request.form["nitrogen"])
            P = float(request.form["phosphorous"])
            K = float(request.form["potassium"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            scaled_data = scaler.transform(input_data)
            result = model.predict(scaled_data)[0]
            prediction = result.capitalize()
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("find_your_crop.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)