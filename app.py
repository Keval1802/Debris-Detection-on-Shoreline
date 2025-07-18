from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load updated model (trained without Shoreline_Mean)
model = joblib.load("debris_model.pkl")

def classify_risk(density):
    if density > 2.0:
        return "High Risk", "red"
    elif density >= 1.0:
        return "Medium Risk", "orange"
    else:
        return "Low Risk", "green"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        total_debris = float(request.form["total_debris"])
        estimated_area = float(request.form["estimated_area"])
        year = int(request.form["year"])
        month = int(request.form["month"])
        storm_category = int(request.form["storm_category"])

        features = [[
            total_debris,
            estimated_area,
            year,
            month,
            storm_category
        ]]

        predicted_density = model.predict(features)[0]
        predicted_density = round(predicted_density, 4)

        risk_label, risk_color = classify_risk(predicted_density)

        return render_template("index.html",
                               prediction=predicted_density,
                               risk=risk_label,
                               color=risk_color)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
