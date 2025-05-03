
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/covid_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[key]) for key in request.form]
        prediction = model.predict([features])[0]
        result = "Positive" if prediction == 1 else "Negative"
        return f"<h3>Prediction Result: {result}</h3>"
    except Exception as e:
        return f"<h3>Error: {e}</h3>"

if __name__ == "__main__":
    app.run(debug=True)
