from flask import Flask, request, render_template, url_for
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and preprocessing tools
model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Ensure static/images directory exists
os.makedirs(os.path.join(app.static_folder, 'images'), exist_ok=True)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Scale features and make prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        scaled = scaler.transform(features)
        pred = model.predict(scaled)
        species = le.inverse_transform(pred)[0]

        # Determine image path
        species_lower = species.lower()
        image_path = f"images/{species_lower}.jpg"

        # Check if image exists
        if not os.path.exists(os.path.join(app.static_folder, image_path)):
            image_path = "images/default.jpg"  # Fallback image

        image_url = url_for('static', filename=image_path)

        return render_template("result.html",
                               species=species,
                               image_url=image_url,
                               sepal_length=sepal_length,
                               sepal_width=sepal_width,
                               petal_length=petal_length,
                               petal_width=petal_width)
    except Exception as e:
        return render_template("error.html", error=str(e))


if __name__ == '__main__':
    app.run(debug=True)