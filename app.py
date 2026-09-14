from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd 
from src.datascienceproject.pipeline.prediction_pipeline import PredictionPipeline
import subprocess, sys
import traceback


app=Flask(__name__)

@app.route('/', methods=["GET"])
def homepage():
    return render_template("index.html")

@app.route('/train', methods=["GET"])
def training():
    # os.system("python main.py")
    result = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
    if result.returncode != 0:
        return f"Training failed:\n{result.stderr}", 500
    return "Training successful"


@app.route("/predict", methods=["POST"])
def predict():
    if request.method =="POST":
        try:
            fixed_acidity = float(request.form["fixed_acidity"])
            volatile_acidity = float(request.form["volatile_acidity"])
            citric_acid = float(request.form["citric_acid"])
            residual_sugar = float(request.form["residual_sugar"])
            chlorides = float(request.form["chlorides"])
            free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
            total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
            density = float(request.form["density"])
            ph = float(request.form["ph"])
            sulphates = float(request.form["sulphates"])
            alcohol = float(request.form["alcohol"])
            data=[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                  chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                  ph, sulphates, alcohol]
            data=np.array(data).reshape(1,11)

            obj=PredictionPipeline()
            prediction = obj.predict(data)
            prediction = float(prediction[0])
            return render_template("index.html", prediction=round(prediction, 3))
        except Exception as e:
            traceback.print_exc()
            return f"Prediction failed: {e}", 500


    else:
        return render_template("index.html")
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)