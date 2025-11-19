from flask import Flask, render_template, request
import numpy as np
from joblib import load


app = Flask(__name__)


model = load("model.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
       
            size = float(request.form["SIZE"])
            distance = float(request.form["DISTANCE"])
            desibel = float(request.form["DESIBEL"])
            airflow = float(request.form["AIRFLOW"])
            frequency = float(request.form["FREQUENCY"])

            fuel_kerosene = int(request.form["FUEL_kerosene"])
            fuel_lpg = int(request.form["FUEL_lpg"])
            fuel_thinner = int(request.form["FUEL_thinner"])

          
            features = np.array([[ 
                size,
                distance,
                desibel,
                airflow,
                frequency,
                fuel_kerosene,
                fuel_lpg,
                fuel_thinner
            ]])

   
            y_pred = model.predict(features)[0]

    
            prediction = int(y_pred)

        except Exception as e:
            print("Error during prediction:", e)
            prediction = None

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
