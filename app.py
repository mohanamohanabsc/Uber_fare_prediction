from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)

#Load model
model=pickle.load(open("models/Final_model_uber_fare.sav","rb"))
model1=pickle.load(open("models/Final_model_uber_dist.sav", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    plo=float(request.form["plo"])
    pla=float(request.form["pla"])
    dlo=float(request.form["dlo"])
    dla=float(request.form["dla"])


    data=pd.DataFrame(np.array([[plo,pla,dlo,dla]]), columns=["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"])

    
    fare = model.predict(data)
    fare_prediction = round(float(fare[0]), 2)

    
    distance = model1.predict(data)
    distance_prediction = round(float(distance[0]), 2)

    distance_prediction_km = distance_prediction/ 1000
  
    return render_template("result.html",fare_prediction=fare_prediction,distance_prediction_km=distance_prediction_km)

   
   

if __name__ == "__main__":
    app.run(debug=True)