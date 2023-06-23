from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle


# Create app
app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
   return render_template("index.html")

@app.route('/home')
def homes():
   return render_template("index.html")
    

@app.route("/predict", methods = ["POST"])
def predict():
   intfeatures = [float(x) for x in request.form.values()]
   features = [np.array(intfeatures)]
   prediction = model.predict(features)
   
   if(prediction):
      return render_template("positive.html")
   return render_template("negative.html")



if __name__ == '__main__':
   app.run(debug = True)