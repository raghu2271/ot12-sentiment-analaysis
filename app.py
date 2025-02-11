from flask import Flask, render_template, request, jsonify
from test import TextToNum
import pickle

app = Flask(__name__)  # Corrected the Flask initialization

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        msg = request.form.get("message")
        print(msg)
        
        # Create instance of TextToNum class and process message
        cl = TextToNum(msg)
        cl.cleaner()  # Assuming this cleans the text
        cl.token()  # Assuming this tokenizes the text
        cl.removeStop()  # Ensure the method is called
        st = cl.stemme()  # Call the stemming method (make sure it returns stemmed text)
        
        # Join the stemmed words into a single string
        stvc = " ".join(st)
        
        # Load vectorizer and model to make prediction
        with open("vectorizer.pickle", "rb") as vc_file:
            vectorizer = pickle.load(vc_file)
        
        dt = vectorizer.transform([stvc]).toarray()  # Transform the stemmed text
        
        with open("model.pickle", "rb") as mb_file:
            model = pickle.load(mb_file)
        
        # Make prediction
        pred = model.predict(dt)
        print(pred)
        
        # Return the prediction as JSON
        return jsonify({"prediction": str(pred[0])})
    
    return render_template("predict.html")

if __name__ == "__main__":  # Corrected the main condition
    app.run(host="0.0.0.0", port=5050)
