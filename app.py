from flask import Flask, render_template, request, jsonify
from test import TextToNum
import pickle

app = Flask(__name__)  # Fixed _name_ to __name__

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        msg = request.form.get("message")  # Use .get() to avoid errors if key is missing
        if not msg:
            return jsonify({"error": "No message provided"}), 400  # Return an error if input is missing
        
        print("Received message:", msg)

        # Process text using TextToNum class
        cl = TextToNum(msg)
        cl.cleaner()  
        cl.token()  
        cl.removeStop()  
        st = cl.stemme()  

        # Join processed words into a string
        stvc = " ".join(st)

        try:
            # Load vectorizer
            with open("vectorizer.pickle", "rb") as vc_file:
                vectorizer = pickle.load(vc_file)

            dt = vectorizer.transform([stvc]).toarray()  # Transform text

            # Load model
            with open("model.pickle", "rb") as mb_file:
                model = pickle.load(mb_file)

            # Make prediction
            pred = model.predict(dt)
            print("Prediction:", pred)

            return jsonify({"prediction": str(pred[0])})
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500  # Handle errors properly
    
    return render_template("predict.html")

if __name__ == "__main__":  # Fixed condition
    app.run(host="0.0.0.0", port=5050)
