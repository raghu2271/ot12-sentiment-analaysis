from flask import Flask, render_template, request, jsonify
from test import TextToNum
import pickle

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        msg = request.form.get("message")
        print(f"Received message: {msg}")

        # Create instance of TextToNum class and process message
        cl = TextToNum(msg)
        cl.cleaner()  # Clean the text
        cl.token()  # Tokenize the text
        cl.removeStop()  # Remove stopwords
        st = cl.stemme()  # Stem words (ensure this method returns a list)

        # Join the stemmed words into a single string
        stvc = " ".join(st)

        # Load vectorizer
        with open("vectorizer.pickle", "rb") as vc_file:
            vectorizer = pickle.load(vc_file)

        # Transform the text input
        dt = vectorizer.transform([stvc]).toarray()

        # Load model
        with open("model.pickle", "rb") as mb_file:
            model = pickle.load(mb_file)

        # Make prediction
        pred = model.predict(dt)[0]
        prediction_label = "Positive" if pred == 1 else "Neutral" if pred == 0 else "Negative"

        # Return the prediction in an HTML template
        return render_template("result.html", prediction=prediction_label)

    return render_template("predict.html")

if __name__ == "__main__":  # Corrected the main condition
    app.run(host="0.0.0.0", port=5050, debug=True)
