from flask import Flask, render_template, request
from test import TextToNum
import pickle

app = Flask(__name__)  # ✅ Corrected double underscores

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        msg = request.form.get("message")
        print(f"Message received: {msg}")

        # Clean and preprocess the message
        cl = TextToNum(msg)
        cl.cleaner()
        cl.token()
        cl.removeStop()
        st = cl.stemme()
        stvc = " ".join(st)

        # Load the vectorizer and model
        with open("vectorizer.pickle", "rb") as vc_file:
            vectorizer = pickle.load(vc_file)

        # Transform the cleaned message into vector format
        dt = vectorizer.transform([stvc]).toarray()

        with open("model.pickle", "rb") as md_file:
            model = pickle.load(md_file)

        # Get the prediction result from the model
        pred = model.predict(dt)
        print(f"Raw prediction: {pred}")  # Debugging the raw prediction

        # Map numeric predictions to sentiment labels
        if pred[0] == 1:
            pred = "Positive"
        elif pred[0] == 0:
            pred = "Neutral"
        else:
            pred = "Negative"
        print(pred) 

        return render_template("result.html", prediction=str(pred))
    else:
        return render_template("predict.html")



if __name__ == "__main__": 
    app.run(host="0.0.0.0", port=5050)
