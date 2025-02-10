from flask import Flask, jsonify,render_template, request
from test import TextToNum
import pickle
app=Flask(__name__)
@app.route("/")
def Home():
    return render_template("index.html")
@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method == "POST":
        msg =request.form.get("message")
        print(msg)
        c1=TextToNum(msg)
        c1.cleaner()
        c1.token
        c1.removeStop()
        st=c1.stemme()
        stvc=" ".join(st);
        with open("vectorizer.pickle","rb") as vc_file:
            vectorizer=pickle.load(vc_file)
        dt=vectorizer.transform([stvc]).toarray()
        with open ("model.pickle","rb") as mb_file:
            model=pickle.load(mb_file)
        pred=model.predict(dt)
        print(pred)
        return jsonify({"prediction":pred})
    else:
        return render_template("predict.html")

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5050)
    