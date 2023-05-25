from flask import Flask,render_template,request
import numpy as np
import pickle

tree_classifier=pickle.load(open("tree_classifier.pkl","rb"))
stream_encoder=pickle.load(open("stream_encoder.pkl","rb"))
gender_encoder=pickle.load(open("gender_encoder.pkl","rb"))
sc_x=pickle.load(open("StandardScaler.pkl","rb"))

app=Flask(__name__)


def prediction(lst):
    lst=sc_x.transform(lst)
    return tree_classifier.predict(lst)[0]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def predict():
    if request.method=="POST":
        age=int(request.form["age"])
        gender=gender_encoder.transform([request.form["gender"]])
        internship=int(request.form["internship"])
        backlog=int(request.form["backlog"])
        stream=stream_encoder.transform([request.form["stream"]])
        cgpa=int(request.form["CGPA"])
        hostel=int(request.form["hostel"])

        pred=prediction(np.asarray([[age,gender,stream,internship,cgpa,hostel,backlog]],dtype=object))

    if pred==0:
        msg="Sorry!! You have less chance to be placed in on-campus placement"

    else:
        msg="Congratulation!! You have high chance to be placed in on-campus placement"

    return render_template("index.html",prediction=pred,st=msg)


if __name__=="__main__":
    app.run(debug=True)