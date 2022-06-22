from flask import Flask,render_template,request,jsonify
import pickle as pkl
import os

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello World"

@app.route('/test')
def test():
    return render_template("index.html")


@app.route('/predict',methods=["POST"])
def predict():
  
    location = 'D:\\\EmailClassification'
    fullpath = os.path.join(location, 'model.pkl')
    fullpath1 = os.path.join(location, 'count_vectorizer.pkl')
    features = [x for x in request.form.values()]
    #classifier = pkl.load(open('model.pkl','rb'))
    classifier = pkl.load(open(fullpath,'rb'))
    #cv = pkl.load(open('count_vectorizer.pkl','rb'))
    cv = pkl.load(open(fullpath1,'rb'))
    ans = classifier.predict(cv.transform(features))
    print(ans)
    if ans[0] == 0:
        return render_template("index.html",answer = str("Not a Spam"))
    else:
        return render_template("index.html",answer = str(" Spam"))

if __name__ == '__main__':
    app.run(debug=True)