from statistics import variance
from flask import Flask,render_template,request,url_for,request
import joblib

app = Flask(__name__)

rfc = joblib.load('rfc_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    #print(request.form.get('skewness'))
    skewness = int(request.form['skewness'])
    curtosis = int(request.form['curtosis'])
    entropy = int(request.form['entropy'])
    variance = int(request.form['variance'])

    result = rfc.predict([[variance, skewness, curtosis, entropy]])

    if result==[0]:
        return "fake"
    return "real"

if __name__=='__main__':
    app.run()