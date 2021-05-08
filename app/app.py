from flask import Flask,render_template,url_for,request
from src.utils import preprocess_text
import string
import joblib
import os

app = Flask(__name__)

path ='src/weight' # path to weight

vectorizer = joblib.load(os.path.join(path,'vectorizer.pkl')) # load tfidfvectorizer

model = joblib.load(os.path.join(path,'model.pkl')) # get model

output_map = ['Hate Speech','Offensive Language','Neither Hateful nor offensive Language'] # map output

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        inp_text = request.form.get('text') # get text input

        value = inp_text

        text = preprocess_text(inp_text) # preprocess input text

        inp_text = inp_text.translate(str.maketrans('', '', string.punctuation)) # check if input is not only set of punctuation

        if len(inp_text.strip())==0 or inp_text is None: # check for empty string
            return render_template('base.html')

        text = vectorizer.transform([text]) # convert text to numbers

        prediction = model.predict(text) # make prediction

        prediction = output_map[prediction[0]] # map prediction

        prediction = 'Prediction: '+ prediction

    return render_template('base.html',output=prediction,message=value)

if __name__=='__main__':
    app.run(host='0.0.0.0')
