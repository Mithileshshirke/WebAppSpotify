import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('dp.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    ans=''

    output = round(prediction[0], 2)
    if output==1:
        ans='Yes'
    else:
        ans='No'

    return render_template('index.html', prediction_text='The spotify song will be a hit or no? - {}'.format(ans))


if __name__ == "__main__":
    app.run(debug=True)