import numpy as np
import os
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
MODEL_PATH = 'salesprofit_predictionmodel.pkl'
model = pickle.load(open(MODEL_PATH, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        print('inside')

        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        
        output = round(prediction[0], 2)
   
        #my_prediction = model.predict(data_pad)
        print (output)
        return render_template('index.html', prediction_text='Profit earned by the company is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)