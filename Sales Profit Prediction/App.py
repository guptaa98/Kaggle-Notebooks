import numpy as np
import pandas as pd
import os
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
MODEL_PATH = 'salesprofit_predictionmodel.pkl'
model = pickle.load(open(MODEL_PATH, 'rb'))
df = pd.read_csv("Sales_project_phaseII.csv")


from sklearn import preprocessing   
label_encoder = preprocessing.LabelEncoder()
df['Country']= label_encoder.fit_transform(df['Country'])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
df[['Region','Country','Item Type','Sales Channel','Order Priority','Total Revenue','Total Cost','Units Sold','Unit Price','Unit Cost','Order Date_year','Order Date_month','Order Date_day','Ship Date_year','Ship Date_month','Ship Date_day']] = scaler.fit_transform(df[['Region','Country','Item Type','Sales Channel','Order Priority','Total Revenue','Total Cost','Units Sold','Unit Price','Unit Cost','Order Date_year','Order Date_month','Order Date_day','Ship Date_year','Ship Date_month','Ship Date_day']])
 #= scaler.fit_transform(df[['Units Sold','Unit Price','Unit Cost','Order Date_year','Order Date_month','Order Date_day','Ship Date_year','Ship Date_month','Ship Date_day']])


@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        print('inside')

        # int_features = [int(x) for x in request.form.values()]
        features = [x for x in request.form.values()]
        # print(features)
        if(features[4] == 'H'):
            features[4] = 3
        else:
            if(features[4] == 'M'):
                features[4] = 2
            else:
                features[4] = 1
         
        features[1] = label_encoder.transform([features[1]])
        # print(features)
        # for i in range(len(features)):
        #     if(i==1):
        #         continue
        #     else:
        #         features[i] = scaler.transform([features[i]])
        features = scaler.transform([features])


        print(features)
        final_features = [np.array(features[0])]
        prediction = model.predict(final_features)
        
        output = round(prediction[0], 2)
   
        #my_prediction = model.predict(data_pad)
        print (output)
        return render_template('index1.html', prediction_text=' {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)