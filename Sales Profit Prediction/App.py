import numpy as np
import pandas as pd
import os
from flask import Flask, request, render_template
import pickle
from sklearn import preprocessing   
label_encoder = preprocessing.LabelEncoder()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 

app = Flask(__name__)
MODEL_PATH = 'salesprofit_predictionmodel.pkl'
model = pickle.load(open(MODEL_PATH, 'rb'))
df = pd.read_csv("Sales_project_phaseII.csv")
df1 = pd.read_csv('50000 Sales Records.csv')

Order_Priority_dict = {'L':1, 'M':2, 'H':3, 'C':4}
Item_Type_dict = {'Meat':1,'Fruits':2,'Cosmetics':3,'Vegetables':4,'Personal Care':5,'Beverages':6,'Snacks':7,
                 'Clothes':8, 'Cereal':9, 'Household':0, 'Office Supplies':10 , 'Baby Food':11}
Sales_Channel_dict = {'Online':1 , 'Offline':0}
Region_dict = {'Sub-Saharan Africa':0,'Europe':1,'Asia':2,'Middle East and North Africa':3,
              'Central America and the Caribbean':4,'Australia and Oceania':5,'North America':6}



df['Country']= label_encoder.fit_transform(df['Country'])
df[['Region','Country','Item Type','Sales Channel','Order Priority','Total Revenue','Total Cost','Units Sold','Unit Price','Unit Cost','Order Date_year','Order Date_month','Order Date_day','Ship Date_year','Ship Date_month','Ship Date_day']] = scaler.fit_transform(df[['Region','Country','Item Type','Sales Channel','Order Priority','Total Revenue','Total Cost','Units Sold','Unit Price','Unit Cost','Order Date_year','Order Date_month','Order Date_day','Ship Date_year','Ship Date_month','Ship Date_day']])

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
        features[4] = Order_Priority_dict[features[4]]
        features[0] = Region_dict[features[0]]
        features[1] = label_encoder.transform([features[1]])
        features[2] = Item_Type_dict[features[2]]
        features[3] = Sales_Channel_dict[features[3]]

        # if(features[4] == 'H'):
        #     features[4] = 3
        # else:
        #     if(features[4] == 'M'):
        #         features[4] = 2
        #     else:
        #         features[4] = 1
         
        
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