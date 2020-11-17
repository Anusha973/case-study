# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 19:38:59 2020

@author: AnushaMacharla
"""

import numpy as np
from flask import Flask, render_template,request
import pickle

app = Flask(__name__,template_folder='template',static_folder='static')
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('anu.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('anu.html', prediction_text='MPG should be  {}'.format(output))

'''if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)
    '''

if __name__ == "__main__":
    app.run(debug=True)
    
  