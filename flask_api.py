# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:50:04 2020

@author: krish.naik
"""

from flask import Flask, request
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)

pickle_in = open("classifier1.pkl","rb")
classifier=pickle.load(pickle_in)


def welcome():
    return "Welcome All"

@app.route('/predict')
def predict_note_authentication():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "Hello The answer is"+str(prediction)

@app.route('/predict_file', methods = ["POST"])
def predict_note_file():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return "The prdicted values are"+str(list(prediction))




if __name__=='__main__':
    app.run()
    
    