from flask import Flask, render_template
import pickle
import pandas as pd
import numpy as np
from xgboost import Booster, XGBClassifier
import flask

app = Flask(__name__)

with open('one_hot.pkl', "rb") as f:
    enc = pickle.load(f)
with open('label.pkl', "rb") as f:
    label = pickle.load(f)
    
with open('pca.pkl', "rb") as f:
    pca = pickle.load(f)    
            
@app.route('/')
def home():
   return render_template('index.html')
            
@app.route('/predict', methods=['POST'])
def predict():
   # url = str(flask.request.form['url'])
   # comment = [text_preprocess(str(x)) for x in request.form.values()]
   data = {}
   data['source'] = [str(flask.request.form['source'])]
   data['browser'] = [str(flask.request.form['browser'])]
   data['sex'] = [str(flask.request.form['sex'])]
   data['age'] = [int(flask.request.form['age'])]
   data['country'] = [str(flask.request.form['country'])]
   data['signup_time'] = [pd.to_datetime(str(flask.request.form['sdate']+" "+str(flask.request.form['stime'])), errors='ignore')]
   data['purchase_time'] = [pd.to_datetime(str(flask.request.form['pdate']+" "+str(flask.request.form['ptime'])), errors='ignore')]
   data['time_difference'] = [(data['purchase_time'][0] - data['signup_time'][0])/np.timedelta64(1,'D')]
   
   X = pd.DataFrame(data)
   X.drop(['signup_time', 'purchase_time'], axis=1, inplace=True)
   
   categoricals = X.select_dtypes(include='object')
   categoricals = categoricals.astype(str)
   categoricals = categoricals.apply(label.fit_transform)
   label_encoding = categoricals['country']
   categoricals.drop(['country'], axis=1, inplace=True)
   X_one = enc.transform(categoricals)
   encoded_data = pd.DataFrame(X_one.todense())
   encoded_data.reset_index(drop=True, inplace=True)
   categoricals.reset_index(drop=True, inplace=True)
    
   original_numeric = X.select_dtypes(include='number')
   original_numeric.reset_index(drop=True, inplace=True)
    
   X = pd.concat([original_numeric, encoded_data, label_encoding], axis=1).values
   Xp = pca.transform(X) 

   booster = XGBClassifier()
   booster.load_model("xgb.model")   
   clf = XGBClassifier()
   booster = Booster()
   booster.load_model('xgb.model')
   clf._Booster = booster
   classes = clf.predict_proba(Xp)
   y_pred = [0 if c[0]>0.5 else 1 for c in classes]

   if y_pred[0]==1:
       result = '"Fraud transaction"'
   else:
       result = '"Not a fraud transaction"'

   return render_template('Result.html', result=result)

if __name__ == "__main__":
   app.run(debug=True)