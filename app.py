from flask import Flask, render_template, request, Response
import pickle
import pandas as pd
import numpy as np
from xgboost import Booster, XGBClassifier
import flask
from io import StringIO

app = Flask(__name__)

with open('one_hot.pkl', "rb") as f:
    enc = pickle.load(f)
with open('label.pkl', "rb") as f:
    label = pickle.load(f)
    
with open('pca.pkl', "rb") as f:
    pca = pickle.load(f)    
            
def get_results(X):    
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
 
   clf = XGBClassifier()
   booster = Booster()
   booster.load_model('xgb.model')
   clf._Booster = booster
   classes = clf.predict_proba(Xp)
   y_pred = [0 if c[0]>0.5 else 1 for c in classes]

   return y_pred

def generate_csv_file(df):
    file_buffer = StringIO()
    df.to_csv(file_buffer, encoding="utf-8", index=False, sep=",")
    file_buffer.seek(0)
    return file_buffer

def fill_na(X):
    if True in X.isna().sum()>0:
        X.source.fillna('SEO')
        X.browser.fillna('Chrome')
        X.sex.fillna('M')
        X.age.fillna(33.14)
        X.country.fillna('United States')
        X.signup_time.fillna('2015-04-20 00:56:09.511313920')
        X.purchase_time.fillna('2015-06-16 02:56:38.759956736')
    return X
    
@app.route('/')
def home():
   return render_template('Choose.html')

@app.route('/single_input')
def single_input():
   return render_template('Index.html')

@app.route('/batch')
def batch():
   return render_template('Batch.html')

@app.route('/batch/batch_pred', methods=['POST'])
def batch_pred():
   uploaded_file = request.files['file']
   if uploaded_file.filename != '':
       data = pd.read_csv(uploaded_file.filename, parse_dates=["signup_time", "purchase_time"])
       data = fill_na(data)
       data['signup_time'] = pd.to_datetime(data['signup_time'], errors='ignore')
       data['purchase_time'] = pd.to_datetime(data['purchase_time'], errors='ignore')
       
       categories = ['source', 'browser', 'sex', 'age', 'country', 'purchase_time', 'signup_time']
       X = data[categories]
       X['time_difference'] = (X['purchase_time'] - X['signup_time'])/np.timedelta64(1,'D')
       X.drop(['purchase_time', 'signup_time'], axis=1, inplace=True)
       y_pred = get_results(X)
       y_pred = pd.DataFrame(y_pred, columns=['Class'])
       
       file = pd.concat([data, y_pred], axis=1)
       generated_file = generate_csv_file(file)
       response = Response(generated_file, mimetype="text/csv")
       # add a filename
       response.headers.set(
          "Content-Disposition", "attachment", filename="output.csv")
       return response

   return render_template('Batch.html')
            
@app.route('/single_input/predict', methods=['POST'])
def predict():
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
   
   y_pred = get_results(X)
   
   if y_pred[0]==1:
       result = '"Fraud transaction"'
   else:
       result = '"Not a fraud transaction"'

   return render_template('Result.html', result=result)

if __name__ == "__main__":
   app.run(debug=True)