from flask import Flask,request
import pandas as pd
import pickle as pkl
from flasgger import Swagger

#point of start of flask app
app = Flask(__name__)
# indication to flask to generate UI
# Run this - http://localhost:5000/apidocs/ in browser to open swagger app
Swagger(app)

# load the model at the start of the app
pickle_in = open('model.pkl','rb')
model_iris = pkl.load(pickle_in)

def get_description(int_code):
    if (int_code==0):
        desc = 'Setosa'
    elif (int_code == 1):
        desc = 'Versicolour'
    else:
        desc = 'Virginica'
    return desc

@app.route('/')
def Welcome():
    return "Hello world, Jayanth"

@app.route('/iris_predict',methods=['GET'])
def iris_predict():
    """
    UI for testing the API
    This is using Doc Strings for specification
    ---
    parameters:
        - name: sl
          in: query
          type: number
          required: true
        - name: sw
          in: query
          type: number
          required: true
        - name: pl
          in: query
          type: number
          required: true
        - name: pw
          in: query
          type: number
          required: true
    responses:
          200:
              description: The output values
    """
    sl = float(request.args.get('sl'))
    sw = float(request.args.get('sw'))
    pl = float(request.args.get('pl'))
    pw = float(request.args.get('pw'))
    prediction = model_iris.predict([[sl,sw,pl,pw]])
    return "The prediction is: " + get_description(int(prediction[0]))

# Bulk can be tested in postman only
@app.route('/iris_predict_bulk',methods=['POST'])
def iris_predict_bulk():
    """
    UI for testing the API - Bulk testing
    This is using Doc Strings for specification
    ---
    parameters:
        - name: test_file
          in: formData
          type: file
          required: true
    responses:
          200:
              description: The output values
    """
    # test_file is the name of variable in url
    df = pd.read_csv(request.files.get('test_file'))
    prediction = list(model_iris.predict(df))
    #generates prediction text for each prediction values
    prediction_text =  [get_description(i) for i in prediction]
    return "The prediction for CSV is: " + str(prediction_text)

if(__name__=='__main__'):
    app.run()
