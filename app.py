import pickle
from flask import Flask, request,app,jsonify,url_for,render_template # later we use redirect,flash,session,escape
import numpy as np
import pandas


## defining the app and this is the satrt point from where the application will run

app = Flask(__name__)

## load the pickle file
scalar = pickle.load(open('scaling.pkl','rb'))
model = pickle.load(open('salesXgpred.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST']) # use this for postman to check model whether it can output the results by taking some inputs
def predict_api():
    data = request.json['data'] # the input is convert to json format
    print(data)
    print(np.array(list(data.values())).reshape(1,-1)) # taking the dictionary values and changing the dimension to 2
    # standardization done on the independent values, use the scaling pickle file to scale the independent variables
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(output[0]) # it is in the two dimensional array, so take first value by [0]
    return jsonify([np.round((output[0]-79370.76),2),np.round((output[0]+79370.76),2)])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    store = request.form["Store"]
    cpi = request.form["CPI"]
    #Temperature = request.form["Temperature"]
    unemployment = request.form["Unemployment"]
    output= model.predict(final_input)[0]
    output_range = ['{:,.2f}'.format(output-67110.06),'{:,.2f}'.format(output+67110.06)]
    return render_template("home.html",prediction_text="The predicted sales range is  $  {} "  "for Store={},CPI={},Unemployment={}".format(output_range,store,cpi,unemployment))


if __name__ =="__main__":
    app.run(debug=True)


