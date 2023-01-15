import pickle
from flask import Flask, request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

def prepare_data(data_pt):
    test = pd.DataFrame(data_pt,columns=orig_columns)

    test['Avg_Temp'] = 0
    test['Min_Temp'] = 0
    test['Max_Temp'] = 0
    test['Avg_RH'] = 0
    test['Min_RH'] = 0
    test['Max_RH'] = 0
    test['Avg_Ws'] = 0
    test['Min_Ws'] = 0
    test['Max_Ws'] = 0
    test['Avg_Rain'] = 0
    test['Min_Rain'] = 0
    test['Max_Rain'] = 0
    test['Date'] = 0
    test['Day of Week'] = 0
    test['Weekday/Weekend'] = 0
    poly_columns = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']
    poly = PolynomialFeatures(2)
    expanded_test = poly.fit_transform(test[poly_columns])
    feature_names = poly.get_feature_names_out(input_features=None)
    expanded_test = pd.DataFrame(expanded_test, columns=feature_names)
    expanded_test = expanded_test.drop(columns=['1'])
    test.reset_index(inplace=True)
    expanded_test.insert(1, "Avg_Temp", test['Avg_Temp'])
    expanded_test.insert(2, "Min_Temp", test['Min_Temp'])
    expanded_test.insert(3, "Max_Temp", test['Max_Temp'])
    expanded_test.insert(5, "Avg_RH", test['Avg_RH'])
    expanded_test.insert(6, "Min_RH", test['Min_RH'])
    expanded_test.insert(7, "Max_RH", test['Max_RH'])
    expanded_test.insert(9, "Avg_Ws", test['Avg_Ws'])
    expanded_test.insert(10, "Min_Ws", test['Min_Ws'])
    expanded_test.insert(11, "Max_Ws", test['Max_Ws'])
    expanded_test.insert(12, "Avg_Rain", test['Avg_Rain'])
    expanded_test.insert(13, "Min_Rain", test['Min_Rain'])
    expanded_test.insert(14, "Max_Rain", test['Max_Rain'])
    expanded_test.insert(0, "Weekday/Weekend", test['Weekday/Weekend'])
    columns = []
    for column in expanded_test.columns:
        columns.append(column)

    #Standardization
    expanded_test.loc[:, expanded_test.columns != 'Classes'] = scaler.transform(expanded_test.loc[:, expanded_test.columns != 'Classes'])
    expanded_test = pd.DataFrame(expanded_test, columns=columns)

    # #Taking only those columns which were selected using SFS on train dataset
    expanded_test = expanded_test[best_model_params['columns']]
    return expanded_test

#Loading our model
best_model_params = pickle.load(open('./best_model.pkl','rb'))
best_model = best_model_params['model']
scaler = best_model_params['scaler']
orig_columns = ['Date', 'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI','BUI']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    data_pt = np.array(list(data.values())).reshape(1,-1)
    expanded_test = prepare_data(data_pt)

    #Prediction
    output = int(best_model.predict(expanded_test))
    print(output)
    return jsonify(output)

@app.route('/predict',methods = ['POST'])
def predict():
    data = [x for x in request.form.values()]
    for i in range(1,len(data)):
        data[i] = float(i)
    data_pt = np.array(data).reshape(1,-1)
    expanded_test = prepare_data(data_pt)

    output = int(best_model.predict(expanded_test))
    if output == 0:
        out_val = "No Fire Predicted"
    else:
        out_val = "Forest Fire Predicted"
    return render_template("home.html",prediction_text='The Prediction is: {}'.format(out_val))
    

if __name__ == '__main__':
    app.run(debug=True)