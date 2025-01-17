import pickle
from flask import Flask,request,app,jsonify,url_for,render_template


import numpy as np
import pandas as pd


app = Flask(__name__,template_folder='template')





##load the model
model = pickle.load(open('house_price_model.pkl','rb'))


file_name = 'C:/Users/sanoj/endtoendMLprojects2025/californiahousepricing/scaling.pkl'


try:
    with open(file_name,'rb') as f:
      scalar = pickle.load(f)

except EOFError:
    print("EOFError: The file is empty or corrupted.")
except FileNotFoundError:
    print("FileNotFoundError: The file does not exist.")
except pickle.UnpicklingError:
    print("UnpicklingError: The file is not a valid pickle.")


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])


def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))

    ###get newdata
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output= model.predict(new_data)
    print(output[0])

    return jsonify(output[0])



@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template('home.html',prediction_text = "THE HOUSE PRICE IS {}".format(output))



if __name__ == "__main__":
    app.run(debug=True)

    
