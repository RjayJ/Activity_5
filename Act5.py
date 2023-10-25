from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the SVM model
model = joblib.load('svm_model.pkl')
sc = joblib.load('scaler.pkl')


@app.route('/')
def index():
    return render_template('Act5_temp.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        experience = float(request.form['experience'])
        income = float(request.form['income'])
        zip_code = float(request.form['zip_code'])
        family = float(request.form['family'])
        CCavg = float(request.form['CCavg'])
        education = float(request.form['education'])
        mortgage = float(request.form['mortgage'])
        personal_loan = float(request.form['personal_loan'])
        securities_account = float(request.form['securities_account'])
        cd_account = float(request.form['cd_account'])
        online = float(request.form['online'])

        user_input = np.array([age, experience, income, zip_code, family, CCavg, education, mortgage, personal_loan, securities_account, cd_account, online]).reshape(1, -1)

        user_input_scaled = sc.transform(user_input)

        prediction = model.predict(user_input_scaled)

        if prediction == 1:
            result = "Prediction result is 1 Eligible for a Credit Card"
        else:
            result = "Prediction result is 0 Not Eligible for a Credit Card"

        return render_template('Act5_temp.html', prediction_result=result)

    except Exception as e:
        return render_template('Act5_temp.html', prediction_result="Error: Please check your input.")


if __name__ == '__main__':
    app.run(debug=True)
