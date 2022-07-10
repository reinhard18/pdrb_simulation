from flask import Flask, render_template, url_for, request, redirect, jsonify
import pickle
import numpy as np

app = Flask(__name__)
final_pertanian_knn = pickle.load(open('models/final_pertanian_knn.pkl', 'rb'))
final_pertanian_rf = pickle.load(open('models/final_pertanian_rf.pkl', 'rb'))


@app.route('/')
def index():
    return redirect('/pertanian')


@app.route('/pertanian')
def pertanian():
    return render_template('pertanian.html')


@app.route('/pertambangan')
def tambang():
    return render_template('pertambangan.html')


@app.route('/perdagangan')
def dagang():
    return render_template('perdagangan.html')


@app.route('/konstruksi')
def konstruksi():
    return render_template('konstruksi.html')


@app.route('/listrik')
def listrik():
    return render_template('listrik.html')


@app.route('/forecast')
def forecast():
    return render_template('forecast.html')


@app.route('/pertanian', methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction_knn = final_pertanian_knn.predict(final_features)
    prediction_rf = final_pertanian_rf.predict(final_features)

    output_knn = round(prediction_knn[0], 2)
    output_rf = round(prediction_rf[0], 2)

    return render_template('pertanian.html', prediction_text_knn='Rp. {}'.format(output_knn), prediction_text_rf='Rp. {}'.format(output_rf))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = final_pertanian_knn.pkl.predict(
        [np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
