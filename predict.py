import pickle
from flask import Flask, request, jsonify, render_template

# Load the model
input_file = 'final_model.bin'

# Load the model from the pickle file
with open(input_file, 'rb') as f_in:
    dv, pca, model = pickle.load(f_in)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    flight = request.get_json()
    X = dv.transform([flight])
    X = pca.transform(X)
    # Modify this line to ignore the 'monotonic_cst' attribute
    y_pred = model.predict_proba(X)[0, 1]
    delay = y_pred >= 0.5
    result = {
        'delay_probability': float(y_pred),
        'delay': bool(delay)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=9696)
