from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    ss = pickle.load(scaler_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract the form data
    form_data = request.form
    
    # Create a DataFrame from the form data
    new_data = pd.DataFrame({
        'Operator': [int(form_data['operator'])],
        'In Out Travelling': [int(form_data['in_out_travelling'])],
        'Network Type': [int(form_data['network_type'])],
        'Rating': [int(form_data['rating'])],
        'Latitude': [float(form_data['latitude'])],
        'Longitude': [float(form_data['longitude'])],
        'State Name': [int(form_data['state_name'])],
        'Call Success Rate (%)': [float(form_data['call_success_rate'])],
        'Time of Day': [int(form_data['time_of_day'])],
        'Complaint Filed (Yes/No)': [int(form_data['complaint_filed'])],
        'Roaming (Yes/No)': [int(form_data['roaming'])],
        'Call Type': [int(form_data['call_type'])]
    })

    # Get the correct feature names from the scaler
    original_features = ss.get_feature_names_out()

    new_data_reconstructed = pd.DataFrame(0, columns=original_features, index=new_data.index)

    for feature in new_data.columns:
        if feature in original_features:
            new_data_reconstructed[feature] = new_data[feature]

    # Scale the new data
    new_data_scaled = pd.DataFrame(ss.transform(new_data_reconstructed), columns=new_data_reconstructed.columns)
    
    # Make predictions using the loaded model
    predictions = model.predict(new_data_scaled)

    # Map predictions to categories
    prediction_map = {0: 'Call dropped',1: 'Poor Voice Quality', 2: 'Satisfactory'}
    prediction = prediction_map[predictions[0]]
    
    return render_template('result.html', prediction=prediction)

if __name__ == '_main_':
    app.run(host='0.0.0.0',port=5000)