from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and label encoder
try:
    with open('smell_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('label_encoder.pkl', 'rb') as le_file:
        label_encoder = pickle.load(le_file)
    print("Model and label encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    model = None
    label_encoder = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoder is None:
        return jsonify({'error': 'Model or label encoder is not loaded'}), 500

    try:
        # Get data from POST request
        data = request.json
        ppm = data.get('ppm')

        # Validate input
        if ppm is None:
            return jsonify({'error': 'No ppm value provided'}), 400

        # Create a DataFrame with the correct feature names
        df = pd.DataFrame([[ppm]], columns=['ppm'])
        print(f"Input DataFrame: {df}")

        # Perform prediction
        prediction_numeric = model.predict(df)
        print(f"Numeric Prediction: {prediction_numeric[0]}")

        # Convert numeric prediction back to original label
        prediction_label = label_encoder.inverse_transform(prediction_numeric)
        print(f"Original Label Prediction: {prediction_label[0]}")

        # Return the result
        return jsonify({'prediction': prediction_label[0]})
    
    except Exception as e:
        # Print error to the console for debugging
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
