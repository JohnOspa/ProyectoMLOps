#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variable to store the loaded model
model_package = None

def load_model(model_path="../train/models/model_complete.bin"):
    """Load the complete model package from .bin file"""
    global model_package
    try:
        with open(model_path, "rb") as f:
            model_package = pickle.load(f)
        logger.info("Model package loaded successfully")
        logger.info(f"Model accuracy: {model_package['accuracy']:.4f}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model_package is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    
    return jsonify({
        'status': 'healthy',
        'model_accuracy': model_package['accuracy'],
        'model_params': model_package['model_params']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if model_package is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Expected features for Adult dataset
        expected_features = [
            'age', 'workclass', 'education', 'marital-status', 
            'occupation', 'relationship', 'race', 'sex', 
            'native-country'
        ]
        
        # Validate that all required features are present
        missing_features = [f for f in expected_features if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}'
            }), 400
        
        # Prepare features dictionary
        features = {key: data[key] for key in expected_features}
        
        # Transform features using the preprocessor
        preprocessor = model_package['preprocessor']
        X = preprocessor.transform([features])
        
        # Make prediction
        model = model_package['model']
        dmatrix = xgb.DMatrix(X)
        prediction_proba = model.predict(dmatrix)[0]
        prediction = 1 if prediction_proba > 0.5 else 0
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'prediction_proba': float(prediction_proba),
            'income_category': '>50K' if prediction == 1 else '<=50K',
            'confidence': float(max(prediction_proba, 1 - prediction_proba))
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if model_package is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    info = {
        'accuracy': model_package['accuracy'],
        'model_params': model_package['model_params'],
        'feature_names': model_package.get('feature_names', 'Not available')
    }
    
    return jsonify(info)

@app.route('/example_request', methods=['GET'])
def example_request():
    """Get an example request format"""
    example = {
        "age": 25,
        "workclass": "Private",
        "education": "HS-grad",
        "marital-status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "native-country": "United-States"
    }
    
    return jsonify({
        "example_request": example,
        "usage": "POST to /predict with this JSON format"
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting Flask API server...")
        app.run(host='0.0.0.0', port=9696, debug=True)
    else:
        logger.error("Failed to start server - model could not be loaded")