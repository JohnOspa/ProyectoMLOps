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
        logger.info(f"Model accuracy: {model_package['metrics']['accuracy']:.4f}")
        logger.info(f"Model threshold: {model_package['threshold_f1']:.3f}")
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
        'model_metrics': model_package['metrics'],
        'model_params': model_package['params'],
        'threshold_f1': model_package['threshold_f1']
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
        
        # Expected features for Adult dataset (según load_data.py)
        expected_features = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
        ]
        
        # Validate that all required features are present
        missing_features = [f for f in expected_features if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}',
                'expected_features': expected_features
            }), 400
        
        # Prepare features dictionary (aplicar el mismo preprocesamiento que en load_data.py)
        features = {key: data[key] for key in expected_features}
        
        # Aplicar el mapeo de 'sex' como en load_data.py
        if 'sex' in features:
            mapeo_sex = {'Female': 1, 'Male': 0}
            if features['sex'] in mapeo_sex:
                features['sex'] = mapeo_sex[features['sex']]
        
        # Transform features using the preprocessor
        preprocessor = model_package['preprocessor']
        X = preprocessor.transform([features])
        
        # Make prediction using optimal threshold
        model = model_package['model']
        dmatrix = xgb.DMatrix(X)
        prediction_proba = model.predict(dmatrix)[0]
        
        # Usar el threshold óptimo encontrado durante el entrenamiento
        optimal_threshold = model_package.get('threshold_f1', 0.5)
        prediction = 1 if prediction_proba > optimal_threshold else 0
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'prediction_proba': float(prediction_proba),
            'income_category': '>50K' if prediction == 1 else '<=50K',
            'confidence': float(max(prediction_proba, 1 - prediction_proba)),
            'threshold_used': float(optimal_threshold),
            'model_metrics': model_package['metrics']
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
        'metrics': model_package['metrics'],
        'params': model_package['params'],
        'threshold_f1': model_package['threshold_f1'],
        'features_names': model_package.get('features_names', 'Not available')
    }
    
    return jsonify(info)

@app.route('/example_request', methods=['GET'])
def example_request():
    """Get an example request format"""
    example = {
        "age": 25,
        "workclass": "Private",
        "fnlwgt": 226802,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    
    return jsonify({
        "example_request": example,
        "usage": "POST to /predict with this JSON format",
        "note": "sex should be 'Male' or 'Female' (will be encoded automatically)"
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting Flask API server...")
        app.run(host='0.0.0.0', port=9696, debug=True)
    else:
        logger.error("Failed to start server - model could not be loaded")