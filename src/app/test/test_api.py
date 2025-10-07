import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API base URL
BASE_URL = "http://localhost:9696"

def test_health():
    """Test health endpoint"""
    logger.info("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    logger.info(f"Status: {response.status_code}")
    logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
    if response.status_code == 200:
        logger.info("Request successful!")
    else:
        logger.info("Request Failed!")
    logger.info("-" * 50)

def test_model_info():
    """Test model info endpoint"""
    logger.info("Testing model info endpoint...")
    response = requests.get(f"{BASE_URL}/model_info")
    logger.info(f"Status: {response.status_code}")
    logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
    if response.status_code == 200:
        logger.info("Request successful!")
    else:
        logger.info("Request Failed!")
    logger.info("-" * 50)

def test_example_request():
    """Get example request format"""
    logger.info("Getting example request format...")
    response = requests.get(f"{BASE_URL}/example_request")
    logger.info(f"Status: {response.status_code}")
    logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
    if response.status_code == 200:
        logger.info("Request successful!")
    else:
        logger.info("Request Failed!")
    logger.info("-" * 50)
    return response.json().get("example_request", {})

def test_prediction(data):
    """Test prediction endpoint"""
    logger.info("Testing prediction endpoint...")
    logger.info(f"Input data: {json.dumps(data, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={'Content-Type': 'application/json'}
    )
    
    logger.info(f"Status: {response.status_code}")
    logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        result = response.json()
        logger.info("Request successful!")
        logger.info(f"\nPREDICTION RESULT:")
        logger.info(f"   Income Category: {result.get('income_category')}")
        logger.info(f"   Probability: {result.get('prediction_proba', 0):.4f}")
        logger.info(f"   Confidence: {result.get('confidence', 0):.4f}")
        logger.info(f"   Threshold Used: {result.get('threshold_used', 0.5):.3f}")
    else:
        logger.error("Request Failed!")
    logger.info("-" * 50)

def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("TESTING FLASK API FOR ADULT INCOME PREDICTION")
    logger.info("=" * 60)
    
    try:
        # Test health
        test_health()
        
        # Test model info
        test_model_info()
        
        # Get example request and test prediction
        example_data = test_example_request()
        if example_data:
            test_prediction(example_data)
        
        # Test with custom data (complete features for high income prediction)
        custom_data = {
            "age": 35,
            "workclass": "Private",
            "fnlwgt": 280464,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 5178,
            "capital-loss": 0,
            "hours-per-week": 45,
            "native-country": "United-States"
        }
        
        logger.info("Testing with custom data (likely high income)...")
        test_prediction(custom_data)
        
        # Test with another custom data (complete features for low income prediction)
        custom_data_2 = {
            "age": 22,
            "workclass": "Private",
            "fnlwgt": 201490,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Never-married",
            "occupation": "Handlers-cleaners",
            "relationship": "Own-child",
            "race": "White",
            "sex": "Female",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 25,
            "native-country": "United-States"
        }
        
        logger.info("Testing with custom data (likely low income)...")
        test_prediction(custom_data_2)
        
        # Summary
        logger.info("=" * 60)
        logger.info("ALL TESTS COMPLETED!")
        logger.info("=" * 60)
        
    except requests.exceptions.ConnectionError:
        logger.error("ERROR: Could not connect to the API. Make sure the Flask server is running!")
        logger.info("To start the server, run: python predict_api.py")
    except Exception as e:
        logger.error(f"ERROR: {e}")

if __name__ == "__main__":
    logger.info("Starting comprehensive test suite for Adult Income Prediction API...")
    logger.info("This will test all endpoints including error handling...")
    main()