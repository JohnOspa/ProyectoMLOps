#!/usr/bin/env python
# coding: utf-8

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
        logger.info("Request successful!")
    else:
        logger.info("Request Failed!")
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
        
        # Test with custom data
        custom_data = {
            "age": 35,
            "workclass": "Private",
            "education": "Bachelors",
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "native-country": "United-States"
        }
        
        logger.info("Testing with custom data (likely high income)...")
        test_prediction(custom_data)
        
        # Test with another custom data
        custom_data_2 = {
            "age": 22,
            "workclass": "Private",
            "education": "HS-grad",
            "marital-status": "Never-married",
            "occupation": "Handlers-cleaners",
            "relationship": "Own-child",
            "race": "White",
            "sex": "Female",
            "native-country": "United-States"
        }
        
        logger.info("Testing with custom data (likely low income)...")
        test_prediction(custom_data_2)
        
    except requests.exceptions.ConnectionError:
        logger.info("ERROR: Could not connect to the API. Make sure the Flask server is running!")
    except Exception as e:
        logger.info(f"ERROR: {e}")

if __name__ == "__main__":
    logger.info("Starting test for Adult API...")
    main()