import requests
import pytest

def test_query_endpoint():
    """
    Test the /query endpoint by making a real HTTP request
    """
    # API endpoint URL
    url = "http://localhost:8000/query"
    
    # Prepare test query
    test_query = "Где найти информацию о социальных услугах в Санкт-Петербурге?"
    
    # Send POST request to the query endpoint
    response = requests.post(
        url, 
        json={"query": test_query}
    )
    
    # Check response status code
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    # Parse response JSON
    result = response.json()
    print(f'Result: {result}')

    # Validate response structure
    assert "response" in result, "Response should contain 'response' field"
    assert "sources" in result, "Response should contain 'sources' field"
    assert "confidence" in result, "Response should contain 'confidence' field"
    
    # Check that response is not empty
    assert result["response"], "Response text should not be empty"
    assert result["confidence"] >= 0, "Confidence should be non-negative"
    assert result["confidence"] <= 1, "Confidence should not exceed 1.0" 