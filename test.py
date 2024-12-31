import requests

# Test a question
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "What is artificial intelligence?"}
)
print(response.json())