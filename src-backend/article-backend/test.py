import requests

URL = "http://127.0.0.1:5002/recommend/article"
payload = {"gender": 2, "age": 6}
response = requests.post(URL, json=payload)

print("✅ Status:", response.status_code)
print(response.json())