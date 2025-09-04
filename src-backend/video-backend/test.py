import requests

# ✅ URL backend Flask (endpoint đã định nghĩa)
URL = "http://127.0.0.1:5001/recommend/video"

# ✅ Data input cho API
payload = {
    "gender": 2,
    "age": 2
}

try:
    print(f"[TEST] Sending POST request to {URL} with payload: {payload}")
    response = requests.post(URL, json=payload)

    print(f"[TEST] Status Code: {response.status_code}")
    print(f"[TEST] Raw Response: {response.text}")  # in raw text để debug

    if response.status_code == 200:
        print("✅ JSON Response:")
        print(response.json())
    else:
        print("❌ Request failed!")
except Exception as e:
    print("❌ Error while sending request:", e)
