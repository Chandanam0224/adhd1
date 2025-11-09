import requests
PAT = "f1e80a4dfb3145c7a12b8428bc62197e"
url = "https://api.clarifai.com/v2/models/phi-4-mini-instruct/outputs"
headers = {"Authorization": f"Key {PAT}", "Content-Type": "application/json"}
payload = {"inputs":[{"data":{"text":{"raw":"Give me 1 ADHD study tip."}}}]}
r = requests.post(url, headers=headers, json=payload, timeout=30)
print("Status:", r.status_code)
print("Body:", r.text)
