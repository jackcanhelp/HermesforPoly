import os
import requests
from dotenv import load_dotenv

def test_ollama_cloud():
    load_dotenv()
    keys_str = os.getenv("OLLAMA_API_KEYS", "")
    if not keys_str:
        print("❌ Error: No API keys found in .env file.")
        return
        
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    test_key = keys[0]
    
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {test_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "glm-4",
        "messages": [
            {"role": "user", "content": "Hello, are you Kimi? Please reply in one sentence."}
        ],
        "temperature": 0.1
    }
    
    print(f"Testing connection to {url} with model 'glm-4'...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            print("Connection SUCCESS!")
            print("Response:", response.json()["choices"][0]["message"]["content"])
        else:
            print(f"Connection FAILED. Status Code: {response.status_code}")
            print("Error Details:", response.text)
            
    except Exception as e:
        print(f"Connection ERROR: {e}")

if __name__ == "__main__":
    test_ollama_cloud()
