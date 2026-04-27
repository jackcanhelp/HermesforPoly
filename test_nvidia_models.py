import os
import requests
from dotenv import load_dotenv

def test_nvidia_models():
    load_dotenv()
    keys_str = os.getenv("NVIDIA_API_KEYS", "")
    if not keys_str:
        print("❌ Error: No NVIDIA API keys found.")
        return
        
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    test_key = keys[0]
    
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {test_key}",
        "Content-Type": "application/json"
    }
    
    models_to_test = ["minimax/minimaxm2.7", "moonshotai/kimi-k2.5"]
    
    for model in models_to_test:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Hello, are you there? Reply in one short sentence."}
            ],
            "temperature": 0.1,
            "max_tokens": 10
        }
        
        print(f"\nTesting connection to NVIDIA NIM with model: '{model}'")
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                print("Connection SUCCESS!")
                print("Response:", response.json()["choices"][0]["message"]["content"])
            else:
                print(f"Connection FAILED. Status Code: {response.status_code}")
                try:
                    print("Error Details:", response.json())
                except:
                    print("Error Details:", response.text)
                
        except Exception as e:
            print(f"Connection ERROR: {e}")

if __name__ == "__main__":
    test_nvidia_models()
