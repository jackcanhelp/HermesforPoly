import requests
import os
from dotenv import load_dotenv

load_dotenv()

def check_models(provider, url, key):
    headers = {"Authorization": f"Bearer {key}"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            models = [m["id"] for m in data["data"]]
            print(f"--- {provider} Models ---")
            print("\n".join(models))
        else:
            print(f"--- {provider} Models ---")
            print(data)
    except Exception as e:
        print(f"Failed for {provider}: {e}")

groq_key = os.getenv("GROQ_API_KEYS", "").split(",")[0]
cerebras_key = os.getenv("CEREBRAS_API_KEYS", "").split(",")[0]
nvidia_key = os.getenv("NVIDIA_API_KEYS", "").split(",")[0]

check_models("Groq", "https://api.groq.com/openai/v1/models", groq_key)
check_models("Cerebras", "https://api.cerebras.ai/v1/models", cerebras_key)
check_models("Nvidia", "https://integrate.api.nvidia.com/v1/models", nvidia_key)
