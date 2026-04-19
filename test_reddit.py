import requests
from ddgs import DDGS

print("--- Testing DDGS site:reddit.com ---")
try:
    results = DDGS().text("site:reddit.com donald trump polymarket", max_results=3)
    for r in results:
        print(r['title'])
except Exception as e:
    print("DDGS Error:", e)

print("\n--- Testing Reddit .json Endpoint ---")
try:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) PolymarketResearchBot/0.1'}
    r = requests.get("https://www.reddit.com/r/politics/search.json?q=trump&restrict_sr=1&sort=new&limit=3", headers=headers)
    if r.status_code == 200:
        data = r.json()
        for child in data['data']['children']:
            print(child['data']['title'])
    else:
        print("Status Code:", r.status_code)
except Exception as e:
    print("Requests Error:", e)
