import requests
url = "https://gamma-api.polymarket.com/markets?limit=1"
r = requests.get(url)
print(r.json()[0].keys())
print(r.json()[0].get('endDate'))
