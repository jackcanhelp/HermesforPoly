from ddgs import DDGS
import json

try:
    results = DDGS().text("site:github.com polymarket agent twitter reddit sentiment", max_results=5)
    for r in results:
        print(r['title'])
        print(r['href'])
        print(r['body'])
        print('-'*40)
except Exception as e:
    print(e)
