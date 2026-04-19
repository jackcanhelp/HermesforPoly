from duckduckgo_search import DDGS

try:
    results = DDGS().text("donald trump greenland", max_results=3)
    print("DDGS results length:", len(list(results)))
    print("Results:", results)
except Exception as e:
    print("DDGS Exception:", e)
