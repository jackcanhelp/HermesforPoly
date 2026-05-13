import sqlite3
import requests
import json
import os

db_path = "d:\\Projects\\HermesforPolymarket\\paper_trading.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT id, market_id, question FROM paper_trades WHERE status = 'OPEN'")
open_trades = cursor.fetchall()

print(f"Total OPEN trades in DB: {len(open_trades)}")

for trade_id, m_id, q in open_trades:
    url = f"https://gamma-api.polymarket.com/markets/{m_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            m = r.json()
            is_closed = m.get("closed")
            active = m.get("active")
            
            prices = m.get('outcomePrices', [])
            if isinstance(prices, str):
                try: prices = json.loads(prices)
                except: pass
            
            print(f"[{m_id}] {q[:50]}")
            print(f"  Keys indicating status: closed={is_closed}, active={active}")
            for k in ["resolvedBy", "conditionId", "questionID", "status", "hasResolved"]:
                if k in m:
                    print(f"  {k} = {m[k]}")
    except Exception as e:
        pass

conn.close()
