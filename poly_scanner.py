import requests
import pandas as pd
import time

GAMMA_API_URL = "https://gamma-api.polymarket.com"

def fetch_active_markets(limit=50):
    url = f"{GAMMA_API_URL}/markets"
    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "order": "volumeNum",
        "ascending": "false"  # Get highest volume first
    }
    print(f"Fetching active markets from {url}...")
    try:
        response = requests.get(url, params=params, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Polymarket API: {e}")
        return None
    
    if response.status_code == 200:
        data = response.json()
        markets = []
        for m in data:
            # We are looking for high volume, potentially mispriced markets
            try:
                question = m.get('question', '')
                volume = m.get('volumeNum', 0)
                liquidity = m.get('liquidity', 0)
                
                # Sometims outcomes are directly in the market dict
                outcomes = m.get('outcomes', [])
                if isinstance(outcomes, str):
                    import json
                    try:
                        outcomes = json.loads(outcomes)
                    except:
                        pass
                
                prices = m.get('outcomePrices', [])
                if isinstance(prices, str):
                    import json
                    try:
                        prices = json.loads(prices)
                    except:
                        pass
                
                markets.append({
                    'id': m.get('id'),
                    'condition_id': m.get('conditionId'),
                    'question': question,
                    'volume': volume,
                    'liquidity': liquidity,
                    'outcomes': outcomes,
                    'prices': prices,
                    'endDate': m.get('endDate', ''),
                    'category': m.get('category', 'Unknown')
                })
            except Exception as e:
                print(f"Error parsing market: {e}")
                
        df = pd.DataFrame(markets)
        # Parse dates
        if not df.empty and 'endDate' in df.columns:
            df['endDate'] = pd.to_datetime(df['endDate'], format='mixed', utc=True)
            
        return df
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

if __name__ == "__main__":
    df = fetch_active_markets(100)  # 擴大抓取數量
    if df is not None:
        print("\n=== 潛在定價誤差市場過濾 ===")
        potential_markets = []
        for _, row in df.iterrows():
            outcomes = row['outcomes'] if isinstance(row['outcomes'], list) else []
            prices = row['prices'] if isinstance(row['prices'], list) else []
            
            # 過濾條件: 
            # 1. 有價格資料 
            # 2. 至少有一個選項的價格在 0.05 到 0.95 之間 (排除幾乎確定或根本不可能的事件)
            # 3. Liquidity 大於 1000
            
            is_interesting = False
            for p in prices:
                try:
                    price_val = float(p)
                    if 0.05 < price_val < 0.95:
                        is_interesting = True
                except:
                    pass
            
            if is_interesting and float(row['liquidity']) > 1000:
                potential_markets.append(row)
                
        print(f"從 100 個熱門市場中，篩選出 {len(potential_markets)} 個具分析價值的目標：\n")
        
        for row in potential_markets[:10]:  # 只顯示前 10 個
            print(f"[類別: {row['category']}] Q: {row['question']}")
            print(f"流動性: ${float(row['liquidity']):,.2f}")
            outcomes = row['outcomes'] if isinstance(row['outcomes'], list) else []
            prices = row['prices'] if isinstance(row['prices'], list) else []
            for i in range(min(len(outcomes), len(prices))):
                print(f"  - {outcomes[i]}: {float(prices[i])*100:.1f}%")
            print("-" * 50)
