import time
import logging
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from poly_scanner import fetch_active_markets
from researcher import PolyResearcher
from agent import HermesAgent
from tracker import PaperTracker

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_potential_markets(df):
    """
    從資料集中過濾出流動性佳，且機率非極端的市場
    """
    potential_markets = []
    now = datetime.now(timezone.utc)
    for _, row in df.iterrows():
        # 濾波器：只選 14 天內會結算的短期事件，加速 Feedback Loop 和賺錢效率
        end_date = row.get('endDate')
        if pd.notna(end_date):
            days_to_end = (end_date - now).days
            if days_to_end > 14:
                continue

        outcomes = row.get('outcomes', [])
        # 嚴格防呆檢查：只允許標準的二元 (Yes/No) 市場，因多選題 (如各國總統候選人) 的 EV 與 Kelly 計算方式完全不同
        if not isinstance(outcomes, list) or len(outcomes) != 2:
            continue
        if "Yes" not in outcomes or "No" not in outcomes:
            continue
            
        prices = row['prices'] if isinstance(row['prices'], list) else []
        is_interesting = False
        
        for p in prices:
            try:
                price_val = float(p)
                if 0.05 < price_val < 0.95:
                    is_interesting = True
            except:
                pass
                
        if is_interesting and float(row.get('liquidity', 0)) > 500: # 將流動性門檻降到 500，納入稍微冷門但可能有更高利潤的市場
            potential_markets.append(row)
            
    return potential_markets

def main():
    # 1. 初始化模組
    researcher = PolyResearcher()
    agent = HermesAgent(model_name="gemma", api_url="http://127.0.0.1:11434/api/chat")
    tracker = PaperTracker()
    
    # 2. 獲取市場 (掃描前 200 個最熱門的市場)
    df = fetch_active_markets(200)
    if df is None:
        return
        
    open_market_ids = tracker.get_open_market_ids()
    all_potential = filter_potential_markets(df)
    
    # 剔除已經在手上的未平倉單 (避免重複計算 EV 與浪費運算)
    markets = [m for m in all_potential if str(m.get('id', '')) not in open_market_ids]
    
    logging.info(f"Filtered {len(all_potential)} interesting markets. After excluding already bought ones, {len(markets)} remaining. Processing top 10...")
    
    # 3. 推理與印出
    for idx, row in enumerate(markets[:10]):
        q = row['question']
        cat = row['category']
        
        print(f"\n{'='*50}\n[Event {idx+1}] {q}")
        
        # 抓取情報
        context = researcher.gather_intelligence(q)
        print(">> Network Context Gathered.")
        
        # Agent 分析
        print(">> Hermes is analyzing via Debate...")
        result = agent.analyze_event_debate(q, cat, context)
        
        if result:
            print("\n--- HERMES VERDICT ---")
            print(f"Reasoning: {result.get('reasoning')}")
            
            true_prob = float(result.get('probability', 0))
            print(f"Agent True Probability: {true_prob*100:.1f}%")
            
            # 對比盤口
            outcomes = row['outcomes'] if isinstance(row['outcomes'], list) else []
            prices = row['prices'] if isinstance(row['prices'], list) else []
            print(f"Current Polymarket Prices:")
            for i in range(min(len(outcomes), len(prices))):
                 print(f"  - {outcomes[i]}: {float(prices[i])*100:.1f}%")
                 
            # 讓 Tracker 進行評估計算並決定是否記錄
            tracker.evaluate_and_log(
                market_id=row.get('id', 'unknown'),
                question=q,
                predicted_prob=true_prob,
                prices=prices,
                outcomes=outcomes,
                context=context,
                reasoning=result.get('reasoning', ''),
                edge_threshold=0.05 # 5%的定價誤差閥值
            )
        else:
            print(">> Agent failed to process this event.")
            
        time.sleep(2) # 避免 Rate limit
        
    print("\n--- 模擬統計報告 ---")
    tracker.show_stats()
    
def daemon_loop():
    from reflection_engine import run_reflection_cycle
    while True:
        logging.info("=== 啟動每小時掃描與反思循環 ===")
        try:
            # 1. 結算舊單與提取反思 (Reflection)
            run_reflection_cycle()
            
            # 2. 進行新一輪的掃描與下單
            main()
        except Exception as e:
            logging.error(f"循環中發生錯誤: {e}")
            
        logging.info("目前循環結束。系統將進入待命 60 分鐘...")
        time.sleep(3600)  # 休眠 1 小時

if __name__ == "__main__":
    try:
        daemon_loop()
    except KeyboardInterrupt:
        logging.info("使用者強制停止。")
    finally:
        input("\n按下 Enter 鍵結束程式...")
