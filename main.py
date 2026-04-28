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
        # 額外防呆：如果 endDate 已經在過去 (days_to_end < 0)，代表該盤口其實已經打完只是 Oracle 還沒官方結算，我們絕對不碰這種「已經發生」的歷史局！
        end_date = row.get('endDate')
        if pd.notna(end_date):
            days_to_end = (end_date - now).days
            if days_to_end > 14 or days_to_end < 0:
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
                
        if is_interesting and float(row.get('liquidity', 0)) > 100: # 降低流動性門檻到 100，挖掘冷門盤口
            potential_markets.append(row)
            
    return potential_markets

def main():
    # 1. 初始化模組 (套用雙雲端混血架構 - 使用 NVIDIA 頂級模型)
    researcher = PolyResearcher(llm_model="nvidia/nemotron-4-340b-instruct") # 負責長文本閱讀
    sentiment_agent = HermesAgent(model_name="mistralai/mixtral-8x22b-instruct-v0.1") # 負責角色與情緒分析
    judge_agent = HermesAgent(model_name="meta/llama-3.1-405b-instruct") # 負責終審裁判
    tracker = PaperTracker()
    
    # 2. 獲取市場 (掃描前 500 個最熱門的市場)
    df = fetch_active_markets(500)
    if df is None:
        return
        
    open_market_ids = tracker.get_open_market_ids()
    all_potential = filter_potential_markets(df)
    
    # 剔除已經在手上的未平倉單 (避免重複計算 EV 與浪費運算)
    markets = [m for m in all_potential if str(m.get('id', '')) not in open_market_ids]
    
    # 過濾同一次 API 回傳中包含相同 question 的重複盤口
    processed_questions = set()
    unique_markets = []
    for m in markets:
        q = m.get('question', '')
        if q not in processed_questions:
            unique_markets.append(m)
            processed_questions.add(q)
            
    logging.info(f"Filtered {len(all_potential)} interesting markets. After excluding already bought ones and duplicates, {len(unique_markets)} remaining. Processing top 30...")
    
    # 3. 推理與印出
    for idx, row in enumerate(unique_markets[:30]):
        q = row['question']
        cat = row['category']
        
        print(f"\n{'='*50}\n[Event {idx+1}] {q}")
        
        # 情報搜集
        print(">> Auto-Researcher is gathering news...")
        context = researcher.gather_intelligence(q)
        
        # 社群情緒雷達 (Phase 5)
        print(">> Social Sentinel is analyzing Reddit momentum...")
        social_context = researcher.gather_social_sentiment(q)
        sentiment_report = sentiment_agent.analyze_social_sentiment(q, social_context)
        
        # 熔斷機制 (Circuit Breaker)：如果情報員真的被擋抓不到資料，一律拒絕讓大腦「腦補瞎猜」
        if "No real-time context found" in context or not context.strip():
            print(">> 🛑 [Circuit Breaker] 無法取得新聞情報，拒絕盲目臆測下單。")
            continue
            
        # Agent 分析 (結合新聞事實與社群情緒)
        print(">> Hermes is analyzing via Debate & Sentiment...")
        result = judge_agent.analyze_event_debate(q, cat, context, sentiment_report)
        
        if result:
            true_prob = float(result.get('probability', 0))
            # 修正 LLM 回傳百分比 (例如 73.5) 的情況，將其轉回 0~1 之間的浮點數
            if true_prob > 1.0:
                true_prob = true_prob / 100.0
                
            full_context_to_log = f"{context}\n\n[Social Sentiment Report]:\n{sentiment_report}"
            
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
                context=full_context_to_log,
                reasoning=result.get('reasoning', ''),
                category=cat,
                edge_threshold=0.08 # 提升為8%的安全定價誤差閥值
            )
        else:
            print(">> Agent failed to process this event.")
            
        time.sleep(15) # 延長至 15 秒，避免觸發 NVIDIA Free Tier 的 API Rate Limit (429 Too Many Requests)
        
    print("\n--- 模擬統計報告 ---")
    tracker.show_stats()
    
def daemon_loop():
    from reflection_engine import run_reflection_cycle
    while True:
        logging.info("=== 啟動每15分鐘掃描與反思循環 ===")
        try:
            # 1. 結算舊單與提取反思 (Reflection)
            run_reflection_cycle()
            
            # 2. 進行新一輪的掃描與下單
            main()
            
            # 3. 進行大腦突觸修剪與分類總結 (Meta-Reflection)
            from meta_reflection import consolidate_memory
            consolidate_memory()
        except Exception as e:
            logging.error(f"循環中發生錯誤: {e}")
            
        logging.info("目前循環結束。系統將進入待命 15 分鐘...")
        time.sleep(900)  # 休眠 15 分鐘

if __name__ == "__main__":
    try:
        daemon_loop()
    except KeyboardInterrupt:
        logging.info("使用者強制停止。")
    finally:
        input("\n按下 Enter 鍵結束程式...")
