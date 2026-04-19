import logging
import sqlite3
import requests
import time
from agent import HermesAgent
from tracker import PaperTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_market_resolution(market_id):
    """查詢單一市場的狀態，回傳是否結算與最終獲勝的結果"""
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            m = r.json()
            if m.get("closed") == True:
                # 判斷獲勝者
                outcomes = m.get('outcomes', [])
                if isinstance(outcomes, str):
                    import json
                    outcomes = json.loads(outcomes)
                
                prices = m.get('outcomePrices', [])
                if isinstance(prices, str):
                    import json
                    prices = json.loads(prices)
                
                # 結算時，獲勝的那一方 price 通常會變成 1.0 (或非常接近 1)
                winning_outcome = None
                for idx, p in enumerate(prices):
                    try:
                        if float(p) >= 0.99:
                            winning_outcome = outcomes[idx]
                    except:
                        pass
                
                return True, winning_outcome
        return False, None
    except Exception as e:
        logging.error(f"Error checking market {market_id}: {e}")
        return False, None

def run_reflection_cycle():
    logging.info("Starting Reflection Cycle...")
    # 初始化 Tracker 以確保 DB Migration 已執行
    _ = PaperTracker()
    
    conn = sqlite3.connect("paper_trading.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, market_id, question, predicted_prob, action, context_at_time, reasoning, trade_size, market_price FROM paper_trades WHERE status = 'OPEN'")
    open_trades = cursor.fetchall()
    
    agent = HermesAgent(model_name="gemma", api_url="http://127.0.0.1:11434/api/chat")
    
    for trade in open_trades:
        trade_id, m_id, q, prob, action, ctx, reason, trade_size, price_paid = trade
        
        is_closed, winner = check_market_resolution(m_id)
        if is_closed and winner is not None:
            logging.info(f"Market [{q}] resolved! Winner: {winner}")
            
            # 結算損益 (Realized PnL & Payout)
            payout = 0
            if action == 'BUY YES' and winner.lower() == 'yes':
                payout = (trade_size / price_paid) * 1.0
            elif action == 'BUY NO' and winner.lower() == 'no':
                payout = (trade_size / price_paid) * 1.0
            
            realized_pnl = payout - trade_size
            
            # 將獎金發派回資金池
            if payout > 0:
                cursor.execute("SELECT balance FROM portfolio ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                current_balance = float(row[0]) if row else 10000.0
                new_balance = current_balance + payout
                from datetime import datetime
                cursor.execute("INSERT INTO portfolio (timestamp, balance) VALUES (?, ?)", (datetime.now().isoformat(), new_balance))
            
            # 判斷是否預測正確
            actual_yes = 1 if winner.lower() == 'yes' else 0
            
            # 使用 LLM 反思
            reflection_prompt = (
                f"Event: {q}\n"
                f"Context at the time of prediction:\n{ctx}\n\n"
                f"Your Reasoning at the time:\n{reason}\n"
                f"Your Predicted Probability for 'Yes': {prob*100}%\n"
                f"Actual Outcome: The event resolved as '{winner}'.\n\n"
                f"Provide a short, critical 'lesson learned' stating WHY your reasoning was flawed or correct. Focus on cognitive bias, over-indexing on news, or missed clues in the context."
            )
            
            sys_p = "You are an AI learning system. You extract universal lessons from failed or successful predictions to avoid future mistakes. Answer directly with the key lesson, no JSON format required here."
            payload = {
                "model": agent.model_name,
                "messages": [
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": reflection_prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.3}
            }
            
            try:
                response = requests.post(agent.api_url, json=payload, timeout=60)
                lesson = response.json().get("message", {}).get("content", "").strip()
                
                # 儲存 Reflection 與真實損益
                cursor.execute("INSERT INTO lessons_learned (timestamp, category, lesson) VALUES (datetime('now'), 'Prediction', ?)", (lesson,))
                cursor.execute("UPDATE paper_trades SET status = 'CLOSED', realized_pnl = ? WHERE id = ?", (realized_pnl, trade_id))
                conn.commit()
                logging.info(f"Lesson extracted and saved: {lesson[:50]}...")
            except Exception as e:
                logging.error(f"Failed to generate reflection: {e}")
                
        time.sleep(1) # API Rate limit
        
    conn.close()
    logging.info("Reflection Cycle Completed.")

if __name__ == "__main__":
    run_reflection_cycle()
