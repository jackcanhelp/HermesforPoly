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
                
                return True, winning_outcome, None
            else:
                outcomes = m.get('outcomes', [])
                if isinstance(outcomes, str):
                    import json
                    outcomes = json.loads(outcomes)
                prices = m.get('outcomePrices', [])
                if isinstance(prices, str):
                    import json
                    prices = json.loads(prices)
                yes_idx = None
                for idx, out in enumerate(outcomes):
                    if isinstance(out, str) and out.lower() == 'yes':
                        yes_idx = idx
                        break
                if yes_idx is None: yes_idx = 0
                
                try:
                    current_yes_price = float(prices[yes_idx])
                    return False, None, current_yes_price
                except:
                    return False, None, None
        return False, None, None
    except Exception as e:
        logging.error(f"Error checking market {market_id}: {e}")
        return False, None, None

def run_reflection_cycle():
    logging.info("Starting Reflection Cycle...")
    # 初始化 Tracker 以確保 DB Migration 已執行
    _ = PaperTracker()
    
    conn = sqlite3.connect("paper_trading.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, market_id, question, predicted_prob, action, context_at_time, reasoning, trade_size, market_price, last_mtm_prob, market_category FROM paper_trades WHERE status = 'OPEN'")
    open_trades = cursor.fetchall()
    # 使用 Llama 3.1 405B 作為終審與反思法官 (NVIDIA NIM API)
    agent = HermesAgent(model_name="meta/llama-3.1-405b-instruct") 
    for trade in open_trades:
        trade_id, m_id, q, prob, action, ctx, reason, trade_size, price_paid, last_mtm_prob, market_category = trade
        
        is_closed, winner, current_yes_price = check_market_resolution(m_id)
        if is_closed and winner is not None:
            logging.info(f"Market [{q}] resolved! Winner: {winner}")
            
            # 結算損益 (Realized PnL & Payout)
            payout = 0
            if action == 'BUY YES' and winner.lower() == 'yes':
                payout = (trade_size / price_paid) * 1.0
            elif action == 'BUY NO' and winner.lower() == 'no':
                payout = (trade_size / price_paid) * 1.0
            
            realized_pnl = payout - trade_size
            
            # 將獎金發派回資金池，並更新淨值
            cursor.execute("SELECT balance, total_equity FROM portfolio ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            current_balance = float(row[0]) if row else 10000.0
            current_equity = float(row[1]) if row else 10000.0
            
            new_balance = current_balance + payout
            new_equity = current_equity + realized_pnl
            
            from datetime import datetime
            cursor.execute("INSERT INTO portfolio (timestamp, balance, total_equity) VALUES (?, ?, ?)", (datetime.now().isoformat(), new_balance, new_equity))
            
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
            
            try:
                providers_chain = [
                    ("nvidia", "meta/llama-3.1-405b-instruct"),
                    ("groq", "llama-3.3-70b-versatile"),
                    ("ollama", agent.model_name)
                ]
                # 使用 Agent 內建的 _call_llm_with_fallback 方法，自動處理備援邏輯
                lesson = agent._call_llm_with_fallback(sys_p, reflection_prompt, json_mode=False, providers=providers_chain)
                
                if lesson:
                    # 儲存 Reflection 與真實損益
                    cursor.execute("INSERT INTO lessons_learned (timestamp, category, lesson, market_category) VALUES (datetime('now'), 'Prediction', ?, ?)", (lesson, market_category))
                    cursor.execute("UPDATE paper_trades SET status = 'CLOSED', realized_pnl = ? WHERE id = ?", (realized_pnl, trade_id))
                    conn.commit()
                    logging.info(f"Lesson extracted and saved: {lesson[:50]}...")
                else:
                    logging.warning("LLM returned empty lesson.")
            except Exception as e:
                logging.error(f"Failed to generate reflection: {e}")
                
        elif not is_closed and current_yes_price is not None:
            # MTM Reflection (盯市反思)
            current_asset_price = current_yes_price if action == 'BUY YES' else (1.0 - current_yes_price)
            ref_prob = last_mtm_prob if last_mtm_prob is not None else price_paid
            if abs(current_asset_price - ref_prob) >= 0.25:
                logging.info(f"Market [{q}] shifted! Expected: {prob:.2f}, Last Ref: {ref_prob:.2f}, Current: {current_asset_price:.2f}. Triggering MTM Reflection.")
                
                mtm_prompt = (
                    f"Event: {q}\n"
                    f"Context at the time of prediction:\n{ctx}\n\n"
                    f"Your Reasoning at the time:\n{reason}\n"
                    f"Your Predicted Probability for 'Yes': {prob*100}%\n"
                    f"Market Shift: The crowd's consensus probability for your side has moved significantly to {current_asset_price*100}%.\n\n"
                    f"Provide a short, critical 'lesson learned' stating WHY your original reasoning might be diverging from the market. Focus on what new information the market might be digesting or what you might have over/under-weighted."
                )
                
                sys_p = "You are an AI learning system. You extract universal lessons from market probability shifts. Answer directly with the key lesson, no JSON format required here."
                
                try:
                    providers_chain = [
                        ("nvidia", "meta/llama-3.1-405b-instruct"),
                        ("groq", "llama-3.3-70b-versatile"),
                        ("ollama", agent.model_name)
                    ]
                    lesson = agent._call_llm_with_fallback(sys_p, mtm_prompt, json_mode=False, providers=providers_chain)
                    if lesson:
                        cursor.execute("INSERT INTO lessons_learned (timestamp, category, lesson, market_category) VALUES (datetime('now'), 'MTM Reflection', ?, ?)", (lesson, market_category))
                        cursor.execute("UPDATE paper_trades SET last_mtm_prob = ? WHERE id = ?", (current_asset_price, trade_id))
                        conn.commit()
                        logging.info(f"MTM Lesson extracted and saved: {lesson[:50]}...")
                    else:
                        logging.warning("LLM returned empty MTM lesson.")
                except Exception as e:
                    logging.error(f"Failed to generate MTM reflection: {e}")
                    
        time.sleep(1) # API Rate limit
        
    conn.close()
    logging.info("Reflection Cycle Completed.")

if __name__ == "__main__":
    try:
        run_reflection_cycle()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        input("\n按下 Enter 鍵結束程式...")
