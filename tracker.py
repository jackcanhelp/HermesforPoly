import sqlite3
import os
import logging
from datetime import datetime
from notifier import send_telegram_alert

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PaperTracker:
    def __init__(self, db_path="paper_trading.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                market_id TEXT,
                question TEXT,
                predicted_prob REAL,
                market_price REAL,
                action TEXT,
                ev REAL,
                status TEXT,
                context_at_time TEXT,
                reasoning TEXT,
                kelly_fraction REAL
            )
        ''')
        # 改良的 Migration (若有缺少欄位則嘗試補上)
        try: cursor.execute("ALTER TABLE paper_trades ADD COLUMN context_at_time TEXT")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE paper_trades ADD COLUMN reasoning TEXT")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE paper_trades ADD COLUMN kelly_fraction REAL DEFAULT 0.0")
        except sqlite3.OperationalError: pass
            
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lessons_learned (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                category TEXT,
                lesson TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def get_open_market_ids(self):
        """獲取目前已經持有未平倉部位的 Market ID，避免重複下單與重複計算期望值"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT market_id FROM paper_trades WHERE status = 'OPEN'")
            rows = cursor.fetchall()
            conn.close()
            return [str(row[0]) for row in rows]
        except Exception as e:
            logging.error(f"Error fetching open markets: {e}")
            return []

    def evaluate_and_log(self, market_id, question, predicted_prob, prices, outcomes, context, reasoning, edge_threshold=0.05):
        """
        評估是否有足夠定價誤差 (Expected Value > threshold)，若有則寫入資料庫
        prices: list of floats [Yes_price, No_price] 
        outcomes: list of strings ['Yes', 'No']
        """
        if not prices or len(prices) == 0:
            return

        yes_idx = None
        for i, out in enumerate(outcomes):
            if isinstance(out, str) and out.lower() == 'yes':
                yes_idx = i
                break
        
        # 如果找不到明確的 Yes 選項，預設取第一個
        if yes_idx is None:
            yes_idx = 0

        try:
            market_price = float(prices[yes_idx])
            
            # 計算期望值: 假設買入 1 股 Yes (成本為 market_price)，獲勝拿 1 元
            # EV = (勝率 * 1) - 成本
            ev_yes = predicted_prob - market_price
            
            # 反之，若看跌 (買 No) 的期望值
            # no_market_price = 1 - market_price (簡單假設) 或是實際 prices 的值
            ev_no = (1 - predicted_prob) - (1 - market_price)
            
            action = None
            ev = 0.0
            price_paid = 0.0
            f = 0
            
            if ev_yes > edge_threshold:
                action = "BUY YES"
                ev = ev_yes
                price_paid = market_price
                # Kelly Criterion for binary options: f = (p - c) / (1 - c)
                f = (predicted_prob - market_price) / (1 - market_price) if market_price < 1 else 0
                
            elif ev_no > edge_threshold:
                action = "BUY NO"
                ev = ev_no
                price_paid = 1 - market_price
                p_no = 1 - predicted_prob
                c_no = 1 - market_price
                f = (p_no - c_no) / (1 - c_no) if c_no < 1 else 0
                
            # 套用 Fractional Kelly (保守起見，只下註建議的四分之一) + 上限 15%
            fractional_kelly = min(max(f * 0.25, 0.0), 0.15)
                
            if action:
                logging.info(f"*** FOUND EDGE! Action: {action} on [{question}] | EV: {ev:.3f} | Kelly: {fractional_kelly*100:.1f}% ***")
                self._log_trade(market_id, question, predicted_prob, price_paid, action, ev, context, reasoning, fractional_kelly)
            else:
                logging.info(f"No sufficient edge. EV_Yes: {ev_yes:.3f}, EV_No: {ev_no:.3f}")
                
        except Exception as e:
            logging.error(f"Failed to evaluate market EV: {e}")

    def _log_trade(self, market_id, question, predicted_prob, market_price, action, ev, context, reasoning, kelly_fraction):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO paper_trades (timestamp, market_id, question, predicted_prob, market_price, action, ev, status, context_at_time, reasoning, kelly_fraction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, str(market_id), question, predicted_prob, market_price, action, ev, "OPEN", context, reasoning, kelly_fraction))
        conn.commit()
        conn.close()
        
        # 發送推播
        alert_msg = (
            f"🎯 <b>Polymarket 機會 (Multi-Agent V2)</b>\n\n"
            f"<b>市場:</b> {question}\n"
            f"<b>Agent 預測勝率:</b> {predicted_prob*100:.1f}%\n"
            f"<b>市場價格:</b> ${market_price:.3f}\n"
            f"-----------------------\n"
            f"👉 <b>建議: {action}</b>\n"
            f"📈 <b>期望值 (EV): {ev*100:.1f}%</b>\n"
            f"💰 <b>防爆倉凱利部位: 總資金的 {kelly_fraction*100:.1f}%</b>"
        )
        send_telegram_alert(alert_msg)
        
    def show_stats(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM paper_trades")
        count = cursor.fetchone()[0]
        
        # 簡單 PnL 模擬如果都押 100 美元
        cursor.execute("SELECT ev FROM paper_trades")
        evs = cursor.fetchall()
        total_ev = sum([x[0] * 100 for x in evs])
        
        logging.info(f"Total simulated trades logged: {count}")
        logging.info(f"Total Expected Profit (assuming $100 per trade): ${total_ev:.2f}")
        conn.close()

if __name__ == "__main__":
    tracker = PaperTracker()
    tracker.show_stats()
