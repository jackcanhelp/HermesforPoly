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
                kelly_fraction REAL,
                trade_size REAL DEFAULT 0.0,
                realized_pnl REAL DEFAULT 0.0
            )
        ''')
        
        # Portfolio table for Bankroll Tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                balance REAL,
                total_equity REAL DEFAULT 10000.0
            )
        ''')
        
        # Initialize Bankroll if empty
        cursor.execute("SELECT COUNT(*) FROM portfolio")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO portfolio (timestamp, balance, total_equity) VALUES (?, ?, ?)", (datetime.now().isoformat(), 10000.0, 10000.0))
            
        # 改良的 Migration
        try: cursor.execute("ALTER TABLE portfolio ADD COLUMN total_equity REAL DEFAULT 10000.0")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE paper_trades ADD COLUMN context_at_time TEXT")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE paper_trades ADD COLUMN reasoning TEXT")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE paper_trades ADD COLUMN kelly_fraction REAL DEFAULT 0.0")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE paper_trades ADD COLUMN trade_size REAL DEFAULT 0.0")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE paper_trades ADD COLUMN realized_pnl REAL DEFAULT 0.0")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE paper_trades ADD COLUMN last_mtm_prob REAL DEFAULT NULL")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE paper_trades ADD COLUMN market_category TEXT")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE lessons_learned ADD COLUMN market_category TEXT")
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

    def evaluate_and_log(self, market_id, question, predicted_prob, prices, outcomes, context, reasoning, category="Unknown", edge_threshold=0.05):
        if not prices or len(prices) == 0:
            return

        yes_idx = None
        for i, out in enumerate(outcomes):
            if isinstance(out, str) and out.lower() == 'yes':
                yes_idx = i
                break
        
        if yes_idx is None: yes_idx = 0

        try:
            market_price = float(prices[yes_idx])
            ev_yes = predicted_prob - market_price
            ev_no = (1 - predicted_prob) - (1 - market_price)
            
            action = None
            ev = 0.0
            price_paid = 0.0
            f = 0
            
            if ev_yes > edge_threshold:
                action = "BUY YES"
                ev = ev_yes
                price_paid = market_price
                f = (predicted_prob - market_price) / (1 - market_price) if market_price < 1 else 0
                
            elif ev_no > edge_threshold:
                action = "BUY NO"
                ev = ev_no
                price_paid = 1 - market_price
                p_no = 1 - predicted_prob
                c_no = 1 - market_price
                f = (p_no - c_no) / (1 - c_no) if c_no < 1 else 0
                
            fractional_kelly = min(max(f * 0.25, 0.0), 0.15)
                
            if action:
                logging.info(f"*** FOUND EDGE! Action: {action} on [{question}] | EV: {ev:.3f} | Kelly: {fractional_kelly*100:.1f}% ***")
                self._log_trade(market_id, question, predicted_prob, price_paid, action, ev, context, reasoning, category, fractional_kelly)
            else:
                logging.info(f"No sufficient edge. EV_Yes: {ev_yes:.3f}, EV_No: {ev_no:.3f}")
                
        except Exception as e:
            logging.error(f"Failed to evaluate market EV: {e}")

    def _log_trade(self, market_id, question, predicted_prob, market_price, action, ev, context, reasoning, category, kelly_fraction):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get Current Balance & Equity
        cursor.execute("SELECT balance, total_equity FROM portfolio ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        current_balance = float(row[0]) if row else 10000.0
        current_equity = float(row[1]) if row else 10000.0
        
        trade_size = current_balance * kelly_fraction
        new_balance = current_balance - trade_size
        
        # Deduct from Portfolio (Equity remains the same when opening a fair-value trade)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO portfolio (timestamp, balance, total_equity) VALUES (?, ?, ?)", (timestamp, new_balance, current_equity))
        
        # Log Trade
        cursor.execute('''
            INSERT INTO paper_trades (timestamp, market_id, question, predicted_prob, market_price, action, ev, status, context_at_time, reasoning, kelly_fraction, trade_size, market_category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, str(market_id), question, predicted_prob, market_price, action, ev, "OPEN", context, reasoning, kelly_fraction, trade_size, category))
        
        conn.commit()
        conn.close()
        
        # 發送推播
        alert_msg = (
            f"🎯 <b>Polymarket 機會 (Bankroll V4)</b>\n\n"
            f"<b>市場:</b> {question}\n"
            f"<b>決策:</b> {action} @ ${market_price:.3f}\n"
            f"<b>勝率預測:</b> {predicted_prob*100:.1f}%\n"
            f"<b>理論EV:</b> {ev*100:.1f}%\n"
            f"-----------------------\n"
            f"💵 <b>虛擬本金庫扣款下注:</b>\n"
            f"- 下注比例: {kelly_fraction*100:.1f}%\n"
            f"- 下注金額: ${trade_size:.2f} USD\n"
            f"- 扣款後餘額: ${new_balance:.2f} USD"
        )
        send_telegram_alert(alert_msg)
        
    def show_stats(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM paper_trades")
        count = cursor.fetchone()[0]
        
        cursor.execute("SELECT balance FROM portfolio ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        balance = row[0] if row else 10000.0
        
        logging.info(f"Total simulated trades logged: {count}")
        logging.info(f"Current Virtual Bankroll: ${balance:.2f}")
        conn.close()

if __name__ == "__main__":
    tracker = PaperTracker()
    tracker.show_stats()
