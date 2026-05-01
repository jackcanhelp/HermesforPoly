import sqlite3
import os
import logging
from datetime import datetime
from notifier import send_telegram_alert

_DATA_DIR = os.getenv("DATA_DIR", ".")
_DB_PATH = os.path.join(_DATA_DIR, "paper_trading.db")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_STOPWORDS = {
    "will", "the", "this", "that", "with", "from", "have", "been", "they",
    "their", "than", "before", "after", "over", "under", "does", "about",
    "what", "when", "where", "which", "into", "also", "more", "most", "some",
    "would", "could", "should", "during", "while", "between", "against",
    "within", "without", "through", "there", "these", "those", "other",
    "first", "last", "next", "then", "than", "such", "even", "just",
    # Years — appear in almost every question, not meaningful correlators
    "2024", "2025", "2026", "2027", "2028", "2029", "2030",
    # Months
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
    # Generic prediction market boilerplate
    "least", "least", "market", "price", "event", "happen", "occur",
    "reach", "least", "total", "least", "united", "states", "least",
}

class PaperTracker:
    def __init__(self, db_path=None):
        db_path = db_path or _DB_PATH
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

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                balance REAL,
                total_equity REAL DEFAULT 10000.0
            )
        ''')

        cursor.execute("SELECT COUNT(*) FROM portfolio")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO portfolio (timestamp, balance, total_equity) VALUES (?, ?, ?)", (datetime.now().isoformat(), 10000.0, 10000.0))

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lessons_learned (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                category TEXT,
                lesson TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS calibration_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                market_id TEXT,
                question TEXT,
                predicted_prob REAL,
                market_price REAL,
                resolved_outcome INTEGER,
                brier_contribution REAL,
                category TEXT
            )
        ''')

        # Migrations
        migrations = [
            "ALTER TABLE portfolio ADD COLUMN total_equity REAL DEFAULT 10000.0",
            "ALTER TABLE paper_trades ADD COLUMN context_at_time TEXT",
            "ALTER TABLE paper_trades ADD COLUMN reasoning TEXT",
            "ALTER TABLE paper_trades ADD COLUMN kelly_fraction REAL DEFAULT 0.0",
            "ALTER TABLE paper_trades ADD COLUMN trade_size REAL DEFAULT 0.0",
            "ALTER TABLE paper_trades ADD COLUMN realized_pnl REAL DEFAULT 0.0",
            "ALTER TABLE paper_trades ADD COLUMN last_mtm_prob REAL DEFAULT NULL",
            "ALTER TABLE paper_trades ADD COLUMN market_category TEXT",
            "ALTER TABLE lessons_learned ADD COLUMN market_category TEXT",
            "ALTER TABLE lessons_learned ADD COLUMN is_consolidated INTEGER DEFAULT 0",
        ]
        for sql in migrations:
            try:
                cursor.execute(sql)
            except sqlite3.OperationalError:
                pass

        conn.commit()
        conn.close()

    def get_open_market_ids(self):
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

    def get_open_position_summary(self):
        """Returns dict of {question: info} for all OPEN trades"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT question, action, trade_size, predicted_prob FROM paper_trades WHERE status = 'OPEN'")
            rows = cursor.fetchall()
            conn.close()
            return {r[0]: {'action': r[1], 'trade_size': r[2], 'predicted_prob': r[3]} for r in rows}
        except Exception as e:
            logging.error(f"Error fetching open positions: {e}")
            return {}

    @staticmethod
    def _extract_keywords(text):
        words = set()
        for w in text.split():
            cleaned = w.lower().strip("?.,!\"'()[]")
            if len(cleaned) > 4 and cleaned not in _STOPWORDS:
                words.add(cleaned)
        return words

    def detect_topic_cluster(self, question, open_questions):
        """Detect if a new question correlates with existing open positions via keyword overlap.
        Returns (top_keyword, count_of_correlated_open_positions)."""
        q_words = self._extract_keywords(question)

        cluster_matches = {}
        for oq in open_questions:
            oq_words = self._extract_keywords(oq)
            shared = q_words & oq_words
            for word in shared:
                cluster_matches[word] = cluster_matches.get(word, 0) + 1

        if not cluster_matches:
            return "none", 0

        top_cluster = max(cluster_matches, key=cluster_matches.get)
        return top_cluster, cluster_matches[top_cluster]

    def record_resolution(self, trade_id, resolved_outcome):
        """Record a resolved trade into calibration_data for Brier score tracking."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT market_id, question, predicted_prob, market_price, market_category FROM paper_trades WHERE id = ?",
                (trade_id,)
            )
            row = cursor.fetchone()
            if not row:
                conn.close()
                return
            market_id, question, predicted_prob, market_price, category = row
            brier_contribution = (predicted_prob - resolved_outcome) ** 2
            cursor.execute('''
                INSERT INTO calibration_data (timestamp, market_id, question, predicted_prob, market_price, resolved_outcome, brier_contribution, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), market_id, question, predicted_prob, market_price, resolved_outcome, brier_contribution, category))
            conn.commit()
            conn.close()
            logging.info(f"Calibration recorded: Brier={brier_contribution:.4f} (pred={predicted_prob:.2f}, actual={resolved_outcome})")
        except Exception as e:
            logging.error(f"Error recording calibration data: {e}")

    def get_calibration_stats(self, category=None, lookback_n=50):
        """Returns calibration stats dict: brier_score, win_rate, mean_pred, mean_actual, bias_direction, bias_amount, sample_size."""
        default = {"brier_score": 0.25, "win_rate": 0.5, "mean_pred": 0.5, "mean_actual": 0.5,
                   "bias_direction": "UNKNOWN", "bias_amount": 0.0, "sample_size": 0}
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            if category:
                cursor.execute(
                    "SELECT predicted_prob, resolved_outcome, brier_contribution FROM calibration_data WHERE category = ? ORDER BY id DESC LIMIT ?",
                    (category, lookback_n)
                )
            else:
                cursor.execute(
                    "SELECT predicted_prob, resolved_outcome, brier_contribution FROM calibration_data ORDER BY id DESC LIMIT ?",
                    (lookback_n,)
                )
            rows = cursor.fetchall()
            conn.close()

            if not rows or len(rows) < 5:
                default["sample_size"] = len(rows)
                return default

            pred_probs = [r[0] for r in rows if r[0] is not None]
            outcomes = [r[1] for r in rows if r[1] is not None]
            brier_scores = [r[2] for r in rows if r[2] is not None]

            mean_brier = sum(brier_scores) / len(brier_scores)
            mean_pred = sum(pred_probs) / len(pred_probs)
            mean_actual = sum(outcomes) / len(outcomes)
            win_rate = mean_actual

            bias_direction = "OVERCONFIDENT" if mean_pred > mean_actual else "UNDERCONFIDENT"
            bias_amount = abs(mean_pred - mean_actual)

            return {
                "brier_score": mean_brier,
                "win_rate": win_rate,
                "mean_pred": mean_pred,
                "mean_actual": mean_actual,
                "bias_direction": bias_direction,
                "bias_amount": bias_amount,
                "sample_size": len(rows)
            }
        except Exception as e:
            logging.error(f"Error getting calibration stats: {e}")
            return default

    def get_dynamic_params(self):
        """Returns (edge_threshold, kelly_multiplier) based on calibration history.
        Falls back to safe defaults until at least 20 closed trades exist."""
        stats = self.get_calibration_stats(lookback_n=30)
        if stats.get("sample_size", 0) < 20:
            return 0.08, 0.25  # default

        brier = stats.get("brier_score", 0.25)
        win_rate = stats.get("win_rate", 0.5)

        # Worse calibration → higher edge threshold required
        edge_threshold = max(0.06, min(0.20, 0.06 + (brier / 0.25) * 0.09))

        # Low win rate → smaller Kelly
        if win_rate < 0.4:
            kelly_multiplier = 0.10
        elif win_rate < 0.5:
            kelly_multiplier = 0.15
        else:
            kelly_multiplier = 0.25

        logging.info(f"Dynamic params: edge={edge_threshold:.3f}, kelly={kelly_multiplier:.2f} (brier={brier:.3f}, win_rate={win_rate:.2%})")
        return edge_threshold, kelly_multiplier

    def get_unconsolidated_lesson_count(self):
        """Returns number of lessons not yet incorporated into master_rulebook."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM lessons_learned WHERE is_consolidated = 0")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logging.error(f"Error counting unconsolidated lessons: {e}")
            return 0

    def mark_lessons_consolidated(self):
        """Mark all current lessons as consolidated into the rulebook."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE lessons_learned SET is_consolidated = 1 WHERE is_consolidated = 0")
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error marking lessons consolidated: {e}")

    def evaluate_and_log(self, market_id, question, predicted_prob, prices, outcomes, context, reasoning,
                         category="Unknown", edge_threshold=0.05, kelly_multiplier=0.25, cluster_count=1):
        if not prices or len(prices) == 0:
            return

        yes_idx = None
        for i, out in enumerate(outcomes):
            if isinstance(out, str) and out.lower() == 'yes':
                yes_idx = i
                break
        if yes_idx is None:
            yes_idx = 0

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

            # Apply correlation penalty: more correlated open bets → smaller position
            effective_multiplier = kelly_multiplier / max(1, cluster_count)
            fractional_kelly = min(max(f * effective_multiplier, 0.0), 0.15)

            if action:
                logging.info(f"*** FOUND EDGE! Action: {action} on [{question}] | EV: {ev:.3f} | Kelly: {fractional_kelly*100:.1f}% (cluster_count={cluster_count}) ***")
                self._log_trade(market_id, question, predicted_prob, price_paid, action, ev, context, reasoning, category, fractional_kelly)
            else:
                logging.info(f"No sufficient edge. EV_Yes: {ev_yes:.3f}, EV_No: {ev_no:.3f}")

        except Exception as e:
            logging.error(f"Failed to evaluate market EV: {e}")

    def _log_trade(self, market_id, question, predicted_prob, market_price, action, ev, context, reasoning, category, kelly_fraction):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT balance, total_equity FROM portfolio ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        current_balance = float(row[0]) if row else 10000.0
        current_equity = float(row[1]) if row else 10000.0

        trade_size = current_balance * kelly_fraction
        new_balance = current_balance - trade_size

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO portfolio (timestamp, balance, total_equity) VALUES (?, ?, ?)", (timestamp, new_balance, current_equity))

        cursor.execute('''
            INSERT INTO paper_trades (timestamp, market_id, question, predicted_prob, market_price, action, ev, status, context_at_time, reasoning, kelly_fraction, trade_size, market_category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, str(market_id), question, predicted_prob, market_price, action, ev, "OPEN", context, reasoning, kelly_fraction, trade_size, category))

        conn.commit()
        conn.close()

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
