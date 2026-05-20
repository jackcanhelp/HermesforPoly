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
    "market", "price", "event", "happen", "occur",
    "reach", "total", "united", "states", "least",
}

class PaperTracker:
    def __init__(self, db_path=None):
        db_path = db_path or _DB_PATH
        self.db_path = db_path
        self._init_db()
        self._skill_cache = None  # lazily-built skill report, used by calibrate_probability

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
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
                "ALTER TABLE paper_trades ADD COLUMN raw_prob REAL",
            ]
            for sql in migrations:
                try:
                    cursor.execute(sql)
                except sqlite3.OperationalError:
                    pass

    def get_open_market_ids(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT market_id FROM paper_trades WHERE status = 'OPEN'")
                rows = cursor.fetchall()
                return [str(row[0]) for row in rows]
        except Exception as e:
            logging.error(f"Error fetching open markets: {e}")
            return []

    def get_open_position_summary(self):
        """Returns dict of {question: info} for all OPEN trades"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT question, action, trade_size, predicted_prob FROM paper_trades WHERE status = 'OPEN'")
                rows = cursor.fetchall()
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
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT market_id, question, predicted_prob, market_price, market_category FROM paper_trades WHERE id = ?",
                    (trade_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return
                market_id, question, predicted_prob, market_price, category = row
                brier_contribution = (predicted_prob - resolved_outcome) ** 2
                cursor.execute('''
                    INSERT INTO calibration_data (timestamp, market_id, question, predicted_prob, market_price, resolved_outcome, brier_contribution, category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (datetime.now().isoformat(), market_id, question, predicted_prob, market_price, resolved_outcome, brier_contribution, category))
                logging.info(f"Calibration recorded: Brier={brier_contribution:.4f} (pred={predicted_prob:.2f}, actual={resolved_outcome})")
        except Exception as e:
            logging.error(f"Error recording calibration data: {e}")

    def get_calibration_stats(self, category=None, lookback_n=50):
        """Returns calibration stats dict: brier_score, win_rate, mean_pred, mean_actual, bias_direction, bias_amount, sample_size."""
        default = {"brier_score": 0.25, "win_rate": 0.5, "mean_pred": 0.5, "mean_actual": 0.5,
                   "bias_direction": "UNKNOWN", "bias_amount": 0.0, "sample_size": 0}
        try:
            with sqlite3.connect(self.db_path) as conn:
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

    def get_skill_report(self):
        """Honest skill check: does the agent actually beat zero-effort baselines?

        Compares the agent's Brier score against (a) simply trusting the current
        market price and (b) always guessing the base rate. A value-betting agent
        that cannot beat the market price has NEGATIVE alpha -- its disagreements
        with the market lose money, so it should not be betting against it.

        Computed directly from CLOSED paper_trades (self-consistent, independent
        of any calibration_data quirks). The YES outcome and YES price are
        reconstructed from action + realized_pnl. Returns a dict with an
        'overall' summary and a 'by_category' breakdown.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT predicted_prob, market_price, action, realized_pnl, trade_size, market_category "
                    "FROM paper_trades WHERE status = 'CLOSED'"
                )
                rows = cursor.fetchall()
        except Exception as e:
            logging.error(f"Error building skill report: {e}")
            return {}

        def _agg(records):
            agent_sq = market_sq = 0.0
            outcomes = []
            for prob, mp, action, pnl, size, _cat in records:
                if None in (prob, mp, pnl, size):
                    continue
                is_win = pnl > -size
                if (action or "").upper() == "BUY YES":
                    actual = 1 if is_win else 0
                    yes_price = mp
                else:  # BUY NO -> stored market_price is the NO price
                    actual = 0 if is_win else 1
                    yes_price = 1.0 - mp
                agent_sq += (prob - actual) ** 2
                market_sq += (yes_price - actual) ** 2
                outcomes.append(actual)
            n = len(outcomes)
            if n == 0:
                return None
            base = sum(outcomes) / n
            baserate_sq = sum((base - o) ** 2 for o in outcomes)
            return {
                "n": n,
                "agent_brier": agent_sq / n,
                "market_brier": market_sq / n,
                "baserate": base,
                "baserate_brier": baserate_sq / n,
                "beats_market": (agent_sq / n) < (market_sq / n),
            }

        report = {"overall": _agg(rows)}
        by_cat = {}
        grouped = {}
        for r in rows:
            grouped.setdefault(r[5] or "Other", []).append(r)
        for cat, recs in grouped.items():
            by_cat[cat] = _agg(recs)
        report["by_category"] = by_cat
        return report

    def calibrate_probability(self, raw_prob, market_yes_price, category):
        """Mechanical calibration layer, applied to the Judge's raw probability
        BEFORE measuring edge. Shrinks the prediction toward the market price by
        a weight that grows with how poorly the agent has been calibrated in this
        category:

          - Where the agent's Brier already beats the market (genuine edge), the
            prediction is left almost untouched (w=0.10).
          - Where the agent is systematically worse (e.g. geopolitics longshots),
            it is pulled hard toward the well-calibrated market price (up to 0.85).
          - With too little history (cold start), no shrink -- trust the agent
            until it is proven wrong.

        This fixes overconfidence AND starves the spurious 'edges' that have
        driven losses (a longshot the agent loves but the market prices low gets
        pulled back to the market, so it no longer clears the edge threshold).

        Returns (calibrated_prob, shrink_weight).
        """
        try:
            raw = float(raw_prob)
            mkt = float(market_yes_price)
        except (TypeError, ValueError):
            return raw_prob, 0.0

        if self._skill_cache is None:
            self._skill_cache = self.get_skill_report() or {}

        MIN_CAT = 8
        cat_stats = (self._skill_cache.get("by_category") or {}).get(category)
        overall = self._skill_cache.get("overall")

        def _weight(stats):
            if not stats:
                return None
            ab, mb = stats["agent_brier"], stats["market_brier"]
            if ab <= mb:
                return 0.10  # agent has demonstrated edge here -> barely shrink
            excess = (ab - mb) / max(mb, 0.05)
            return max(0.30, min(0.85, 0.30 + excess * 0.35))

        if cat_stats and cat_stats["n"] >= MIN_CAT:
            w = _weight(cat_stats)
        elif overall and overall["n"] >= 20:
            w = _weight(overall)
        else:
            w = 0.0  # not enough history yet

        if w is None:
            w = 0.0

        cal = (1.0 - w) * raw + w * mkt
        cal = max(0.01, min(0.99, cal))
        return cal, w

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
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM lessons_learned WHERE is_consolidated = 0")
                count = cursor.fetchone()[0]
                return count
        except Exception as e:
            logging.error(f"Error counting unconsolidated lessons: {e}")
            return 0

    def mark_lessons_consolidated(self):
        """Mark all current lessons as consolidated into the rulebook."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE lessons_learned SET is_consolidated = 1 WHERE is_consolidated = 0")
        except Exception as e:
            logging.error(f"Error marking lessons consolidated: {e}")

    def evaluate_and_log(self, market_id, question, predicted_prob, prices, outcomes, context, reasoning,
                         category="Unknown", edge_threshold=0.05, kelly_multiplier=0.25, cluster_count=1, roi_threshold=0.10):
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

            # --- Mechanical calibration layer: shrink the Judge's raw probability
            # toward the (better-calibrated) market price before measuring edge. ---
            raw_prob = predicted_prob
            predicted_prob, shrink_w = self.calibrate_probability(raw_prob, market_price, category)
            if shrink_w > 0:
                logging.info(f"Calibrated prob: raw={raw_prob:.2f} -> {predicted_prob:.2f} "
                             f"(shrink {shrink_w:.0%} toward market {market_price:.2f}, cat={category})")

            ev_yes = predicted_prob - market_price
            ev_no = (1 - predicted_prob) - (1 - market_price)

            action = None
            ev = 0.0
            price_paid = 0.0
            f = 0
            
            roi_yes = (ev_yes / market_price) if market_price > 0 else 0
            roi_no = (ev_no / (1 - market_price)) if (1 - market_price) > 0 else 0

            if ev_yes > edge_threshold and roi_yes > roi_threshold:
                action = "BUY YES"
                ev = ev_yes
                price_paid = market_price
                f = (predicted_prob - market_price) / (1 - market_price) if market_price < 1 else 0

            elif ev_no > edge_threshold and roi_no > roi_threshold:
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
                logging.info(f"*** FOUND EDGE! Action: {action} on [{question}] | EV: {ev:.3f} | ROI: {ev/price_paid*100:.1f}% | Kelly: {fractional_kelly*100:.1f}% (cluster_count={cluster_count}) ***")
                self._log_trade(market_id, question, predicted_prob, price_paid, action, ev, context, reasoning, category, fractional_kelly, raw_prob)
            else:
                logging.info(f"No sufficient edge/ROI. EV_Yes: {ev_yes:.3f} (ROI: {roi_yes*100:.1f}%), EV_No: {ev_no:.3f} (ROI: {roi_no*100:.1f}%)")

        except Exception as e:
            logging.error(f"Failed to evaluate market EV: {e}")

    def _log_trade(self, market_id, question, predicted_prob, market_price, action, ev, context, reasoning, category, kelly_fraction, raw_prob=None):
        with sqlite3.connect(self.db_path) as conn:
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
                INSERT INTO paper_trades (timestamp, market_id, question, predicted_prob, market_price, action, ev, status, context_at_time, reasoning, kelly_fraction, trade_size, market_category, raw_prob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, str(market_id), question, predicted_prob, market_price, action, ev, "OPEN", context, reasoning, kelly_fraction, trade_size, category, raw_prob))

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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM paper_trades")
            count = cursor.fetchone()[0]

            cursor.execute("SELECT balance FROM portfolio ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            balance = row[0] if row else 10000.0

            logging.info(f"Total simulated trades logged: {count}")
            logging.info(f"Current Virtual Bankroll: ${balance:.2f}")

if __name__ == "__main__":
    tracker = PaperTracker()
    tracker.show_stats()

    rep = tracker.get_skill_report()
    ov = rep.get("overall")
    if ov:
        print("\n=== SKILL REPORT (agent vs zero-effort baselines) ===")
        print(f"resolved trades : {ov['n']}")
        print(f"agent  Brier    : {ov['agent_brier']:.4f}")
        print(f"market Brier    : {ov['market_brier']:.4f}   <- just trust the market price")
        print(f"base-rate Brier : {ov['baserate_brier']:.4f}   (always guess {ov['baserate']:.1%} YES)")
        verdict = "YES (positive alpha)" if ov["beats_market"] else "NO -- worse than the market"
        print(f"agent beats market? {verdict}")
        print("\n--- by category (sorted by sample size) ---")
        items = sorted(rep.get("by_category", {}).items(),
                       key=lambda kv: -(kv[1]["n"] if kv[1] else 0))
        for cat, s in items:
            if not s:
                continue
            flag = "OK " if s["beats_market"] else "NEG"
            print(f"[{flag}] {cat:13s} n={s['n']:2d}  agent={s['agent_brier']:.3f}  "
                  f"market={s['market_brier']:.3f}  base-rate={s['baserate']:.0%}")
