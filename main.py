import time
import logging
from collections import Counter
from datetime import datetime, timezone
from dotenv import load_dotenv
from poly_scanner import fetch_active_markets
from researcher import PolyResearcher
from agent import HermesAgent
from tracker import PaperTracker

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_potential_markets(df):
    potential_markets = []
    now = datetime.now(timezone.utc)
    for _, row in df.iterrows():
        end_date = row.get('endDate')
        if end_date is not None:
            import pandas as pd
            if pd.notna(end_date):
                days_to_end = (end_date - now).days
                if days_to_end > 14 or days_to_end < 0:
                    continue

        outcomes = row.get('outcomes', [])
        if not isinstance(outcomes, list) or len(outcomes) != 2:
            continue
        if "Yes" not in outcomes or "No" not in outcomes:
            continue

        prices = row['prices'] if isinstance(row['prices'], list) else []
        is_interesting = False
        for p in prices:
            try:
                if 0.05 < float(p) < 0.95:
                    is_interesting = True
            except:
                pass

        if is_interesting and float(row.get('liquidity', 0)) > 100:
            potential_markets.append(row)

    return potential_markets

def main():
    researcher = PolyResearcher()
    sentiment_agent = HermesAgent(model_name="llama3.2")
    judge_agent = HermesAgent(model_name="llama3.1")
    tracker = PaperTracker()

    # Dynamic edge threshold and Kelly multiplier based on calibration history
    edge_threshold, kelly_multiplier = tracker.get_dynamic_params()
    logging.info(f"Active params: edge_threshold={edge_threshold:.3f}, kelly_multiplier={kelly_multiplier:.2f}")

    df = fetch_active_markets(500)
    if df is None:
        return

    open_market_ids = tracker.get_open_market_ids()
    all_potential = filter_potential_markets(df)

    markets = [m for m in all_potential if str(m.get('id', '')) not in open_market_ids]

    # Deduplicate by question text within this API response
    processed_questions = set()
    unique_markets = []
    for m in markets:
        q = m.get('question', '')
        if q not in processed_questions:
            unique_markets.append(m)
            processed_questions.add(q)

    # Cap at 3 markets per category to prevent correlated batch bets
    category_counts = Counter()
    capped_markets = []
    for m in unique_markets:
        cat = m.get('category') or 'Unknown'
        if category_counts[cat] < 3:
            capped_markets.append(m)
            category_counts[cat] += 1
    unique_markets = capped_markets

    logging.info(f"Filtered {len(all_potential)} interesting markets → {len(unique_markets)} after dedup and category cap. Processing top 30...")

    # Snapshot current open position questions for correlation detection
    open_position_summary = tracker.get_open_position_summary()
    open_questions = list(open_position_summary.keys())

    for idx, row in enumerate(unique_markets[:30]):
        q = row['question']
        cat = row.get('category') or 'Unknown'

        print(f"\n{'='*50}\n[Event {idx+1}] {q}")

        # Correlation guard: block if 3+ open bets are already correlated with this question
        cluster_name, cluster_count = tracker.detect_topic_cluster(q, open_questions)
        if cluster_count >= 3:
            logging.info(f"[CORRELATION BLOCK] {cluster_count} open bets already related to '{cluster_name}'. Skipping: {q}")
            continue

        print(">> Auto-Researcher is gathering news...")
        context = researcher.gather_intelligence(q)

        print(">> Social Sentinel is analyzing Reddit momentum...")
        social_context = researcher.gather_social_sentiment(q)
        sentiment_report = sentiment_agent.analyze_social_sentiment(q, social_context)

        if "No real-time context found" in context or not context.strip():
            print(">> [Circuit Breaker] No intelligence found. Refusing to guess.")
            continue

        print(">> Hermes is analyzing via Debate & Sentiment...")
        yes_idx = next((i for i, o in enumerate(outcomes) if isinstance(o, str) and o.lower() == 'yes'), 0)
        market_yes_price = float(prices[yes_idx]) if prices else 0.5
        result = judge_agent.analyze_event_debate(q, cat, context, sentiment_report, market_yes_price)

        if result:
            true_prob = float(result.get('probability', 0))
            if true_prob > 1.0:
                true_prob = true_prob / 100.0

            full_context_to_log = f"{context}\n\n[Social Sentiment Report]:\n{sentiment_report}"

            outcomes = row['outcomes'] if isinstance(row['outcomes'], list) else []
            prices = row['prices'] if isinstance(row['prices'], list) else []
            print(f"Current Polymarket Prices:")
            for i in range(min(len(outcomes), len(prices))):
                print(f"  - {outcomes[i]}: {float(prices[i])*100:.1f}%")

            tracker.evaluate_and_log(
                market_id=row.get('id', 'unknown'),
                question=q,
                predicted_prob=true_prob,
                prices=prices,
                outcomes=outcomes,
                context=full_context_to_log,
                reasoning=result.get('reasoning', ''),
                category=cat,
                edge_threshold=edge_threshold,
                kelly_multiplier=kelly_multiplier,
                cluster_count=max(1, cluster_count)
            )

            # Add this question to the in-memory list so subsequent markets in this
            # cycle also benefit from the correlation check
            open_questions.append(q)
        else:
            print(">> Agent failed to process this event.")

        time.sleep(2)  # Polite delay; 429 handling is in _call_llm_with_provider

    print("\n--- 模擬統計報告 ---")
    tracker.show_stats()

def daemon_loop():
    from reflection_engine import run_reflection_cycle
    while True:
        logging.info("=== 啟動每15分鐘掃描與反思循環 ===")
        try:
            run_reflection_cycle()
            main()
            from meta_reflection import consolidate_memory
            consolidate_memory()
        except Exception as e:
            logging.error(f"循環中發生錯誤: {e}")

        logging.info("目前循環結束。系統將進入待命 15 分鐘...")
        time.sleep(900)

if __name__ == "__main__":
    try:
        daemon_loop()
    except KeyboardInterrupt:
        logging.info("使用者強制停止。")
