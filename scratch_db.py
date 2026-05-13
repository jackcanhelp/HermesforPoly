import sqlite3
import pandas as pd

conn = sqlite3.connect('paper_trading.db')

try:
    df = pd.read_sql_query("SELECT timestamp, realized_pnl FROM paper_trades WHERE status='CLOSED' ORDER BY timestamp ASC", conn)
    print("Total CLOSED trades:", len(df))
    if not df.empty:
        print("\nFirst 5 closed trades:")
        print(df.head(5))
        print("\nLast 5 closed trades:")
        print(df.tail(5))
        
        # Calculate cumulative PNL in first half vs second half
        mid = len(df) // 2
        first_half = df.iloc[:mid]['realized_pnl'].sum()
        second_half = df.iloc[mid:]['realized_pnl'].sum()
        print(f"\nFirst half PNL sum: {first_half:.2f}")
        print(f"Second half PNL sum: {second_half:.2f}")
    
    # Also check tracker history
    print("\n\nLessons learned count:")
    lessons = pd.read_sql_query("SELECT COUNT(*) as count FROM lessons_learned", conn)
    print(lessons)
    
except Exception as e:
    print(e)
finally:
    conn.close()
