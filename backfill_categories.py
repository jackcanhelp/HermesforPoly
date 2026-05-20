"""One-time (idempotent) migration: assign real categories to historical rows.

The Gamma API returned empty / "Unknown" categories for every market, so the
52 resolved trades and their calibration records all landed in one bucket. That
made per-category calibration and lesson retrieval useless. This reclassifies
every paper_trades and calibration_data row from its question text via
categorizer.classify_market, so the existing history immediately feeds the
per-category feedback loops.

Note: lessons_learned has no question column and cannot be reliably
reclassified. Future lessons inherit the corrected paper_trades.market_category
at write time (reflection_engine reads it when a trade resolves).

Safe to re-run: rows already holding a real category are left untouched.
"""

import sqlite3
import os
from categorizer import classify_market

db = os.path.join(os.getenv("DATA_DIR", "."), "paper_trading.db")
conn = sqlite3.connect(db)
cur = conn.cursor()


def reclassify(table, cat_col):
    cur.execute(f"SELECT id, question, {cat_col} FROM {table}")
    rows = cur.fetchall()
    changed = 0
    for rid, question, old in rows:
        new = classify_market(question, old)
        if new != old:
            cur.execute(f"UPDATE {table} SET {cat_col} = ? WHERE id = ?", (new, rid))
            changed += 1
    return len(rows), changed


for table, col in [("paper_trades", "market_category"), ("calibration_data", "category")]:
    total, changed = reclassify(table, col)
    print(f"{table}: {changed}/{total} rows reclassified")

conn.commit()

print("\nCategory distribution after backfill:")
for table, col in [("paper_trades", "market_category"), ("calibration_data", "category")]:
    cur.execute(f"SELECT {col}, COUNT(*) FROM {table} GROUP BY {col} ORDER BY COUNT(*) DESC")
    print(f"\n  {table}:")
    for c, n in cur.fetchall():
        print(f"    {c}: {n}")

conn.close()
