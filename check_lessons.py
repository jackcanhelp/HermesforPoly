import sqlite3
import os

db_path = os.path.join(os.getenv("DATA_DIR", "."), "paper_trading.db")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT category, COUNT(*) FROM lessons_learned GROUP BY category")
print("Lessons by category:", cursor.fetchall())
cursor.execute("SELECT timestamp, category, lesson FROM lessons_learned ORDER BY id DESC LIMIT 3")
print("Latest 3 lessons:")
for row in cursor.fetchall():
    print(row)
conn.close()
