"""SQLite repository for analysed calls, plus the SQL behind the ops report."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY,
    intent TEXT NOT NULL,
    intent_confidence REAL,
    sentiment REAL,
    transfer_risk REAL,
    transferred INTEGER,          -- filled in later when the outcome is known (NULL until then)
    summary TEXT,
    masked_text TEXT
);
"""

REPORT_SQL = """
SELECT intent,
       COUNT(*)                         AS calls,
       ROUND(AVG(sentiment), 3)         AS avg_sentiment,
       ROUND(AVG(transfer_risk), 3)     AS avg_predicted_risk,
       ROUND(AVG(transferred), 3)       AS actual_transfer_rate,
       SUM(transfer_risk >= :threshold) AS flagged_for_supervisor
FROM calls
GROUP BY intent
ORDER BY avg_predicted_risk DESC;
"""


class CallStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def add(self, call_id: str, result: dict, transferred: int | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO calls VALUES (:id, :intent, :intent_confidence, :sentiment, :transfer_risk, "
            ":transferred, :summary, :masked_text)", {"id": call_id, "transferred": transferred, **result})
        self.conn.commit()

    def set_outcome(self, call_id: str, transferred: int) -> bool:
        cur = self.conn.execute("UPDATE calls SET transferred = ? WHERE id = ?", (transferred, call_id))
        self.conn.commit()
        return cur.rowcount == 1

    def get(self, call_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()
        return dict(row) if row else None

    def report(self, threshold: float = 0.5) -> list[dict]:
        return [dict(r) for r in self.conn.execute(REPORT_SQL, {"threshold": threshold})]
