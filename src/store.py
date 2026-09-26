"""样本存储（SQLite）：保存每条评价的判断结果，供后续复核/再训练使用。"""
import json
import os
import sqlite3
import threading
import time

from src import config as C

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    text         TEXT NOT NULL,
    source       TEXT DEFAULT 'api',
    pred_label   TEXT,
    pred_id      INTEGER,
    probs        TEXT,
    score        REAL,
    error_prob   REAL,
    severity     TEXT,
    need_process INTEGER DEFAULT 0,
    need_review  INTEGER DEFAULT 0,
    processed    INTEGER DEFAULT 0,
    true_label   TEXT,
    created_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred ON reviews(pred_label);
CREATE INDEX IF NOT EXISTS idx_review ON reviews(need_review);
CREATE INDEX IF NOT EXISTS idx_process ON reviews(need_process);
"""

_INSERT_SQL = """INSERT INTO reviews
    (text, source, pred_label, pred_id, probs, score, error_prob,
     severity, need_process, need_review, created_at)
    VALUES (?,?,?,?,?,?,?,?,?,?,?)"""


class ReviewStore:
    def __init__(self, path=C.DB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # WAL：读写并发更好，避免写时锁库；busy_timeout 减少瞬时锁冲突报错
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def add(self, text, result, source="api"):
        conn = self._conn()
        cur = conn.execute(
            _INSERT_SQL,
            (text, source, result["label"], result["label_id"],
             json.dumps(result["probs"], ensure_ascii=False),
             result["score"], result["error_prob"], result["severity"],
             int(result["need_process"]), int(result["need_review"]),
             time.strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        return cur.lastrowid

    def add_many(self, items):
        """items: list of (text, result, source)。批量入库（单事务，executemany）。

        相比逐条 add（每条 commit），这里用一次 executemany + 一次 commit，
        批量写入吞吐更高，且在 WAL 模式下只占一次写锁。
        """
        if not items:
            return []
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        rows = [
            (text, source, result["label"], result["label_id"],
             json.dumps(result["probs"], ensure_ascii=False),
             result["score"], result["error_prob"], result["severity"],
             int(result["need_process"]), int(result["need_review"]), now)
            for text, result, source in items
        ]
        conn = self._conn()
        conn.executemany(_INSERT_SQL, rows)
        conn.commit()
        # executemany 后 cursor.lastrowid 为 None，改查 last_insert_rowid()；
        # AUTOINCREMENT + 单事务内顺序插入 => rowid 连续，据此还原每个样本的 id
        last = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        first = last - len(rows) + 1
        return list(range(first, last + 1))

    def get(self, review_id):
        row = self._conn().execute("SELECT * FROM reviews WHERE id=?", (review_id,)).fetchone()
        return dict(row) if row else None

    def list(self, limit=50, offset=0, pred_label=None, need_review=None,
             need_process=None, processed=None):
        sql = "SELECT * FROM reviews WHERE 1=1"
        params = []
        if pred_label is not None:
            sql += " AND pred_label=?"; params.append(pred_label)
        if need_review is not None:
            sql += " AND need_review=?"; params.append(int(need_review))
        if need_process is not None:
            sql += " AND need_process=?"; params.append(int(need_process))
        if processed is not None:
            sql += " AND processed=?"; params.append(int(processed))
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        rows = self._conn().execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def human_labeled(self, limit=None):
        """返回已人工复核（回填 true_label）的样本，供主动学习再训练使用。"""
        base = ("SELECT id, text, true_label FROM reviews "
                "WHERE true_label IS NOT NULL AND true_label != '' ")
        if limit is not None:
            base += "ORDER BY id DESC LIMIT ?"
            rows = self._conn().execute(base, (limit,)).fetchall()
        else:
            base += "ORDER BY id DESC"
            rows = self._conn().execute(base).fetchall()
        return [dict(r) for r in rows]

    def update_true_label(self, review_id, true_label):
        conn = self._conn()
        conn.execute("UPDATE reviews SET true_label=?, processed=1 WHERE id=?",
                     (true_label, review_id))
        conn.commit()
        return self.get(review_id)

    def mark_processed(self, review_id):
        conn = self._conn()
        conn.execute("UPDATE reviews SET processed=1 WHERE id=?", (review_id,))
        conn.commit()
        return self.get(review_id)

    def negative_texts(self, limit=20000):
        rows = self._conn().execute(
            "SELECT text FROM reviews WHERE pred_label=? ORDER BY id DESC LIMIT ?",
            (C.ID2LABEL[2], limit)).fetchall()
        return [r["text"] for r in rows]

    def stats(self):
        conn = self._conn()
        total = conn.execute("SELECT COUNT(*) c FROM reviews").fetchone()["c"]
        by_label = {r["pred_label"]: r["c"] for r in conn.execute(
            "SELECT pred_label, COUNT(*) c FROM reviews GROUP BY pred_label")}
        pending = conn.execute(
            "SELECT COUNT(*) c FROM reviews WHERE need_review=1 AND processed=0").fetchone()["c"]
        critical = conn.execute(
            "SELECT COUNT(*) c FROM reviews WHERE need_process=1 AND processed=0").fetchone()["c"]
        avg_score = conn.execute("SELECT AVG(score) s FROM reviews").fetchone()["s"]
        low_confidence = conn.execute(
            "SELECT COUNT(*) c FROM reviews WHERE need_review=1").fetchone()["c"]
        return {
            "total": total,
            "by_label": {k: v for k, v in by_label.items() if k},
            "pending_review": pending,
            "pending_process": critical,
            "low_confidence": low_confidence,
            "avg_score": round(avg_score, 4) if avg_score is not None else None,
        }
