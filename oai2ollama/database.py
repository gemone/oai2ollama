import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_path: str = "api_metrics.db"):
        self.db_path = Path(db_path)
        self.init_database()

    def init_database(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # API调用记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    model TEXT,
                    request_tokens INTEGER DEFAULT 0,
                    response_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    status_code INTEGER,
                    response_time_ms REAL,
                    error_message TEXT,
                    request_data TEXT,
                    metadata TEXT
                )
            """)

            # Model usage statistics table (for fast queries)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_stats (
                    model TEXT PRIMARY KEY,
                    total_calls INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    request_tokens INTEGER DEFAULT 0,
                    response_tokens INTEGER DEFAULT 0,
                    last_updated TEXT NOT NULL
                )
            """)

            # 每日统计表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    total_calls INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    unique_models INTEGER DEFAULT 0,
                    last_updated TEXT NOT NULL
                )
            """)

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    def record_api_call(
        self, endpoint: str, method: str, model: str | None = None, request_tokens: int = 0, response_tokens: int = 0, status_code: int | None = None, response_time_ms: float | None = None, error_message: str | None = None, request_data: dict[str, object] | None = None, metadata: dict[str, object] | None = None
    ):
        """记录API调用"""
        timestamp = datetime.now(timezone.utc).isoformat()
        total_tokens = request_tokens + response_tokens

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 插入API调用记录
            cursor.execute(
                """
                INSERT INTO api_calls 
                (timestamp, endpoint, method, model, request_tokens, response_tokens, 
                 total_tokens, status_code, response_time_ms, error_message, request_data, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (timestamp, endpoint, method, model, request_tokens, response_tokens, total_tokens, status_code, response_time_ms, error_message, json.dumps(request_data) if request_data else None, json.dumps(metadata) if metadata else None),
            )

            # 更新模型统计
            if model:
                cursor.execute(
                    """
                    INSERT INTO model_stats (model, total_calls, total_tokens, request_tokens, response_tokens, last_updated)
                    VALUES (?, 1, ?, ?, ?, ?)
                    ON CONFLICT(model) DO UPDATE SET
                        total_calls = total_calls + 1,
                        total_tokens = total_tokens + ?,
                        request_tokens = request_tokens + ?,
                        response_tokens = response_tokens + ?,
                        last_updated = ?
                """,
                    (model, total_tokens, request_tokens, response_tokens, timestamp, total_tokens, request_tokens, response_tokens, timestamp),
                )

            # 更新每日统计
            date_str = timestamp.split("T")[0]
            cursor.execute(
                """
                INSERT INTO daily_stats (date, total_calls, total_tokens, unique_models, last_updated)
                VALUES (?, 1, ?, 1, ?)
                ON CONFLICT(date) DO UPDATE SET
                    total_calls = total_calls + 1,
                    total_tokens = total_tokens + ?,
                    last_updated = ?
            """,
                (date_str, total_tokens, timestamp, total_tokens, timestamp),
            )

            # If there's a new model, update unique model count
            if model:
                cursor.execute(
                    """
                    UPDATE daily_stats 
                    SET unique_models = (
                        SELECT COUNT(DISTINCT model) 
                        FROM api_calls 
                        WHERE date(timestamp) = ? AND model IS NOT NULL
                    )
                    WHERE date = ?
                """,
                    (date_str, date_str),
                )

            conn.commit()

    def get_model_stats(self, model: str | None = None) -> list[dict[str, object]]:
        """获取模型统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if model:
                cursor.execute(
                    """
                    SELECT * FROM model_stats WHERE model = ?
                """,
                    (model,),
                )
                rows = cursor.fetchall()
            else:
                cursor.execute("""
                    SELECT * FROM model_stats ORDER BY total_tokens DESC
                """)
                rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_daily_stats(self, days: int = 7) -> list[dict[str, object]]:
        """获取每日统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM daily_stats 
                WHERE date >= date('now', '-{} days')
                ORDER BY date DESC
            """.format(days)
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_overall_stats(self) -> dict[str, object]:
        """获取总体统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 总体统计
            cursor.execute("SELECT COUNT(*) as total_calls FROM api_calls")
            total_calls = cursor.fetchone()[0]

            cursor.execute("SELECT SUM(total_tokens) as total_tokens FROM api_calls")
            total_tokens = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(DISTINCT model) as unique_models FROM api_calls WHERE model IS NOT NULL")
            unique_models = cursor.fetchone()[0]

            # 今日统计
            cursor.execute("""
                SELECT COUNT(*) as today_calls, SUM(total_tokens) as today_tokens
                FROM api_calls 
                WHERE date(timestamp) = date('now')
            """)
            today_stats = cursor.fetchone()

            # 最活跃的模型
            cursor.execute("""
                SELECT model, SUM(total_tokens) as tokens
                FROM api_calls 
                WHERE model IS NOT NULL
                GROUP BY model 
                ORDER BY tokens DESC 
                LIMIT 1
            """)
            top_model = cursor.fetchone()

            return {"total_calls": total_calls, "total_tokens": total_tokens, "unique_models": unique_models, "today_calls": today_stats[0] or 0, "today_tokens": today_stats[1] or 0, "top_model": {"model": top_model[0] if top_model else None, "tokens": top_model[1] if top_model else 0}}

    def get_recent_calls(self, limit: int = 100, model: str | None = None) -> list[dict[str, object]]:
        """获取最近的API调用记录"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if model:
                cursor.execute(
                    """
                    SELECT * FROM api_calls 
                    WHERE model = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """,
                    (model, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM api_calls 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """,
                    (limit,),
                )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]


# 全局数据库实例
db_manager = DatabaseManager()
