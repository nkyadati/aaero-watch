"""
Aero-Watch Detection Store — SQLite Persistence
=================================================

Replaces in-memory lists with a SQLite database so that detection
history, aggregate stats, and drift baselines survive API restarts.

Tables:
  detections  — one row per processed image
  drift_state — single-row store for baseline + last report

Usage:
  store = DetectionStore("aerowatch.db")
  store.insert_detection({...})
  history = store.get_history(limit=50)
  stats = store.get_aggregate_stats()
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH_DEFAULT = "aerowatch.db"


class DetectionStore:
    """SQLite-backed detection store.

    Thread-safe: uses check_same_thread=False and WAL journal mode
    for concurrent reads from async endpoints.
    """

    def __init__(self, db_path: str = DB_PATH_DEFAULT):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self):
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()
        logger.info("Detection store: %s", self.db_path)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Schema ────────────────────────────────────────────────────────

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id    TEXT NOT NULL,
                camera_id   TEXT NOT NULL,
                sector      TEXT NOT NULL DEFAULT 'default',
                timestamp   TEXT NOT NULL,
                total_birds     INTEGER NOT NULL DEFAULT 0,
                black_bird_count INTEGER NOT NULL DEFAULT 0,
                color_counts    TEXT NOT NULL DEFAULT '{}',
                detections_json TEXT NOT NULL DEFAULT '[]',
                processing_time_ms REAL NOT NULL DEFAULT 0.0,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_detections_timestamp
                ON detections(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_detections_camera
                ON detections(camera_id);

            CREATE TABLE IF NOT EXISTS drift_state (
                id              INTEGER PRIMARY KEY CHECK (id = 1),
                baseline_json   TEXT,
                last_report_json TEXT,
                updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Ensure single row
            INSERT OR IGNORE INTO drift_state (id) VALUES (1);
        """)
        self._conn.commit()

    # ── Detection CRUD ────────────────────────────────────────────────

    def insert_detection(self, record: dict):
        """Insert a detection result."""
        self._conn.execute(
            """INSERT INTO detections
               (image_id, camera_id, sector, timestamp,
                total_birds, black_bird_count, color_counts,
                detections_json, processing_time_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.get("image_id", ""),
                record.get("camera_id", ""),
                record.get("sector", "default"),
                record.get("timestamp", ""),
                record.get("total_birds", 0),
                record.get("black_bird_count", 0),
                json.dumps(record.get("color_counts", {})),
                json.dumps(record.get("detections", [])),
                record.get("processing_time_ms", 0.0),
            ),
        )
        self._conn.commit()

    def get_history(self, limit: int = 50) -> list[dict]:
        """Return recent detections, newest first."""
        rows = self._conn.execute(
            """SELECT image_id, camera_id, sector, timestamp,
                      total_birds, black_bird_count, color_counts,
                      detections_json, processing_time_ms
               FROM detections ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()

        return [
            {
                "image_id": r["image_id"],
                "camera_id": r["camera_id"],
                "sector": r["sector"],
                "timestamp": r["timestamp"],
                "total_birds": r["total_birds"],
                "black_bird_count": r["black_bird_count"],
                "color_counts": json.loads(r["color_counts"]),
                "detections": json.loads(r["detections_json"]),
                "processing_time_ms": r["processing_time_ms"],
            }
            for r in rows
        ]

    def get_aggregate_stats(self) -> dict:
        """Compute aggregate stats from all stored detections."""
        row = self._conn.execute(
            """SELECT
                 COUNT(*)           AS images_processed,
                 COALESCE(SUM(total_birds), 0)      AS total_birds,
                 COALESCE(SUM(black_bird_count), 0)  AS total_black_birds,
                 COALESCE(AVG(processing_time_ms), 0) AS avg_latency_ms
               FROM detections"""
        ).fetchone()

        # Aggregate colour counts
        color_rows = self._conn.execute(
            "SELECT color_counts FROM detections"
        ).fetchall()
        color_totals: dict[str, int] = {}
        for cr in color_rows:
            for color, count in json.loads(cr["color_counts"]).items():
                color_totals[color] = color_totals.get(color, 0) + count

        return {
            "images_processed": row["images_processed"],
            "total_birds": row["total_birds"],
            "total_black_birds": row["total_black_birds"],
            "avg_latency_ms": round(row["avg_latency_ms"], 1),
            "color_totals": color_totals,
        }

    def get_detection_count(self) -> int:
        """Total number of processed images."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM detections").fetchone()
        return row["n"]

    # ── Drift State ───────────────────────────────────────────────────

    def save_drift_baseline(self, baseline: dict):
        """Persist the drift reference baseline."""
        self._conn.execute(
            """UPDATE drift_state
               SET baseline_json = ?, updated_at = datetime('now')
               WHERE id = 1""",
            (json.dumps(baseline),),
        )
        self._conn.commit()

    def load_drift_baseline(self) -> Optional[dict]:
        """Load persisted drift baseline, or None."""
        row = self._conn.execute(
            "SELECT baseline_json FROM drift_state WHERE id = 1"
        ).fetchone()
        if row and row["baseline_json"]:
            return json.loads(row["baseline_json"])
        return None

    def save_drift_report(self, report: dict):
        """Persist the latest drift report."""
        self._conn.execute(
            """UPDATE drift_state
               SET last_report_json = ?, updated_at = datetime('now')
               WHERE id = 1""",
            (json.dumps(report),),
        )
        self._conn.commit()

    def load_drift_report(self) -> Optional[dict]:
        """Load the last drift report, or None."""
        row = self._conn.execute(
            "SELECT last_report_json FROM drift_state WHERE id = 1"
        ).fetchone()
        if row and row["last_report_json"]:
            return json.loads(row["last_report_json"])
        return None

    def reset(self):
        """Clear all data (useful for fresh simulation runs)."""
        self._conn.executescript("""
            DELETE FROM detections;
            UPDATE drift_state SET baseline_json = NULL,
                   last_report_json = NULL, updated_at = datetime('now')
            WHERE id = 1;
        """)
        self._conn.commit()
        logger.info("Detection store reset")
