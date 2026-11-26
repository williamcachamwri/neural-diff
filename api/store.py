"""SQLite persistence layer for experiments, regression baselines, and settings."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "neural_diff.db"

DEFAULT_SETTINGS: dict[str, Any] = {
    "preferred_device": "auto",
    "atol": 1e-6,
    "rtol": 1e-5,
    "zero_threshold": 0.0,
    "chunk_elements": 1_048_576,
    "allow_pytorch_weights_only": True,
    "max_new_tokens": 96,
}


class ExperimentStore:
    """Thread-safe SQLite store backing experiment history, live progress, baselines, and settings."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    comparison_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage_message TEXT NOT NULL,
                    completed_steps INTEGER NOT NULL,
                    total_steps INTEGER NOT NULL,
                    experiment_type TEXT NOT NULL,
                    model_a_id TEXT NOT NULL,
                    model_b_id TEXT NOT NULL,
                    selected_modules_json TEXT NOT NULL,
                    verdict TEXT,
                    summary_json TEXT,
                    bundle_json TEXT,
                    markdown_report TEXT,
                    html_report TEXT,
                    error_json TEXT
                );

                CREATE TABLE IF NOT EXISTS baselines (
                    baseline_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    suite_id TEXT NOT NULL,
                    suite_version TEXT NOT NULL,
                    overall_pass_rate REAL NOT NULL,
                    overall_mean_score REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                """
            )

    def create_job(
        self,
        *,
        comparison_id: str,
        created_at_utc: str,
        experiment_type: str,
        model_a_id: str,
        model_b_id: str,
        selected_modules: list[str],
        total_steps: int,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    comparison_id, created_at_utc, status, stage_message,
                    completed_steps, total_steps, experiment_type,
                    model_a_id, model_b_id, selected_modules_json,
                    verdict, summary_json, bundle_json, markdown_report, html_report, error_json
                ) VALUES (?, ?, 'running', 'Initializing comparison', 0, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL)
                """,
                (
                    comparison_id,
                    created_at_utc,
                    total_steps,
                    experiment_type,
                    model_a_id,
                    model_b_id,
                    json.dumps(selected_modules),
                ),
            )

    def update_progress(
        self,
        comparison_id: str,
        stage_message: str,
        completed_steps: int,
        total_steps: int,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE experiments
                SET stage_message = ?, completed_steps = ?, total_steps = ?
                WHERE comparison_id = ?
                """,
                (stage_message, completed_steps, total_steps, comparison_id),
            )

    def save_completed(
        self,
        *,
        comparison_id: str,
        model_a_id: str,
        model_b_id: str,
        verdict: str | None,
        summary_dict: dict[str, Any],
        bundle_json: str,
        markdown_report: str,
        html_report: str,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE experiments
                SET status = 'completed',
                    stage_message = 'Completed',
                    completed_steps = total_steps,
                    model_a_id = ?,
                    model_b_id = ?,
                    verdict = ?,
                    summary_json = ?,
                    bundle_json = ?,
                    markdown_report = ?,
                    html_report = ?
                WHERE comparison_id = ?
                """,
                (
                    model_a_id,
                    model_b_id,
                    verdict,
                    json.dumps(summary_dict),
                    bundle_json,
                    markdown_report,
                    html_report,
                    comparison_id,
                ),
            )

    def save_failed(
        self,
        comparison_id: str,
        error_dict: dict[str, Any],
        *,
        cancelled: bool = False,
    ) -> None:
        status = "cancelled" if cancelled else "failed"
        msg = str(error_dict.get("message", status))
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE experiments
                SET status = ?, stage_message = ?, error_json = ?
                WHERE comparison_id = ?
                """,
                (status, msg[:240], json.dumps(error_dict), comparison_id),
            )

    def list_experiments(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT comparison_id, created_at_utc, status, stage_message,
                       completed_steps, total_steps, experiment_type,
                       model_a_id, model_b_id, selected_modules_json,
                       verdict, summary_json, error_json
                FROM experiments
                ORDER BY created_at_utc DESC
                LIMIT ?
                """,
                (max(1, min(limit, 500)),),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for r in rows:
            results.append(
                {
                    "comparison_id": r["comparison_id"],
                    "created_at_utc": r["created_at_utc"],
                    "status": r["status"],
                    "stage_message": r["stage_message"],
                    "completed_steps": r["completed_steps"],
                    "total_steps": r["total_steps"],
                    "experiment_type": r["experiment_type"],
                    "model_a_id": r["model_a_id"],
                    "model_b_id": r["model_b_id"],
                    "selected_modules": json.loads(r["selected_modules_json"]),
                    "verdict": r["verdict"],
                    "summary": json.loads(r["summary_json"]) if r["summary_json"] else None,
                    "error": json.loads(r["error_json"]) if r["error_json"] else None,
                }
            )
        return results

    def get_experiment(self, comparison_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE comparison_id = ?",
                (comparison_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "comparison_id": row["comparison_id"],
            "created_at_utc": row["created_at_utc"],
            "status": row["status"],
            "stage_message": row["stage_message"],
            "completed_steps": row["completed_steps"],
            "total_steps": row["total_steps"],
            "experiment_type": row["experiment_type"],
            "model_a_id": row["model_a_id"],
            "model_b_id": row["model_b_id"],
            "selected_modules": json.loads(row["selected_modules_json"]),
            "verdict": row["verdict"],
            "summary": json.loads(row["summary_json"]) if row["summary_json"] else None,
            "bundle": json.loads(row["bundle_json"]) if row["bundle_json"] else None,
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
        }

    def get_export(self, comparison_id: str, fmt: str) -> tuple[str, str] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT bundle_json, markdown_report, html_report FROM experiments WHERE comparison_id = ?",
                (comparison_id,),
            ).fetchone()
        if row is None or row["bundle_json"] is None:
            return None
        if fmt == "json":
            return row["bundle_json"], "application/json"
        if fmt == "markdown":
            return row["markdown_report"], "text/markdown; charset=utf-8"
        if fmt == "html":
            return row["html_report"], "text/html; charset=utf-8"
        return None

    def save_baseline(self, payload: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO baselines (
                    baseline_id, created_at_utc, model_id, suite_id, suite_version,
                    overall_pass_rate, overall_mean_score, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["baseline_id"],
                    payload["created_at_utc"],
                    payload["model_id"],
                    payload["suite_id"],
                    payload["suite_version"],
                    float(payload["overall_pass_rate"]),
                    float(payload["overall_mean_score"]),
                    json.dumps(payload),
                ),
            )

    def list_baselines(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM baselines ORDER BY created_at_utc DESC"
            ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def get_baseline(self, baseline_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM baselines WHERE baseline_id = ?",
                (baseline_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def delete_baseline(self, baseline_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM baselines WHERE baseline_id = ?", (baseline_id,))
            return cur.rowcount > 0

    def get_settings(self) -> dict[str, Any]:
        merged = dict(DEFAULT_SETTINGS)
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT key, value_json FROM settings").fetchall()
        for r in rows:
            merged[r["key"]] = json.loads(r["value_json"])
        return merged

    def update_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            for k, v in updates.items():
                conn.execute(
                    "INSERT OR REPLACE INTO settings (key, value_json) VALUES (?, ?)",
                    (k, json.dumps(v)),
                )
        return self.get_settings()
