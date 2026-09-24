"""DuckDB persistence layer for append-only execution history.

Architectural boundaries:
- Appends immutable RunResult records to disk.
- Exposes structured queries for metrics calculation.
- JSON payloads are dumped as raw strings for schema-less storage within columns.
"""

import dataclasses
import json

import duckdb

from src.models import RunResult


class RunStore:
    def __init__(self, db_path: str = "jev_runs.duckdb"):
        self.db_path = db_path

    def create_tables(self) -> None:
        """Initialize the DuckDB append-only tables."""
        with duckdb.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS run_batches (
                    batch_id VARCHAR PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    use_case VARCHAR,
                    metadata JSON
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS run_results (
                    run_id VARCHAR PRIMARY KEY,
                    batch_id VARCHAR,
                    provider VARCHAR,
                    model_version VARCHAR,
                    use_case VARCHAR,
                    question_key VARCHAR,

                    original_state VARCHAR,
                    perturbed_state VARCHAR,
                    perturbation_category VARCHAR,
                    transform_name VARCHAR,
                    expectation VARCHAR,

                    raw_api_response JSON,
                    parsed_response JSON,

                    latency_ms DOUBLE,
                    cost_usd DOUBLE,
                    http_status INTEGER,
                    retry_count INTEGER,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def save_batch(
        self, batch_id: str, results: list[RunResult], use_case: str, metadata: dict | None = None
    ) -> None:
        """Save a complete batch of execution results atomically."""
        if not results:
            return

        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO run_batches (batch_id, use_case, metadata) VALUES (?, ?, ?)",
                (batch_id, use_case, json.dumps(metadata) if metadata else None),
            )

            rows = []
            for r in results:
                raw_json = json.dumps(r.raw_api_response)

                if isinstance(r.parsed_response, dict):
                    parsed_json = json.dumps(r.parsed_response)
                else:
                    parsed_json = json.dumps(dataclasses.asdict(r.parsed_response))

                rows.append(
                    (
                        r.run_id,
                        r.batch_id,
                        r.provider,
                        r.model_version,
                        r.use_case,
                        r.question_key,
                        r.original_state,
                        r.perturbed_state,
                        r.perturbation_category,
                        r.transform_name,
                        r.expectation,
                        raw_json,
                        parsed_json,
                        r.latency_ms,
                        r.cost_usd,
                        r.http_status,
                        r.retry_count,
                    )
                )

            conn.executemany(
                """
                INSERT INTO run_results (
                    run_id, batch_id, provider, model_version, use_case, question_key,
                    original_state, perturbed_state, perturbation_category, transform_name,
                    expectation, raw_api_response, parsed_response,
                    latency_ms, cost_usd, http_status, retry_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
