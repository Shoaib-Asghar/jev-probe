"""DuckDB persistence layer for append-only execution history.

Architectural boundaries:
- Appends immutable RunResult records to disk.
- Exposes structured queries for metrics calculation.
- JSON payloads are dumped as raw strings for schema-less storage within columns.
"""

import dataclasses
import json

import duckdb

from src.models import ChoiceResponse, NoulResponse, RunResult, ScoreResponse


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

    def _row_to_result(self, row: tuple) -> RunResult:
        """Convert a DuckDB row tuple back to a RunResult dataclass."""
        (
            run_id,
            batch_id,
            provider,
            model_version,
            use_case,
            question_key,
            original_state,
            perturbed_state,
            perturbation_category,
            transform_name,
            expectation,
            raw_api_response,
            parsed_response,
            latency_ms,
            cost_usd,
            http_status,
            retry_count,
            _,
        ) = row

        parsed = json.loads(parsed_response)
        typed_response = parsed

        # Attempt to infer the typed response based on dictionary shape
        if "judgment" in parsed and "probability" in parsed:
            typed_response = NoulResponse(
                probability=parsed["probability"], judgment=parsed["judgment"]
            )
        elif "selected" in parsed and "probabilities" in parsed:
            typed_response = ChoiceResponse(
                selected=parsed["selected"],
                probabilities=parsed["probabilities"],
                confidence=parsed.get("confidence", 0.0),
            )
        elif "score" in parsed and "distribution" in parsed:
            typed_response = ScoreResponse(
                score=parsed["score"],
                distribution=parsed["distribution"],
                confidence=parsed.get("confidence", 0.0),
            )

        return RunResult(
            run_id=run_id,
            batch_id=batch_id,
            provider=provider,
            model_version=model_version,
            use_case=use_case,
            question_key=question_key,
            original_state=original_state,
            perturbed_state=perturbed_state,
            perturbation_category=perturbation_category,
            transform_name=transform_name,
            expectation=expectation,  # type: ignore
            raw_api_response=json.loads(raw_api_response),
            parsed_response=typed_response,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            http_status=http_status,
            retry_count=retry_count,
        )

    def get_batch(self, batch_id: str) -> list[RunResult]:
        """Retrieve all results for a specific batch ID."""
        with duckdb.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM run_results WHERE batch_id = ?",
                (batch_id,)
            ).fetchall()
            return [self._row_to_result(row) for row in rows]

    def get_results_by_category(self, batch_id: str, category: str) -> list[RunResult]:
        """Retrieve results for a specific batch filtered by perturbation category."""
        with duckdb.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM run_results WHERE batch_id = ? AND perturbation_category = ?",
                (batch_id, category),
            ).fetchall()
            return [self._row_to_result(row) for row in rows]

    def get_all_batches(self) -> list[dict]:
        """Retrieve all execution batches and their metadata."""
        with duckdb.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT batch_id, created_at, use_case, metadata "
                "FROM run_batches ORDER BY created_at DESC"
            ).fetchall()

            batches = []
            for row in rows:
                batches.append(
                    {
                        "batch_id": row[0],
                        "created_at": row[1],
                        "use_case": row[2],
                        "metadata": json.loads(row[3]) if row[3] else None,
                    }
                )
            return batches
