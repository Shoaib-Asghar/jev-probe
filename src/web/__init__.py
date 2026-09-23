"""Web presentation layer: Interactive evaluation dashboard and streaming endpoints.

Architectural boundary:
- FastAPI routes serving HTML fragments (HTMX), client state (Alpine.js), and charts (Chart.js).
- Server-Sent Events (SSE) endpoints streaming real-time progress of evaluation runs.
- Consumes engine and store APIs; does not execute low-level model calls directly.
"""
