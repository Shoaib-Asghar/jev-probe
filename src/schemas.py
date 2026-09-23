"""Schemas layer: Boundary validation and serialization.

Architectural boundary:
- Models for incoming requests (e.g., initiating a benchmark run) and outgoing API/SSE responses.
- Enforces data integrity at external boundaries before objects enter the domain engine.
"""
