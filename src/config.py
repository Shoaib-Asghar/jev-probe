"""Configuration and pricing registry.

Architectural boundary:
- Centralizes runtime settings (database paths, default timeouts, batch sizes).
- Houses official pricing tables for token and request cost calculation.
- Never reads or hardcodes API keys or sensitive secrets.
"""
