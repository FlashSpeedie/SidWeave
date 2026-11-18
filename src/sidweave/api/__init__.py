"""
OpenAI-compatible API gateway for SidWeave.
"""

from sidweave.api.gateway import app, create_app

__all__ = [
    "app",
    "create_app",
]
