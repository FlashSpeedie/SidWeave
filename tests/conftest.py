"""
Shared test configuration and fixtures.
"""

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from sidweave.config import Settings, reset_settings
from sidweave.storage import close_database, get_session, init_database


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_settings_fixture():
    """Reset settings before each test."""
    reset_settings()
    yield
    reset_settings()


@pytest_asyncio.fixture
async def db_engine():
    """Initialize database engine for tests using in-memory SQLite."""
    s = Settings()
    s.storage.data_dir = Path()
    s.storage.db_filename = ":memory:"
    s.debug = True
    s.provider.enable_cloud_providers = False
    s.telemetry.enabled = False

    engine = await init_database(s)
    yield engine
    await close_database()


@pytest_asyncio.fixture
async def db_session(db_engine):
    """Get database session for tests."""
    async with get_session(db_engine) as session:
        yield session


# Async test utilities
def async_test(func):
    """Decorator to run async test functions."""
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper
