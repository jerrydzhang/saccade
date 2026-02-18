"""Pytest configuration for Talos tests."""

pytest_plugins = ("pytest_asyncio",)


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "unit: unit tests (isolated component tests)")
    config.addinivalue_line("markers", "integration: integration tests (multi-component tests)")
