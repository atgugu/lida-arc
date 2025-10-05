import pytest

pytest_plugins = ("pytest_asyncio",)

@pytest.fixture(autouse=True, scope="session")
def _pytest_asyncio_auto():
    # Ensure pytest-asyncio default mode for async tests
    pass
