import os
import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("API_KEYS", "dev-key-123")
