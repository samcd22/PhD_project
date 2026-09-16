import os
import pytest


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run each test in an empty temporary working directory so 'data/' and 'results/' land there."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
