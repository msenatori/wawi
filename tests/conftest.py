"""Shared test setup.

Every test gets its own private folder (wawi/paths.py), so nothing a test
writes by default can land in -- or read from -- the real ``private/``.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_private_dir(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("WAWI_PRIVATE_DIR", str(tmp_path_factory.mktemp("private")))
