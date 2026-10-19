"""Shared test setup.

Every test gets its own private folder (wawi/paths.py), so nothing a test
writes by default can land in -- or read from -- the real ``private/``.
Module- and session-scoped fixtures run before the per-test one, so the
whole run starts pointed at a throwaway folder too.
"""

import os
import tempfile

import pytest

os.environ["WAWI_PRIVATE_DIR"] = tempfile.mkdtemp(prefix="wawi-test-private-")


@pytest.fixture(autouse=True)
def _isolated_private_dir(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("WAWI_PRIVATE_DIR", str(tmp_path_factory.mktemp("private")))
