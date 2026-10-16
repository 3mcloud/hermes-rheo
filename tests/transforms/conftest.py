from __future__ import annotations

import shutil
from pathlib import Path

import pytest


@pytest.fixture()
def datadir(tmp_path: Path) -> Path:
    """Copy file-reader fixtures for transforms that need input files."""
    source = Path(__file__).parents[1] / "fileio" / "test_data"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    return tmp_path
