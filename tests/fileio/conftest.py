import os
import shutil

import pytest


@pytest.fixture()
def datadir(tmpdir, request):
    test_dir = os.path.join(os.path.dirname(request.module.__file__), "test_data")
    if os.path.isdir(test_dir):
        shutil.copytree(test_dir, str(tmpdir), dirs_exist_ok=True)
    return tmpdir
