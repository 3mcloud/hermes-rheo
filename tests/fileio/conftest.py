import os
from distutils import dir_util
import pytest


@pytest.fixture()
def datadir(tmpdir, request):
    # locate the directory containing test files
    filename = request.module.__file__
    test_dir = os.path.join(os.path.dirname(filename), 'test_data')  # Ensure it points to 'test_data' directory

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))  # Copy the entire folder to the tmpdir

    return tmpdir