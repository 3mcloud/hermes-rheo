import os
import shutil
import pytest

@pytest.fixture()
def datadir(tmpdir, request):
    # Locate the directory containing test files
    filename = request.module.__file__
    test_dir = os.path.join(os.path.dirname(filename), 'test_data')  # Ensure it points to 'test_data' directory

    if os.path.isdir(test_dir):
        shutil.copytree(test_dir, str(tmpdir), dirs_exist_ok=True)  # Copy the entire folder to the tmpdir

    return tmpdir
