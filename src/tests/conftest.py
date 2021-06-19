import sys
# sys.path.insert(0, 'main')
import pytest
from preprocess.data_manager import load_dataset



@pytest.fixture()
def get_data():
    return load_dataset(path='tests/test_dataset/test_data.csv')
