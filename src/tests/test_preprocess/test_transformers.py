import sys
sys.path.insert(0, 'main')
import pytest
from config import config
from conftest import get_data
from preprocess.transformers import TokenizerToSequence, PaddingSequences



@pytest.fixture()
def get_TokenizerToSequence():
    return TokenizerToSequence(config.NUM_WORDS)


@pytest.fixture()
def get_PaddingSequences():
    return PaddingSequences(config.MAXLEN)


def test_TokenizerToSequence_fit(get_data, get_TokenizerToSequence):
    # When
    df = get_TokenizerToSequence.fit(x = get_data[config.FEATURES])
    # Then
    pass


def test_TokenizerToSequence_transform(get_data, get_TokenizerToSequence):
    # When
    df = get_TokenizerToSequence.fit(x = get_data[config.FEATURES])
    df = get_TokenizerToSequence.transform(x = get_data[config.FEATURES])
    # Then
    pass


def test_PaddingSequences_fit(get_data, get_TokenizerToSequence):
    # When
    # Then
    pass


def test_PaddingSequences_transform(get_data, get_TokenizerToSequence):
    # When
    # Then
    pass