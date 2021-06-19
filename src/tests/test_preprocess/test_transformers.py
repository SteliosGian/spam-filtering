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


@pytest.fixture()
def get_sequence():
    seq = [[422, 1, 1770, 10, 608, 1261, 20, 1, 520, 3404, 27, 3405, 3406, 193],
            [371, 62, 226, 805, 32, 26, 112, 220],
            [572, 25, 1, 181, 45, 21, 35, 340, 1105, 3407, 146]]
    return seq


def test_TokenizerToSequence_fit(get_data, get_TokenizerToSequence):
    # When
    fitted_tokenizer = get_TokenizerToSequence.fit(x = get_data[config.FEATURES])
    # Then
    assert len(fitted_tokenizer.word_counts) > 0
    assert fitted_tokenizer.document_count == get_data.shape[0]


def test_TokenizerToSequence_transform(get_data, get_TokenizerToSequence):
    # When
    get_TokenizerToSequence.fit(x = get_data[config.FEATURES])
    sequence = get_TokenizerToSequence.transform(x = get_data[config.FEATURES])
    # Then
    assert len(sequence) == get_data.shape[0]


def test_PaddingSequences_transform(get_sequence, get_PaddingSequences):
    # When
    padded_seq = get_PaddingSequences.transform(get_sequence)
    # Then
    assert padded_seq.shape[0] == len(get_sequence)
    assert padded_seq.shape[1] == config.MAXLEN
