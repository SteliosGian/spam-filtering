import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TokenizerToSequence(BaseEstimator, TransformerMixin):
    """
    Convert text to sequences.
    """
    def __init__(self, num_words: int):
        """
        Initialize the TokenizerToSequence
        :param num_words: The maximum number of words in a sentence.
        """
        self.num_words = num_words
        self.tokenizer = Tokenizer(num_words=self.num_words)
    
    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None):
        self.tokenizer.fit_on_texts(x)
        return self
        
    def transform(self, x: pd.DataFrame) -> list:
        sequences = self.tokenizer.texts_to_sequences(x)
        return sequences


class PaddingSequences(BaseEstimator, TransformerMixin):
    """
    Pad sequences.
    """
    def __init__(self, maxlen: int):
        """
        Initialize the PaddingSequences
        :param maxlen: Maximum length of sequences
        """
        self.maxlen = maxlen
    
    def fit(self, x, y=None):
        return self
        
    def transform(self, x: list) -> np.ndarray:
        pad_seq = pad_sequences(x, maxlen=self.maxlen)
        return pad_seq
