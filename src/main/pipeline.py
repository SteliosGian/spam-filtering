from sklearn.pipeline import Pipeline
from preprocess.transformers import TokenizerToSequence, PaddingSequences
from config import config
from models.lstm_model import LSTMModel


pipe = Pipeline([('tokenizer', TokenizerToSequence(num_words=config.NUM_WORDS)),
                 ('padding', PaddingSequences(maxlen=config.MAXLEN)),
                 ('Estimator', LSTMModel())])
