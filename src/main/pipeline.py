from sklearn.pipeline import Pipeline
from preprocess.transformers import TokenizerToSequence, PaddingSequences
from config import config


pipe = Pipeline([('tokenizer', TokenizerToSequence(num_words=config.NUM_WORDS)),
                 ('padding', PaddingSequences(maxlen=config.MAXLEN))])

