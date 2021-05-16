from sklearn.pipeline import Pipeline
from preprocess.transformers import TokenizerToSequence, PaddingSequences


pipe = Pipeline([('tokenizer', TokenizerToSequence(num_words=2000)),
                 ('padding', PaddingSequences(maxlen=189))])
