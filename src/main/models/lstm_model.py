import numpy as np
from typing import List, Tuple, Optional
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM, Embedding
from tensorflow.keras.models import Model
from models.base_model import BaseModel
from sklearn.base import BaseEstimator, ClassifierMixin


class LSTMModel(BaseModel, BaseEstimator, ClassifierMixin):
    def __init__(self, lstm_units: int = 15, 
                 embedding_dim: int = 20, input_dim: int = 7295, 
                 input_shape: int = 189, validation_data: Optional[Tuple[np.ndarray]] = None, 
                 loss: str = 'binary_crossentropy', optimizer: str = 'adam', 
                 metrics: List[str] = ['accuracy'], epochs: int = 10):
        self.lstm_units = lstm_units
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.input_shape = input_shape
        self.validation_data = validation_data
        self.loss = loss
        self. optimizer = optimizer
        self.metrics = metrics
        self.epochs = epochs
    
    def _model_arch(self) -> Model:
        i = Input(shape=(self.input_shape,))
        x = Embedding(self.input_dim + 1, self.embedding_dim)(i)
        x = LSTM(self.lstm_units, return_sequences=True)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(1, activation='sigmoid')(x)
        self.model = Model(i, x)
        
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self._model_arch()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.model.fit(X, y, epochs=self.epochs, validation_data=self.validation_data)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = self.model.predict(X)
        return predictions
    
    def save(self, path: str) -> None:
        self.model.save(path)
