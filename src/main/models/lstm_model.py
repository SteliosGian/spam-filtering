import numpy as np
from typing import List
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM, Embedding
from tensorflow.keras.models import Model
from models.base_model import BaseModel


class LSTMModel(BaseModel):
    """
    Long Short-term Memory Neural Network.
    """
    def __init__(self, lstm_units: int = 15, 
                 embedding_dim: int = 20, 
                 input_dim: int = 7295, 
                 loss: str = 'binary_crossentropy', 
                 optimizer: str = 'adam', 
                 metrics: List[str] = ['accuracy']):
        """
        Initialize the LSTM model.
        :param lstm_units: Number of units in a layer
        :param embedding_dim: Embedding dimension
        :param input_dim: Input dimension
        :param input_shape: Input shape
        :param loss: Loss function
        :param optimizer: Model optimizer
        :param metrics: Metrics for evaluation
        :return: Initialized model
        """
        self.lstm_units = lstm_units
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.loss = loss
        self. optimizer = optimizer
        self.metrics = metrics
    
    def _model_arch(self) -> Model:
        """
        Create the model architecture.
        :return: Model
        """
        i = Input(shape=(self.input_shape,))
        x = Embedding(self.input_dim + 1, self.embedding_dim)(i)
        x = LSTM(self.lstm_units, return_sequences=True)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(1, activation='sigmoid')(x)
        self.model = Model(i, x)
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        return self
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs = 2, validation_data = None):
        """
        Train the model.
        :param X: Predictors array
        :param y: Target feature
        :param epochs: Number of epochs
        :param validation_data: A tuple of validation data (x_test, y_test)
        :return: Class instance
        """
        self.epochs = epochs
        self.input_shape = X.shape[1]
        self._model_arch()
        self.model.fit(X, y, epochs=self.epochs, validation_data=validation_data)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        :param X: Input data array
        :return: Predictions
        """
        predictions = self.model.predict(X)
        return predictions
    
    def save(self, path: str) -> None:
        """
        Save the trained model.
        :param path: Path to save the model
        :return: None
        """
        self.model.save(path)
    
    @property
    def get_numeric_params(self) -> dict:
        """Get the parameters"""
        
        return {'lstm_units': self.lstm_units,
                'embedding_dim': self.embedding_dim,
                'input_dim': self.input_dim,
                'loss': self.loss,
                'optimizer': self.optimizer,
                'metrics': self.metrics,
                'epochs': self.epochs
                }
