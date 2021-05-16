import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM, Embedding
from tensorflow.keras.models import Model
from models.base_model import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, lstm_units: int, embedding_dim: int, input_dim: int, input_shape: int):
        self.lstm_units = lstm_units
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.input_shape = input_shape
    
    def __model_arch(self):
        i = Input(shape=(self.input_shape,))
        x = Embedding(self.input_dim + 1, self.embedding_dim)(i)
        x = LSTM(self.lstm_units, return_sequences=True)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(1, activation='sigmoid')(x)
        self.model = Model(i, x)
        
    
    def fit(self, x: np.ndarray, y: np.ndarray, validation_data: Tuple(np.ndarray), loss: str, optimizer: str, metrics: List[str], epochs: int):
        self.__model_arch()
        self.model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
        self.model.fit(x, y, epochs=epochs, validation_data=validation_data)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = self.model.predict(x)
        return predictions
    
    def save(self, path: str):
        self.model.save(path)
