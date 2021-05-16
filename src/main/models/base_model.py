from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError()
    
    @abstractmethod
    def model_arch(self):
        raise NotImplementedError()
    
    @abstractmethod
    def fit(self, x, y):
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, x):
        raise NotImplementedError()
    
    @abstractmethod
    def save(self, path):
        raise NotImplementedError()
