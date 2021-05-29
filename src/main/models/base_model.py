from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base model template
    """
    @abstractmethod
    def __init__(self):
        raise NotImplementedError()
    
    @abstractmethod
    def _model_arch(self):
        raise NotImplementedError()
    
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()
    
    @abstractmethod
    def save(self, path):
        raise NotImplementedError()
