from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

class BaseModel(ABC):
    """Base interface for all models in the system"""
    
    @abstractmethod
    def predict(self, texts: List[str]) -> Dict[str, Any]:
        """
        Make predictions on input texts
        
        Args:
            texts: List of input texts to predict
            
        Returns:
            Dict containing:
            - predictions: List of predicted classes
            - confidence: List of confidence scores (optional)
            - probabilities: List of probability distributions (optional)
        """
        pass
    
    @abstractmethod
    def load(self, model_path: str) -> None:
        """
        Load model from artifact path
        
        Args:
            model_path: Path to model artifacts
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata
        
        Returns:
            Dict containing model information
        """
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type identifier"""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        pass