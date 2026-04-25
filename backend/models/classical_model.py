import pickle
import os
import logging
import sys
from typing import Dict, Any, List
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.core.models.base import BaseModel
from backend.core.config.settings import config
from backend.scripts.data_processing import get_class_names

logger = logging.getLogger(__name__)

class ClassicalModel(BaseModel):
    """Classical ML model wrapper implementing BaseModel interface"""
    
    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.model = None
        self.vectorizer = None
        self.svd = None
        self._loaded = False
    
    def predict(self, texts: List[str]) -> Dict[str, Any]:
        """
        Make predictions using classical ML model
        
        Args:
            texts: List of input texts
            
        Returns:
            Dict with predictions and metadata
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Vectorize texts
        X = self.vectorizer.transform(texts)
        
        # Apply SVD if available
        if self.svd is not None:
            X = self.svd.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X)
        confidences = []
        
        # Get confidence scores if available
        if hasattr(self.model, 'predict_proba'):
            try:
                probabilities = self.model.predict_proba(X)
                confidences = [float(max(proba)) for proba in probabilities]
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")
                confidences = [None] * len(texts)
        else:
            confidences = [None] * len(texts)
        
        # Convert to class names
        class_names = get_class_names()
        predicted_classes = [class_names[p] for p in predictions]
        
        return {
            "predictions": predicted_classes,
            "prediction_ids": predictions.tolist(),
            "confidence": confidences,
            "model_used": self.algorithm_name,
            "model_type": "classical"
        }
    
    def load(self, model_path: str) -> None:
        """
        Load classical ML model from pickle file
        
        Args:
            model_path: Path to model artifacts
        """
        try:
            # Load model pickle file
            pickle_path = os.path.join(model_path, f"{self.algorithm_name}.pkl")
            if not os.path.exists(pickle_path):
                raise FileNotFoundError(f"Model file not found: {pickle_path}")
            
            with open(pickle_path, 'rb') as f:
                obj = pickle.load(f)
            
            # Handle different pickle formats
            if isinstance(obj, dict):
                self.vectorizer = obj.get('vec')
                self.svd = obj.get('svd')
                self.model = obj.get('model')
            elif isinstance(obj, tuple) and len(obj) == 2:
                self.vectorizer, self.model = obj
                self.svd = None
            else:
                raise ValueError(f"Unknown pickle format for {self.algorithm_name}")
            
            if self.model is None or self.vectorizer is None:
                raise ValueError(f"Invalid model data in {pickle_path}")
            
            self._loaded = True
            logger.info(f"Classical model '{self.algorithm_name}' loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load classical model '{self.algorithm_name}': {e}")
            raise
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata"""
        return {
            "algorithm_name": self.algorithm_name,
            "model_type": "classical",
            "has_svd": self.svd is not None,
            "has_probabilities": hasattr(self.model, 'predict_proba') if self.model else False,
            "is_loaded": self._loaded
        }
    
    @property
    def model_type(self) -> str:
        return "classical"
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded