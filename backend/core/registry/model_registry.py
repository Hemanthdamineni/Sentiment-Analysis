from typing import Dict, Optional, List, Any
import os
import logging
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.core.models.base import BaseModel
from backend.core.config.settings import config

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Centralized model registry with caching and lifecycle management"""
    
    def __init__(self):
        self._models: Dict[str, BaseModel] = {}
        self._metadata: Dict[str, Dict] = {}
        self._load_order: List[str] = []
    
    def register(self, name: str, model: BaseModel, metadata: Optional[Dict] = None) -> None:
        """
        Register a model in the registry
        
        Args:
            name: Model name/identifier
            model: Model instance implementing BaseModel
            metadata: Optional metadata dictionary
        """
        if name in self._models:
            logger.warning(f"Model {name} already registered, overwriting")
        
        self._models[name] = model
        self._metadata[name] = metadata or {}
        self._load_order.append(name)
        logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> BaseModel:
        """
        Get a model by name
        
        Args:
            name: Model name
            
        Returns:
            BaseModel instance
            
        Raises:
            KeyError: If model not found
        """
        if name not in self._models:
            available = list(self._models.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        
        model = self._models[name]
        if not model.is_loaded:
            # Prefer explicit path stored in metadata, fall back to convention
            metadata = self._metadata.get(name, {})
            model_path = metadata.get("model_path") or os.path.join(config.models_dir, name)
            model.load(model_path)
            logger.info(f"Loaded model: {name}")
        
        return model
    
    def list_models(self) -> List[str]:
        """List all registered model names"""
        return list(self._models.keys())
    
    def get_metadata(self, name: str) -> Dict:
        """Get metadata for a specific model"""
        return self._metadata.get(name, {})
    
    def is_model_loaded(self, name: str) -> bool:
        """Check if a model is loaded"""
        if name not in self._models:
            return False
        return self._models[name].is_loaded
    
    def preload_models(self, model_names: Optional[List[str]] = None) -> None:
        """
        Preload models into memory
        
        Args:
            model_names: List of models to preload. If None, preload all.
        """
        if model_names is None:
            model_names = self.list_models()
        
        logger.info(f"Preloading {len(model_names)} models...")
        for name in model_names:
            try:
                self.get_model(name)
            except Exception as e:
                logger.error(f"Failed to preload model {name}: {e}")
    
    def unload_model(self, name: str) -> None:
        """Unload a model from memory (if supported by model)"""
        if name in self._models:
            model = self._models[name]
            if hasattr(model, 'unload'):
                model.unload()
                logger.info(f"Unloaded model: {name}")

    def readiness_check(self) -> Dict[str, Any]:
        """Check readiness of prediction service for load balancing"""
        loaded_models = []
        failed_models = []

        for model_name in self.list_models():
            try:
                model = self.get_model(model_name)  # This will load if needed
                if model.is_loaded:
                    loaded_models.append(model_name)
                else:
                    failed_models.append(model_name)
            except Exception as e:
                failed_models.append(f"{model_name}: {str(e)}")

        # Ready if at least one model is loaded
        ready = len(loaded_models) > 0

        return {
            "ready": ready,
            "loaded_models": loaded_models,
            "failed_models": failed_models,
            "total_models": len(self.list_models()),
            "cache_usage": f"{len(self._models)}/{config.model_cache_size}"
        }

    def health_check(self) -> Dict[str, Any]:
        """Check health of prediction service"""
        loaded_models = []
        failed_models = []

        for model_name in self.list_models():
            try:
                if self.is_model_loaded(model_name):
                    loaded_models.append(model_name)
                else:
                    failed_models.append(model_name)
            except Exception as e:
                failed_models.append(f"{model_name}: {str(e)}")

        return {
            "status": "healthy" if len(failed_models) == 0 else "degraded",
            "loaded_models": loaded_models,
            "failed_models": failed_models,
            "total_models": len(self.list_models())
        }

    def cleanup_all_models(self):
        """Cleanup all loaded models for shutdown"""
        logger.info("Cleaning up all models...")
        for model_name in list(self._models.keys()):
            try:
                model = self._models[model_name]
                if model.is_loaded:
                    if hasattr(model, 'cleanup'):
                        model.cleanup()
                    elif hasattr(model, 'model') and hasattr(model.model, 'cpu'):
                        model.model.cpu()

                    if hasattr(model, 'device') and 'cuda' in str(model.device):
                        import torch
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"Error during cleanup for model {model_name}: {e}")

        # Clear CUDA cache one more time
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("All models cleaned up")

# Global registry instance
registry = ModelRegistry()