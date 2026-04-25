import torch
import torch.nn as nn
import os
import logging
import sys
from typing import Dict, Any, List
from transformers import DistilBertTokenizer, DistilBertModel
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.core.models.base import BaseModel
from backend.core.config.settings import config
from backend.scripts.data_processing import get_class_names

logger = logging.getLogger(__name__)

class TransformerModel(BaseModel):
    """Transformer model wrapper implementing BaseModel interface"""
    
    def __init__(self, model_name: str = "transformer"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False
        self._setup_device()
    
    def _setup_device(self) -> None:
        """Setup computation device"""
        try:
            import torch_directml
            _dml = torch_directml.device()
        except Exception:
            _dml = None
        
        if config.device == "auto":
            self.device = torch.device('cuda') if torch.cuda.is_available() else (_dml if _dml is not None else torch.device('cpu'))
        else:
            self.device = torch.device(config.device)
        
        logger.info(f"Using device: {self.device}")
    
    def predict(self, texts: List[str]) -> Dict[str, Any]:
        """
        Make predictions using transformer model
        
        Args:
            texts: List of input texts
            
        Returns:
            Dict with predictions and metadata
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        self.model.eval()
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=config.max_sequence_length,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                
                # Get confidence scores
                probs = torch.softmax(outputs, dim=1)
                confidence = torch.max(probs, dim=1)[0].item()
                
                predictions.append(preds.item())
                confidences.append(confidence)
        
        # Convert to class names
        class_names = get_class_names()
        predicted_classes = [class_names[p] for p in predictions]
        
        return {
            "predictions": predicted_classes,
            "prediction_ids": predictions,
            "confidence": confidences,
            "model_used": f"DistilBERT ({self.model_name})",
            "model_type": "transformer"
        }
    
    def load(self, model_path: str) -> None:
        """
        Load transformer model and tokenizer
        
        Args:
            model_path: Path to model artifacts
        """
        try:
            # Load tokenizer
            if os.path.exists(os.path.join(model_path, "tokenizer.json")):
                self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            else:
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
            # Load model
            self.model = SentimentClassifier(n_classes=3, dropout_p=config.dropout).to(self.device)
            
            # Try different model file names
            model_files = ["best_model.pt", "pytorch_model.bin"]
            model_loaded = False
            
            for model_file in model_files:
                full_path = os.path.join(model_path, model_file)
                if os.path.exists(full_path):
                    self.model.load_state_dict(torch.load(full_path, map_location=self.device))
                    logger.info(f"Loaded model from {full_path}")
                    model_loaded = True
                    break
            
            if not model_loaded:
                raise FileNotFoundError(f"No model file found in {model_path}")
            
            self._loaded = True
            logger.info(f"Transformer model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")
            raise
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata"""
        return {
            "model_name": self.model_name,
            "model_type": "transformer",
            "architecture": "DistilBERT",
            "num_classes": 3,
            "max_sequence_length": config.max_sequence_length,
            "device": str(self.device),
            "is_loaded": self._loaded
        }
    
    @property
    def model_type(self) -> str:
        return "transformer"
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded

# Import the original SentimentClassifier
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=3, dropout_p=0.4):
        super(SentimentClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(self.distilbert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        output = self.drop(pooled_output)
        return self.out(output)