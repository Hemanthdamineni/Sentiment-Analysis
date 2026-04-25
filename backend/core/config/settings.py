from pydantic import validator, Field
from pydantic_settings import BaseSettings
from typing import Optional
import os

class MLConfig(BaseSettings):
    # Model settings with validation (Production-optimized)
    model_cache_size: int = Field(5, ge=1, le=10, description="Number of models to cache in memory")
    max_sequence_length: int = Field(256, ge=16, le=512, description="Maximum token sequence length")
    batch_size: int = Field(32, ge=1, le=64, description="Training batch size")
    models_dir: str = Field("backend/models/sentiment", description="Directory containing model artifacts")

    # API settings (Production-tuned)
    api_host: str = Field("0.0.0.0", description="API host address")
    api_port: int = Field(8000, ge=1024, le=65535, description="API port number")
    rate_limit: int = Field(1000, ge=1, le=10000, description="Rate limit per minute")
    request_timeout: int = Field(60, ge=5, le=300, description="Request timeout in seconds")

    # Infrastructure (Production-scaled)
    device: str = Field("auto", description="Computation device (auto/cuda/cpu)")
    num_workers: int = Field(8, ge=1, le=16, description="Number of worker processes")

    # Training defaults (for backward compatibility)
    epochs: int = Field(10, ge=1, le=100, description="Number of training epochs")
    learning_rate: float = Field(2e-5, ge=1e-6, le=1e-2, description="Learning rate")
    weight_decay: float = Field(0.01, ge=0, le=0.1, description="Weight decay")
    dropout: float = Field(0.5, ge=0, le=0.9, description="Dropout rate")
    warmup_ratio: float = Field(0.06, ge=0, le=0.2, description="Warmup ratio")
    use_class_weights: bool = Field(True, description="Use class weights")
    use_weighted_sampler: bool = Field(True, description="Use weighted sampler")
    use_huggingface: bool = Field(True, description="Use HuggingFace datasets")

    # Batching configuration with validation (Production-throughput)
    inference_batch_size: int = Field(24, ge=1, le=32, description="Inference batch size")
    batch_window_ms: int = Field(100, ge=10, le=500, description="Batch time window in milliseconds")

    # Circuit breaker configuration (Production-stable)
    circuit_failure_threshold: int = Field(5, ge=1, le=10, description="Circuit breaker failure threshold")
    circuit_cooldown_seconds: int = Field(60, ge=10, le=300, description="Circuit breaker cooldown in seconds")

    @validator('device')
    def validate_device(cls, v):
        if v not in ['auto', 'cuda', 'cpu']:
            raise ValueError('device must be one of: auto, cuda, cpu')
        return v

    @validator('models_dir')
    def validate_models_dir(cls, v):
        if not v or not v.strip():
            raise ValueError('models_dir cannot be empty')
        return v.strip()

    class Config:
        env_file = ".env"
        env_prefix = "ML_"

# Global config instance with validation
config = MLConfig()