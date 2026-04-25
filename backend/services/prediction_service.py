import logging
import os
import asyncio
import time
import threading
import sys
from typing import Dict, Any, List, Tuple
from collections import defaultdict, deque
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.core.registry.model_registry import registry
from backend.core.config.settings import config
from backend.models.transformer_model import TransformerModel
from backend.models.classical_model import ClassicalModel

logger = logging.getLogger(__name__)

class PredictionService:
    """Service layer for handling predictions with batching, metrics, and circuit breaking"""

    def __init__(self):
        # Don't initialize models at import time
        self._batch_queue = defaultdict(list)  # model_name -> list of (timestamp, text, future)
        self._batch_lock = asyncio.Lock()
        self._batch_size = config.inference_batch_size
        self._batch_window_ms = config.batch_window_ms
        self._batch_processor_task = None

        # Thread-safe metrics with locks
        self._metrics_lock = threading.RLock()
        self._metrics = {
            'request_count': 0,
            'error_count': 0,
            'latency_samples': deque(maxlen=1000),  # Keep last 1000 samples
            'batch_sizes': deque(maxlen=1000),
            'model_errors': defaultdict(int)
        }

        # Thread-safe circuit breaker state with hysteresis
        self._circuit_lock = threading.RLock()
        self._circuit_breakers = {}  # model_name -> {'failures': int, 'last_failure': float, 'state': str}
        self._circuit_failure_threshold = config.circuit_failure_threshold
        self._circuit_cooldown_seconds = config.circuit_cooldown_seconds
        self._circuit_recovery_threshold = 2  # Successes needed for recovery

        # Shutdown flag for task cleanup
        self._shutdown = False

    def initialize_models(self) -> None:
        """Initialize and register all available models - called from startup"""
        # Register transformer model
        transformer_model = TransformerModel()
        registry.register("transformer", transformer_model, {
            "description": "DistilBERT transformer model",
            "class_names": ["Negative", "Neutral", "Positive"]
        })

        # Register classical models
        classical_dir = os.path.join(config.models_dir, "classical")
        if os.path.exists(classical_dir):
            for filename in os.listdir(classical_dir):
                if filename.endswith('.pkl'):
                    algorithm_name = filename[:-4]  # Remove .pkl extension
                    classical_model = ClassicalModel(algorithm_name)
                    registry.register(algorithm_name, classical_model, {
                        "description": f"Classical ML algorithm: {algorithm_name}",
                        "algorithm": algorithm_name
                    })

        # Preload models with bounded cache
        registry.preload_models()

    async def predict(self, model_type: str, text: str, algorithm: str = "") -> Dict[str, Any]:
        """
        Make prediction using specified model with batching and circuit breaking

        Args:
            model_type: "transformer" or "classical"
            text: Input text to predict
            algorithm: Algorithm name (required for classical models)

        Returns:
            Prediction result dictionary
        """
        start_time = time.time()

        # Thread-safe metrics increment
        with self._metrics_lock:
            self._metrics['request_count'] += 1

        model_name = ""  # Initialize to avoid unbound variable

        try:
            # Determine model name
            if model_type == "transformer":
                model_name = "transformer"
            elif model_type == "classical":
                if not algorithm or algorithm == "":
                    raise ValueError("Algorithm name required for classical models")
                model_name = algorithm
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Check circuit breaker with thread safety
            if self._is_circuit_open(model_name):
                with self._metrics_lock:
                    self._metrics['error_count'] += 1
                return {
                    "valid": False,
                    "message": f"Model '{model_name}' temporarily unavailable due to repeated failures"
                }

            # Create future for batch processing
            future = asyncio.Future()

            # Add to batch queue
            async with self._batch_lock:
                if self._shutdown:
                    raise RuntimeError("Service is shutting down")
                self._batch_queue[model_name].append((time.time(), text, future))

                # Start batch processor if not running
                if self._batch_processor_task is None:
                    self._batch_processor_task = asyncio.create_task(self._batch_processor())

            # Wait for batch processing result
            result = await future

            # Record latency on success (thread-safe)
            latency_ms = (time.time() - start_time) * 1000
            with self._metrics_lock:
                self._metrics['latency_samples'].append(latency_ms)

            # Record circuit breaker success (thread-safe)
            self._record_circuit_success(model_name)

            return result

        except Exception as e:
            # Record error metrics (thread-safe)
            with self._metrics_lock:
                self._metrics['error_count'] += 1
                self._metrics['model_errors'][model_name] += 1

            # Trigger circuit breaker check (thread-safe)
            self._trigger_circuit_breaker(model_name)

            logger.error(f"Prediction failed: {e}")
            return {
                "valid": False,
                "message": str(e)
            }

    def _is_circuit_open(self, model_name: str) -> bool:
        """Check if circuit breaker is open for a model (thread-safe)"""
        with self._circuit_lock:
            if model_name not in self._circuit_breakers:
                return False

            breaker = self._circuit_breakers[model_name]
            if breaker['state'] == 'open':
                # Check if cooldown period has passed
                if time.time() - breaker['last_failure'] > self._circuit_cooldown_seconds:
                    # Move to half-open state for gradual recovery
                    breaker['state'] = 'half_open'
                    logger.info(f"Circuit breaker for model '{model_name}' moved to half-open state")
                    return False
                return True
            return False

    def _trigger_circuit_breaker(self, model_name: str):
        """Increment failure count for circuit breaker (thread-safe)"""
        with self._circuit_lock:
            if model_name not in self._circuit_breakers:
                self._circuit_breakers[model_name] = {'failures': 0, 'last_failure': 0, 'state': 'closed', 'successes': 0}

            breaker = self._circuit_breakers[model_name]
            breaker['failures'] += 1
            breaker['last_failure'] = time.time()
            breaker['successes'] = 0  # Reset success count

            if breaker['failures'] >= self._circuit_failure_threshold:
                breaker['state'] = 'open'
                logger.warning(f"Circuit breaker opened for model '{model_name}': {breaker['failures']} failures")

    def _record_circuit_success(self, model_name: str):
        """Record success for circuit breaker hysteresis (thread-safe)"""
        with self._circuit_lock:
            if model_name not in self._circuit_breakers:
                return

            breaker = self._circuit_breakers[model_name]
            breaker['successes'] += 1

            # Gradual recovery: need multiple successes to close circuit
            if breaker['state'] == 'half_open' and breaker['successes'] >= self._circuit_recovery_threshold:
                breaker['state'] = 'closed'
                breaker['failures'] = 0  # Reset failure count
                logger.info(f"Circuit breaker closed for model '{model_name}' after recovery")

    async def _batch_processor(self):
        """Background task to process batches"""
        while True:
            try:
                await asyncio.sleep(self._batch_window_ms / 1000.0)

                async with self._batch_lock:
                    # Check if any batches need processing
                    models_to_process = []
                    for model_name, batch in self._batch_queue.items():
                        if (len(batch) >= self._batch_size or
                            (batch and time.time() - batch[0][0] >= self._batch_window_ms / 1000.0)):
                            models_to_process.append((model_name, batch[:]))
                            self._batch_queue[model_name] = batch[len(batch):]

                    if not models_to_process:
                        continue

                # Process batches outside lock
                for model_name, batch in models_to_process:
                    try:
                        await self._process_batch(model_name, batch)
                    except Exception as e:
                        logger.error(f"Batch processing failed for {model_name}: {e}")
                        # Set error on all futures in this batch
                        for _, _, future in batch:
                            if not future.done():
                                future.set_exception(e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")

        self._batch_processor_task = None

    async def _process_batch(self, model_name: str, batch: List[Tuple[float, str, asyncio.Future]]):
        """Process a single batch of predictions"""
        if not batch:
            return

        # Record batch size for metrics
        self._metrics['batch_sizes'].append(len(batch))

        # Extract texts and futures
        _, texts, futures = zip(*batch)
        texts = list(texts)
        futures = list(futures)

        try:
            # Get model from registry
            model = registry.get_model(model_name)

            # Make batch prediction
            result = model.predict(texts)

            # Set results on all futures
            for i, future in enumerate(futures):
                if not future.done():
                    future.set_result({
                        "valid": True,
                        "sentiment": result["predictions"][i],
                        "confidence": result["confidence"][i] if result["confidence"][i] is not None else None,
                        "model_used": result["model_used"]
                    })

        except Exception as e:
            # Set error on all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    def get_metrics(self) -> Dict[str, Any]:
        """Get lightweight runtime metrics (thread-safe)"""
        with self._metrics_lock:
            latency_samples = list(self._metrics['latency_samples'])
            batch_sizes = list(self._metrics['batch_sizes'])
            request_count = self._metrics['request_count']
            error_count = self._metrics['error_count']
            model_errors = dict(self._metrics['model_errors'])

        with self._circuit_lock:
            circuit_breakers = dict(self._circuit_breakers)

        metrics = {
            'request_count': request_count,
            'error_count': error_count,
            'error_rate': error_count / max(1, request_count),
            'latency_p50': 0,
            'latency_p95': 0,
            'avg_batch_size': 0,
            'model_errors': model_errors,
            'circuit_breakers': circuit_breakers
        }

        if latency_samples:
            latency_samples.sort()
            metrics['latency_p50'] = latency_samples[len(latency_samples) // 2]
            metrics['latency_p95'] = latency_samples[int(len(latency_samples) * 0.95)]

        if batch_sizes:
            metrics['avg_batch_size'] = sum(batch_sizes) / len(batch_sizes)

        return metrics

    async def shutdown(self):
        """Graceful shutdown with task cleanup"""
        logger.info("Shutting down prediction service...")
        self._shutdown = True

        # Cancel batch processor task
        if self._batch_processor_task and not self._batch_processor_task.done():
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass

        logger.info("Prediction service shutdown complete")
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models by type"""
        models = registry.list_models()
        classical_models = []
        
        for model_name in models:
            model = registry.get_model(model_name)
            if model.model_type == "classical":
                classical_models.append(model_name)
        
        return {
            "classical_algorithms": sorted(classical_models)
        }
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        try:
            model = registry.get_model(model_name)
            return model.get_metadata()
        except KeyError:
            return {"error": f"Model '{model_name}' not found"}
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of prediction service"""
        loaded_models = []
        failed_models = []
        
        for model_name in registry.list_models():
            try:
                if registry.is_model_loaded(model_name):
                    loaded_models.append(model_name)
                else:
                    failed_models.append(model_name)
            except Exception as e:
                failed_models.append(f"{model_name}: {str(e)}")
        
        return {
            "status": "healthy" if len(failed_models) == 0 else "degraded",
            "loaded_models": loaded_models,
            "failed_models": failed_models,
            "total_models": len(registry.list_models())
        }

# Global service instance
prediction_service = PredictionService()