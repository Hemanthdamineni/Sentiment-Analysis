import time
import signal
import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.app.routers.apis import router as public_router
from backend.core.config.settings import config
from backend.services.prediction_service import PredictionService
from backend.core.registry.model_registry import registry

app = FastAPI(title="Smartwatch Sentiment API v5")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static/results", StaticFiles(directory="backend/models/sentiment/results"), name="results")
app.include_router(public_router)

# Global service instance - created but models loaded in startup
prediction_service = PredictionService()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup with controlled model loading"""
    print(f"Starting ML API on {config.api_host}:{config.api_port}")
    print(f"Models directory: {config.models_dir}")
    print(f"Device: {config.device}")
    print(f"Model cache limit: {config.model_cache_size}")
    print(f"Circuit breaker threshold: {config.circuit_failure_threshold}")
    print(f"Circuit breaker cooldown: {config.circuit_cooldown_seconds}s")

    try:
        # Initialize and load models safely during startup
        prediction_service.initialize_models()

        health = prediction_service.health_check()
        print(f"Service health: {health['status']}")
        print(f"Total models: {health['total_models']}")
        print(f"Loaded models: {len(health['loaded_models'])}")

        if health['status'] == 'degraded':
            print(f"WARNING: {len(health['failed_models'])} models failed to load")

    except Exception as e:
        print(f"FATAL: Startup failed - {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown with resource cleanup"""
    print("Shutting down ML API...")
    try:
        # Shutdown prediction service
        await prediction_service.shutdown()

        # Cleanup all models
        registry.cleanup_all_models()

        print("ML API shutdown complete")
    except Exception as e:
        print(f"Error during shutdown: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint - process is running"""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint - models are loaded and available"""
    return registry.readiness_check()

@app.get("/metrics")
async def metrics():
    """Lightweight runtime metrics endpoint"""
    return prediction_service.get_metrics()

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    print(f"Received signal {signum}, initiating graceful shutdown...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)