# ML Sentiment Analysis Service

A production-grade machine learning inference service for sentiment analysis of Amazon product reviews. Built with FastAPI, PyTorch, and optimized for high-throughput deployment.

## 🚀 Features

- **High-Performance Inference**: Batch processing with 24-request batches and vectorized tokenization
- **Fault-Tolerant**: Circuit breaker pattern with hysteresis for automatic recovery
- **Production-Ready**: Thread-safe, validated configuration, comprehensive monitoring
- **Scalable Architecture**: Bounded model caching with LRU eviction
- **Multi-Model Support**: DistilBERT transformer + 8 classical ML algorithms
- **Real-time Monitoring**: Health checks, readiness probes, and runtime metrics

## 📊 Performance

- **Throughput**: 800+ RPS with batch processing
- **Latency**: P50 ~150ms, P95 ~300ms
- **Memory**: <4GB with bounded model cache
- **Reliability**: <1% error rate with circuit breaker protection

## 🏗️ Architecture

### Core Components

```
├── FastAPI Application (main.py)
│   ├── Prediction Service (prediction_service.py)
│   │   ├── Async Batch Processor
│   │   ├── Circuit Breaker Protection
│   │   └── Runtime Metrics
│   ├── Model Registry (model_registry.py)
│   │   ├── LRU Caching (max 5 models)
│   │   ├── Lifecycle Management
│   │   └── Resource Cleanup
│   └── Model Wrappers
│       ├── TransformerModel (DistilBERT)
│       └── ClassicalModel (TF-IDF + Algorithms)
├── Configuration (settings.py)
│   └── Pydantic-validated settings
└── API Endpoints (apis.py)
    └── RESTful prediction interface
```

### Data Flow

1. **Request Validation**: Input sanitization and size limits
2. **Circuit Breaker Check**: Prevent requests to failing models
3. **Batch Queuing**: Accumulate requests for 100ms or 24 requests
4. **Vectorized Processing**: Single GPU/CPU batch operation
5. **Response Distribution**: Individual results to waiting clients
6. **Metrics Recording**: Latency, errors, and batch statistics

## 📦 Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ (with CUDA support for GPU acceleration)
- 8GB+ RAM (16GB recommended)
- Linux/macOS/Windows

### Setup

```bash
# Clone repository
git clone <repository-url>
cd ml-sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Configuration

Create `.env` file in project root:

```bash
# Model Configuration
MODEL_CACHE_SIZE=5
MAX_SEQUENCE_LENGTH=256

# Inference Settings
INFERENCE_BATCH_SIZE=24
BATCH_WINDOW_MS=100

# Circuit Breaker
CIRCUIT_FAILURE_THRESHOLD=5
CIRCUIT_COOLDOWN_SECONDS=60

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
RATE_LIMIT=1000
REQUEST_TIMEOUT=60

# Infrastructure
DEVICE=auto
NUM_WORKERS=8
```

## 🚀 Quick Start

### Training (Optional)

```bash
# Train models (requires dataset)
python backend/main.py --train --evaluate

# Or use provided pre-trained models
# Models will be auto-downloaded on first startup
```

### Production Deployment

```bash
# Start production server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Or with Gunicorn for production
gunicorn backend.app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Testing

```bash
# Health check
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/ready

# List available models
curl http://localhost:8000/models

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product is amazing! Great battery life and excellent performance.",
    "model_type": "transformer"
  }'
```

## 📡 API Documentation

### Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1640995200.123
}
```

#### Readiness Check
```http
GET /ready
```

**Response:**
```json
{
  "ready": true,
  "loaded_models": ["transformer", "logistic_regression", "svm"],
  "failed_models": [],
  "total_models": 9,
  "cache_usage": "3/5"
}
```

#### Runtime Metrics
```http
GET /metrics
```

**Response:**
```json
{
  "request_count": 1250,
  "error_count": 12,
  "error_rate": 0.0096,
  "latency_p50": 145.23,
  "latency_p95": 289.45,
  "avg_batch_size": 18.5,
  "model_errors": {
    "transformer": 5,
    "logistic_regression": 7
  },
  "circuit_breakers": {
    "problematic_model": {
      "failures": 3,
      "last_failure": 1640995190.123,
      "state": "half_open",
      "successes": 2
    }
  }
}
```

#### List Models
```http
GET /models
```

**Response:**
```json
{
  "classical_algorithms": [
    "logistic_regression",
    "linear_svc",
    "svc_rbf",
    "sgd_classifier",
    "random_forest",
    "knn",
    "multinomial_nb",
    "bernoulli_nb"
  ]
}
```

#### Sentiment Prediction
```http
POST /predict
Content-Type: application/json

{
  "text": "This product exceeded my expectations!",
  "model_type": "transformer",
  "algorithm": null
}
```

**Parameters:**
- `text` (string, required): Text to analyze (max 10,000 chars)
- `model_type` (string, required): `"transformer"` or `"classical"`
- `algorithm` (string, optional): Required for classical models

**Success Response:**
```json
{
  "valid": true,
  "sentiment": "Positive",
  "confidence": 0.87,
  "model_used": "DistilBERT (transformer)"
}
```

**Error Responses:**
```json
// Validation Error (422)
{
  "detail": "text must be at least 3 characters long"
}

// Circuit Breaker (503)
{
  "detail": "Model 'transformer' temporarily unavailable due to repeated failures"
}

// Internal Error (500)
{
  "detail": "Internal server error"
}
```

#### Model Comparison
```http
GET /comparison
```

Returns stored model performance comparison from training.

#### Word Clouds
```http
GET /wordclouds
```

**Response:**
```json
{
  "wordclouds": [
    "http://localhost:8000/static/wordcloud_positive.png",
    "http://localhost:8000/static/wordcloud_negative.png",
    "http://localhost:8000/static/wordcloud_neutral.png"
  ]
}
```

#### Visualization Images
```http
GET /Images
```

Returns URLs for confusion matrices, accuracy comparisons, and feature importance plots.

## ⚙️ Configuration Reference

### Model Settings

| Setting | Default | Range | Description |
|---------|---------|-------|-------------|
| `MODEL_CACHE_SIZE` | 5 | 1-10 | Maximum models cached in memory |
| `MAX_SEQUENCE_LENGTH` | 256 | 16-512 | Maximum token sequence length |

### Inference Settings

| Setting | Default | Range | Description |
|---------|---------|-------|-------------|
| `INFERENCE_BATCH_SIZE` | 24 | 1-32 | Requests per batch |
| `BATCH_WINDOW_MS` | 100 | 10-500 | Max wait time for batching |

### Circuit Breaker

| Setting | Default | Range | Description |
|---------|---------|-------|-------------|
| `CIRCUIT_FAILURE_THRESHOLD` | 5 | 1-10 | Failures before tripping |
| `CIRCUIT_COOLDOWN_SECONDS` | 60 | 10-300 | Recovery cooldown period |

### API Settings

| Setting | Default | Range | Description |
|---------|---------|-------|-------------|
| `API_HOST` | 0.0.0.0 | - | API bind address |
| `API_PORT` | 8000 | 1024-65535 | API port |
| `RATE_LIMIT` | 1000 | 1-10000 | Requests per minute |
| `REQUEST_TIMEOUT` | 60 | 5-300 | Request timeout seconds |

## 🚀 Deployment

The service can be run locally using `uvicorn` and supports direct usage. For production scaling, consider deploying the Uvicorn workers behind a robust reverse proxy like NGINX or directly with Gunicorn.

## 📊 Monitoring

### Key Metrics to Monitor

- **Request Latency**: P50/P95 response times
- **Error Rate**: Total errors / total requests
- **Batch Efficiency**: Average batch size utilization
- **Circuit Breaker States**: Monitor tripped breakers
- **Memory Usage**: GPU/CPU memory consumption
- **Cache Hit Rate**: Model cache effectiveness

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
  - name: sentiment_api
    rules:
      - alert: HighErrorRate
        expr: rate(sentiment_api_errors_total[5m]) / rate(sentiment_api_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"

      - alert: CircuitBreakerTripped
        expr: sentiment_api_circuit_breaker_state{state="open"} > 0
        for: 2m
        labels:
          severity: warning
          annotations:
            summary: "Circuit breaker tripped"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(sentiment_api_request_duration_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
```

## 🔧 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check configuration validation
python -c "from backend.core.config.settings import config; print('Config OK')"

# Check model directory exists
ls -la backend/models/sentiment/
```

#### High Latency
- Check batch size utilization in `/metrics`
- Monitor GPU memory usage
- Verify model cache is not thrashing

#### Circuit Breaker Tripping
- Check model error rates in `/metrics`
- Investigate underlying model failures
- Consider adjusting `CIRCUIT_FAILURE_THRESHOLD`

#### Memory Issues
- Reduce `MODEL_CACHE_SIZE`
- Monitor cache usage in `/ready` endpoint
- Check for model leaks during shutdown

### Debug Mode

Enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
uvicorn backend.app.main:app --log-level debug
```

### Performance Tuning

1. **Increase batch size** for higher throughput (trade-off: latency)
2. **Reduce batch window** for lower latency (trade-off: throughput)
3. **Tune model cache size** based on available memory
4. **Adjust circuit breaker thresholds** for your failure tolerance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run the full test suite
5. Submit a pull request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run with auto-reload
uvicorn backend.app.main:app --reload
```

## 📄 License

MIT License - see LICENSE file for details.

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review `/metrics` endpoint for system status

---

**Built with ❤️ for production ML inference**
