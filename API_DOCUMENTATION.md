# ML Sentiment Analysis Service - OpenAPI Documentation

This document provides detailed API specifications for the production ML inference service.

## Overview

The ML Sentiment Analysis Service provides real-time sentiment classification for Amazon product reviews using both transformer and classical ML models. The service is designed for high-throughput production deployment with comprehensive monitoring and fault tolerance.

## Authentication

Currently, no authentication is required. For production deployments, consider adding API key authentication.

## Rate Limiting

- Default: 1000 requests per minute
- Configurable via `RATE_LIMIT` environment variable
- Applied per IP address

## Error Handling

The API uses standard HTTP status codes:

- `200`: Success
- `422`: Validation error (invalid input)
- `503`: Service unavailable (circuit breaker tripped)
- `500`: Internal server error

## Endpoints

### GET /health

**Health check endpoint for load balancers and monitoring systems.**

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1640995200.123
}
```

### GET /ready

**Readiness check endpoint for Kubernetes and deployment systems.**

**Response:**
```json
{
  "ready": true,
  "loaded_models": ["transformer", "logistic_regression"],
  "failed_models": [],
  "total_models": 9,
  "cache_usage": "3/5"
}
```

### GET /metrics

**Runtime metrics for monitoring and debugging.**

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

### GET /models

**List all available classical ML algorithms.**

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

### POST /predict

**Main sentiment prediction endpoint.**

**Request Body:**
```json
{
  "text": "This product exceeded my expectations!",
  "model_type": "transformer",
  "algorithm": null
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text to analyze (max 10,000 chars) |
| `model_type` | string | Yes | `"transformer"` or `"classical"` |
| `algorithm` | string | No* | Required when `model_type` is `"classical"` |

**Success Response (200):**
```json
{
  "valid": true,
  "sentiment": "Positive",
  "confidence": 0.87,
  "model_used": "DistilBERT (transformer)"
}
```

**Validation Error (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "ensure this value has at least 3 characters",
      "type": "value_error.const"
    }
  ]
}
```

**Circuit Breaker Error (503):**
```json
{
  "detail": "Model 'transformer' temporarily unavailable due to repeated failures"
}
```

### GET /comparison

**Model performance comparison data from training.**

**Response:**
```json
{
  "transformer": {
    "accuracy": 0.7017,
    "report": {
      "Negative": {"precision": 0.72, "recall": 0.68, "f1-score": 0.70},
      "Neutral": {"precision": 0.45, "recall": 0.42, "f1-score": 0.43},
      "Positive": {"precision": 0.78, "recall": 0.81, "f1-score": 0.79},
      "accuracy": 0.7017,
      "macro avg": {"precision": 0.65, "recall": 0.64, "f1-score": 0.64},
      "weighted avg": {"precision": 0.71, "recall": 0.70, "f1-score": 0.70}
    }
  },
  "classical": {
    "multinomial_nb": {
      "accuracy": 0.6537,
      "precision": 0.68,
      "recall": 0.65,
      "f1": 0.66
    }
    // ... other algorithms
  }
}
```

### GET /wordclouds

**URLs for sentiment-specific word cloud visualizations.**

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

### GET /Images

**URLs for performance visualization images.**

**Response:**
```json
{
  "Images": [
    "http://localhost:8000/static/confusion_matrix.png",
    "http://localhost:8000/static/model_comparison_accuracy.png",
    "http://localhost:8000/static/top_tfidf_features_logreg.png"
  ]
}
```

## Data Types

### Sentiment Classes
- **Negative**: Rating 1-2
- **Neutral**: Rating 3
- **Positive**: Rating 4-5

### Model Types
- **transformer**: DistilBERT-based deep learning model
- **classical**: Traditional ML algorithms (Logistic Regression, SVM, etc.)

### Confidence Scores
- Range: 0.0 to 1.0
- Higher values indicate stronger sentiment classification
- Only available for probabilistic models (not all classical algorithms)

## Rate Limits

- **Per IP**: 1000 requests/minute (configurable)
- **Per Model Type**: No additional limits
- **Burst Handling**: Queue-based batching absorbs traffic spikes

## Timeouts

- **Request Timeout**: 60 seconds (configurable)
- **Batch Window**: 100ms (configurable)
- **Circuit Breaker Cooldown**: 60 seconds (configurable)

## Circuit Breaker States

- **Closed**: Normal operation
- **Open**: Model disabled after repeated failures
- **Half-Open**: Testing recovery with limited traffic

## Monitoring

### Key Metrics

- **Request Latency**: P50/P95 response times
- **Throughput**: Requests per second
- **Error Rate**: Failed requests percentage
- **Batch Efficiency**: Average batch size utilization
- **Circuit Breaker Status**: Model availability states
- **Memory Usage**: GPU/CPU resource consumption
- **Cache Hit Rate**: Model loading efficiency

### Health Checks

Use `/health` for liveness and `/ready` for readiness in load balancers and orchestrators.

## Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| 422 | Validation Error | Check input format and constraints |
| 503 | Service Unavailable | Model temporarily disabled, try later |
| 500 | Internal Error | Contact system administrator |

## Examples

### Basic Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing product, highly recommend!", "model_type": "transformer"}'
```

### Classical ML Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "It works as expected, nothing special.",
    "model_type": "classical",
    "algorithm": "logistic_regression"
  }'
```

### Monitoring
```bash
# Health check
curl http://localhost:8000/health

# System metrics
curl http://localhost:8000/metrics

# Model availability
curl http://localhost:8000/ready
```

## Version History

- **v1.0**: Initial production release
  - Transformer and classical ML models
  - Batch processing and circuit breakers
  - Comprehensive monitoring and health checks

---

For additional support, please refer to the main README.md or create an issue on GitHub.