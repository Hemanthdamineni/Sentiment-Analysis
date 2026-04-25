from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator
import os, json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.services.prediction_service import prediction_service

router = APIRouter()

class PredictIn(BaseModel):
    text: str = Field(..., max_length=10000, description="Text to analyze (max 10K characters)")
    model_type: str = Field(..., description="Model type: transformer or classical")
    algorithm: str | None = Field(None, description="Algorithm name (required for classical models)")

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('text cannot be empty')
        if len(v.strip()) < 3:
            raise ValueError('text must be at least 3 characters long')
        return v.strip()

    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ['transformer', 'classical']:
            raise ValueError('model_type must be either "transformer" or "classical"')
        return v

    @validator('algorithm')
    def validate_algorithm(cls, v, values):
        if values.get('model_type') == 'classical' and (not v or not v.strip()):
            raise ValueError('algorithm is required when model_type is "classical"')
        return v.strip() if v else None

@router.get("/models")
def list_models():
    """List all available models"""
    return prediction_service.list_available_models()

@router.get("/comparison")
def get_comparison():
    comp_path = os.path.join("backend", "models", "sentiment", "results", "comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

@router.get("/wordclouds")
def get_wordclouds():
    base = os.path.join("backend", "models", "sentiment", "results")

    # Only pick these three exact files
    target_files = [
        "wordcloud_positive.png",
        "wordcloud_negative.png",
        "wordcloud_neutral.png"
    ]

    # Filter only those that actually exist in the folder
    available = [
        f"http://localhost:8000/static/{name}"
        for name in target_files
        if os.path.isfile(os.path.join(base, name))
    ]

    return {"wordclouds": available}

@router.get("/Images")
def get_Images():
    base = os.path.join("backend", "models", "sentiment", "results")

    # Only pick these three exact files
    target_files = [
        "confusion_matrix.png",
        "model_comparison_accuracy.png",
        "top_tfidf_features_logreg.png"
    ]

    # Filter only those that actually exist in the folder
    available = [
        f"http://localhost:8000/static/{name}"
        for name in target_files
        if os.path.isfile(os.path.join(base, name))
    ]

    return {"Images": available}

@router.post("/predict")
async def predict(body: PredictIn):
    """Make prediction using specified model type and algorithm"""
    try:
        result = await prediction_service.predict(
            model_type=body.model_type,
            text=body.text,
            algorithm=body.algorithm or ""
        )

        # Check if prediction failed due to circuit breaker
        if not result.get("valid") and "temporarily unavailable" in result.get("message", ""):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result["message"]
            )

        return result

    except ValueError as e:
        # Pydantic validation errors
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        # Other errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
