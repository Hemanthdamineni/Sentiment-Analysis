from fastapi import APIRouter
from pydantic import BaseModel
import os, json, pickle
from ...scripts.data_processing import get_class_names

router = APIRouter()

class PredictIn(BaseModel):
    text: str
    model_type: str
    algorithm: str | None = None

@router.get("/models")
def list_models():
    classical_dir = os.path.join("backend", "models", "sentiment", "classical")
    algos = []
    if os.path.isdir(classical_dir):
        for fname in os.listdir(classical_dir):
            if fname.endswith('.pkl'):
                algos.append(fname[:-4])
    return {"classical_algorithms": sorted(algos)}

@router.get("/comparison")
def get_comparison():
    comp_path = os.path.join("backend", "models", "sentiment", "results", "comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

@router.get("/wordclouds")
def get_wordclouds():
    base = os.path.join("backend", "app", "models", "sentiment", "results")

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
    base = os.path.join("backend", "app", "models", "sentiment", "results")

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
def predict(body: PredictIn):
    if not body.text or not body.text.strip():
        return {"valid": False, "message": "Empty text"}
    class_names = get_class_names()
    if body.model_type == 'transformer':
        import torch
        from transformers import DistilBertTokenizer
        from ...scripts.transformers import SentimentClassifier
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SentimentClassifier(n_classes=3)
        model_path = os.path.join("backend", "models", "sentiment")
        if os.path.exists(os.path.join(model_path, "best_model.pt")):
            model.load_state_dict(torch.load(os.path.join(model_path, "best_model.pt"), map_location=device))
        elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device))
        else:
            return {"valid": False, "message": "model_not_found"}
        model = model.to(device)
        tok = DistilBertTokenizer.from_pretrained(model_path)
        enc = tok.encode_plus(body.text, add_special_tokens=True, max_length=128, return_token_type_ids=False, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
        return {"valid": True, "sentiment": class_names[preds.item()]}
    elif body.model_type == 'classical':
        if not body.algorithm:
            return {"valid": False, "message": "algorithm_required"}
        path = os.path.join("backend", "models","sentiment","classical", f"{body.algorithm}.pkl")
        if not os.path.exists(path):
            return {"valid": False, "message": "model_not_found"}
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            vec = obj.get('vec'); svd = obj.get('svd'); clf = obj.get('model')
        else:
            vec, clf = obj; svd = None
        X = vec.transform([body.text])
        if svd is not None:
            X = svd.transform(X)
        pred = clf.predict(X)[0]
        conf = None
        if hasattr(clf, 'predict_proba'):
            try:
                proba = clf.predict_proba(X)[0]; conf = float(max(proba))
            except Exception: pass
        return {"valid": True, "sentiment": class_names[pred], "confidence": conf}
    else:
        return {"valid": False, "message": "unknown_model_type"}
