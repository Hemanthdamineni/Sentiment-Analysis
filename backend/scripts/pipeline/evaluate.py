import os
import torch
import json
import pickle
from transformers import DistilBertTokenizer
from scripts.data_processing import load_data, create_data_loaders, get_class_names
from scripts.transformers import SentimentClassifier, eval_model
from scripts.evaluation import compute_metrics, plot_confusion_matrix, analyze_misclassifications, save_evaluation_results, plot_model_comparison_bar, plot_top_tfidf_features, plot_sentiment_frequency
from scripts.classical_ml import train_all_classical_models, load_all_classical_models, evaluate_all_classical_models, plot_and_save_classical_rocs, save_wordclouds

def run_evaluate(use_huggingface=True, output_dir="backend/models/sentiment", model_path="backend/models/sentiment", sample_size=None, batch_size=16, max_length=128):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_df, test_df = load_data(use_huggingface=use_huggingface, sample_size=sample_size)
    print("Data loaded for evaluation.")
    _, _, test_loader = create_data_loaders(train_df, test_df, tokenizer, batch_size=batch_size, max_length=max_length)
    model = SentimentClassifier(n_classes=3, dropout_p=0.4).to(device)
    if os.path.exists(os.path.join(model_path, "best_model.pt")):
        model.load_state_dict(torch.load(os.path.join(model_path, "best_model.pt"), map_location=device))
    elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device))
    else:
        return {"error": "model_not_found"}
    print("Evaluating transformer model...")
    test_loss, test_acc, predictions, targets = eval_model(model, test_loader, device)
    print(f"Transformer test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
    class_names = get_class_names()
    metrics = compute_metrics(predictions, targets)
    cm_fig = plot_confusion_matrix(metrics['confusion_matrix'], class_names)
    misclassified = analyze_misclassifications(model, test_loader, device, n_examples=5)
    figures = {'confusion_matrix': cm_fig}
    freq_train = plot_sentiment_frequency(train_df, title='Sentiment Frequency – Train')
    freq_test = plot_sentiment_frequency(test_df, title='Sentiment Frequency – Test')
    freq_train.savefig(os.path.join(output_dir, "results", "sentiment_frequency_train.png"))
    freq_test.savefig(os.path.join(output_dir, "results", "sentiment_frequency_test.png"))
    save_evaluation_results({'test_loss': test_loss, 'test_accuracy': test_acc, 'classification_report': metrics['classification_report'], 'misclassified_examples': misclassified}, figures, os.path.join(output_dir, "results"))
    print(f"Saved evaluation artifacts to {os.path.join(output_dir, 'results')}")
    classical_dir = os.path.join(output_dir, "classical")
    if not os.path.isdir(classical_dir) or not any(p.endswith('.pkl') for p in os.listdir(classical_dir)):
        print("Training classical models for evaluation...")
        train_all_classical_models(train_df, classical_dir)
    models = load_all_classical_models(classical_dir)
    print("Evaluating classical models...")
    classical_results = evaluate_all_classical_models(models, test_df)
    comp = {'transformer': {'accuracy': float(test_acc), 'report': metrics['classification_report']}, 'classical': classical_results}
    results_dir = os.path.join(output_dir, "results")
    with open(os.path.join(results_dir, "comparison.json"), 'w') as f:
        json.dump(comp, f, indent=2)
    bar = plot_model_comparison_bar({'Transformer': {'accuracy': float(test_acc)}, **{name: {'accuracy': m['accuracy']} for name, m in classical_results.items()}}, metric_key='accuracy')
    bar.savefig(os.path.join(results_dir, 'model_comparison_accuracy.png'))
    logreg_path = os.path.join(classical_dir, 'logistic_regression.pkl')
    if os.path.exists(logreg_path):
        with open(logreg_path, 'rb') as f: obj = pickle.load(f)
        if isinstance(obj, dict): vec = obj['vec']; logreg = obj['model']
        else: vec, logreg = obj
        top = plot_top_tfidf_features(vec, logreg)
        top.savefig(os.path.join(results_dir, 'top_tfidf_features_logreg.png'))
    plot_and_save_classical_rocs(models, test_df, results_dir)
    save_wordclouds(train_df, results_dir)
    return {"test_acc": float(test_acc)}
